from __future__ import annotations

import gc, inspect, os, time,psutil, torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ...utils.common import info as log_info

def detect_model_type(name: str) -> str:
    lowered = name.lower()
    for marker in ("qwen", "llama", "mistral", "deepseek", "yi", "glm", "baichuan"):
        if marker in lowered:
            return marker
    return "unknown"


@dataclass
class ModelInfo:
    name: str
    path: str
    model_type: str
    quantization: str | None = None
    dtype: str = "float16"
    parameters: str = "unknown"
    context_length: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    prompt_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class MemoryUsage:
    gpu_allocated_mb: float = 0
    gpu_reserved_mb: float = 0
    gpu_total_mb: float = 0
    gpu_free_mb: float = 0
    process_rss_mb: float = 0
    system_available_mb: float = 0
    details: dict[str, Any] | None = None


class baseInference(ABC):
    """推理框架抽象基类"""

    _SAMPLING_KEY_MAP: dict[str, str] = {}
    _EXTRA_SAMPLING_KEYS: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_info: ModelInfo | None = None
        self._effective_load: dict[str, Any] = {}
        self._sleeping = False
        self._sleep_capable = False

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def effective_load(self) -> dict[str, Any]:
        return dict(self._effective_load)

    @property
    def supported_modalities(self) -> set[str]:
        """模型支持的输入模态，默认纯文本，多模态子类重写覆盖。"""
        return {"text"}

    @abstractmethod
    def load(
        self,
        model_path: str,
        *,
        config: dict[str, Any] | None = None,
        draft: str | None = None,
        lora: str | None = None,
        **kwargs: Any,
    ) -> "baseInference":
        pass

    @abstractmethod
    def _unload_engine(self) -> None:
        pass

    def unload(self) -> None:
        self._unload_engine()

    @abstractmethod
    def count_tokens(self, text_or_messages: str | list[dict[str, Any]]) -> int:
        """精确统计文本或上下文消息的 Token 数量。"""
        pass

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    def supports_sleep_to_ram(self) -> bool:
        return self._sleep_capable

    def sleep_holds_ram(self) -> bool:
        return True

    def sleep_to_ram(self) -> bool:
        if self._model is None or self._sleeping or not self.supports_sleep_to_ram():
            return self._sleeping
        start = time.perf_counter()
        try:
            self._engine_sleep()
        except Exception as exc:
            log_info("引擎休眠失败", type(self).__name__, exc)
            return False
        self._sleeping = True
        self.release_cache()
        log_info("引擎已休眠至 RAM", type(self).__name__, f"{round(time.perf_counter() - start, 2)}s")
        return True

    def wake(self) -> None:
        if not self._sleeping:
            return
        start = time.perf_counter()
        try:
            self._engine_wake()
        except Exception as exc:
            log_info("引擎唤醒失败", type(self).__name__, exc)
            raise
        self._sleeping = False
        log_info("引擎已唤醒", type(self).__name__, f"{round(time.perf_counter() - start, 2)}s")

    def _engine_sleep(self) -> None:
        raise NotImplementedError

    def _engine_wake(self) -> None:
        raise NotImplementedError

    def _mark_unloaded(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_info = None
        self._effective_load = {}
        self._sleeping = False
        self._sleep_capable = False

    @staticmethod
    def _accepted_engine_kwargs(engine: Any, values: dict[str, Any], excluded: set[str] | None = None) -> dict[str, Any]:
        try:
            params = inspect.signature(engine).parameters
        except (TypeError, ValueError):
            return {}
        ignored = excluded or set()
        return {k: v for k, v in values.items() if k in params and k not in ignored}

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        usage = MemoryUsage()
        if torch.cuda.is_available():
            dev = self._get_gpu_device_index()
            usage.gpu_allocated_mb = round(torch.cuda.memory_allocated(dev) / (1024 ** 2), 1)
            usage.gpu_reserved_mb = round(torch.cuda.memory_reserved(dev) / (1024 ** 2), 1)
            usage.gpu_total_mb = round(torch.cuda.get_device_properties(dev).total_memory / (1024 ** 2), 1)
            usage.gpu_free_mb = round(usage.gpu_total_mb - usage.gpu_reserved_mb, 1)

        usage.process_rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)
        usage.system_available_mb = round(psutil.virtual_memory().available / (1024 ** 2), 1)

        if verbose and self._model_info:
            usage.details = {"model_name": self._model_info.name, "loaded": self.is_loaded}
        return usage

    def release_cache(self) -> MemoryUsage:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return self.memory_usage()

    def _get_gpu_device_index(self) -> int:
        return 0

    def _extract_model_info(self, model_path: str, **kwargs: Any) -> ModelInfo:
        path = Path(model_path)
        name = path.name if path.exists() else model_path.split("/")[-1]
        return ModelInfo(
            name=name, path=model_path, model_type=detect_model_type(name),
            quantization=kwargs.get("quantization"), dtype=kwargs.get("dtype", "float16"),
        )

    def __enter__(self) -> "baseInference":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    def generate(self, prompt: str, *,sampling: dict[str, Any] | None = None, system_prompt: str = "",messages: list[dict[str, Any]] | None = None, **kwargs: Any,) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        norm_messages = messages or ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
        rendered = self._build_chat_prompt(norm_messages)
        sampling_params = self._build_sampling(sampling or {}, kwargs)

        start = time.perf_counter()
        text, gen_tokens, prompt_tokens = self._run_engine(rendered, sampling_params)
        elapsed = time.perf_counter() - start

        max_tokens = sampling_params.get("max_tokens", 512)
        return GenerationResult(
            text=text, tokens_generated=gen_tokens, time_seconds=elapsed,
            tokens_per_second=gen_tokens / elapsed if elapsed > 0 else 0,
            prompt_tokens=prompt_tokens,
            finish_reason="length" if gen_tokens >= max_tokens else "stop",
        )

    def stream_generate(self,prompt: str, *,sampling: dict[str, Any] | None = None,system_prompt: str = "",messages: list[dict[str, Any]] | None = None,**kwargs: Any,) -> Iterator[str]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        norm_messages = messages or ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
        rendered = self._build_chat_prompt(norm_messages)
        sampling_params = self._build_sampling(sampling or {}, kwargs)
        yield from self._run_engine_stream(rendered, sampling_params)

    def _run_engine(self, rendered_prompt: str, sampling: dict[str, Any]) -> tuple[str, int, int]:
        raise NotImplementedError

    def _run_engine_stream(self, rendered_prompt: str, sampling: dict[str, Any]) -> Iterator[str]:
        raise NotImplementedError

    def _get_tokenizer(self) -> Any:
        return self._tokenizer

    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        try:
            tokenizer = self._get_tokenizer()
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)

    def _build_sampling(self, sampling: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = {
            "max_tokens": sampling.get("max_new_tokens", sampling.get("max_tokens", 512)),
            "temperature": max(sampling.get("temperature", 0.3), 0.01),
            "top_p": sampling.get("top_p", 0.95),
            "top_k": sampling.get("top_k", 50),
            "repetition_penalty": sampling.get("repetition_penalty", 1.05),
            "stop": sampling.get("stop_sequences"),
        }
        for k in self._EXTRA_SAMPLING_KEYS:
            if k in sampling:
                merged[k] = sampling[k]
            elif k in extra:
                merged[k] = extra[k]

        for old_k, new_k in self._SAMPLING_KEY_MAP.items():
            if old_k in merged:
                merged[new_k] = merged.pop(old_k)
        return merged