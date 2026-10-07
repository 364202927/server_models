from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import psutil
import torch

from ...utils.common import info as log_info
from ..chatDataFilter import chatDataFilter


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


class baseInference(ABC):
    """推理框架抽象基类：入参规范收敛。"""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._effective_load: dict[str, Any] = {}
        self._sleeping = False
        self._sleep_capable = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def effective_load(self) -> dict[str, Any]:
        return dict(self._effective_load)

    @property
    def supported_modalities(self) -> set[str]:
        return {"text"}

    @abstractmethod
    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "baseInference":
        """仅需 2 个入参：模型路径与完整加载字典。"""
        pass

    @abstractmethod
    def _unload_engine(self) -> None:
        pass

    def unload(self) -> None:
        self._unload_engine()

    @abstractmethod
    def count_tokens(self, text_or_messages: str | list[dict[str, Any]]) -> int:
        pass

    def sleep_to_ram(self) -> bool:
        if self._model is None or self._sleeping or not self._sleep_capable:
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
        self._effective_load = {}
        self._sleeping = False
        self._sleep_capable = False

    def release_cache(self) -> MemoryUsage:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return self.memory_usage()

    def memory_usage(self) -> MemoryUsage:
        usage = MemoryUsage()
        if torch.cuda.is_available():
            usage.gpu_allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
            usage.gpu_reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
            usage.gpu_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
            usage.gpu_free_mb = round(usage.gpu_total_mb - usage.gpu_reserved_mb, 1)

        usage.process_rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)
        usage.system_available_mb = round(psutil.virtual_memory().available / (1024 ** 2), 1)
        return usage

    def generate(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """核心模板方法：仅 2 个入参。"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        cfg = dict(gen_cfg or {})
        system_instruction = str(cfg.pop("system_prompt", "") or "")

        cleaned_messages = chatDataFilter.preprocess_messages(
            messages=messages,
            supported_modalities=self.supported_modalities,
            system_instruction=system_instruction,
        )

        start = time.perf_counter()
        raw_text, gen_tokens, prompt_tokens, calls, finish_reason = self._response(
            messages=cleaned_messages,
            gen_cfg=cfg,
        )
        elapsed = time.perf_counter() - start

        final_text, final_calls, final_reason = chatDataFilter.postprocess_result(
            raw_text, calls, finish_reason
        )

        return GenerationResult(
            text=final_text,
            tokens_generated=gen_tokens,
            time_seconds=elapsed,
            tokens_per_second=gen_tokens / elapsed if elapsed > 0 else 0.0,
            prompt_tokens=prompt_tokens,
            tool_calls=final_calls,
            finish_reason=final_reason,
        )

    @abstractmethod
    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> tuple[str, int, int, list[dict[str, Any]], str]:
        """子类物理调用：仅 2 个入参。"""
        pass

    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        tokenizer = getattr(self, "_tokenizer", None)
        try:
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)