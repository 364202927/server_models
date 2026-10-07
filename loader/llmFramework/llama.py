from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Iterator

from ...utils.hardware import detect_gpu
from ...utils.common import info as log_info, error
from ..chatDataFilter import flatten_messages
from .baseInference import baseInference

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


def _gpu_offload_supported() -> bool | None:
    for module_name in ("llama_cpp", "llama_cpp.llama_cpp"):
        try:
            fn = getattr(importlib.import_module(module_name), "llama_supports_gpu_offload", None)
            if callable(fn):
                return bool(fn())
        except (ImportError, AttributeError, TypeError):
            continue
    return None


class llama(baseInference):
    """llama-cpp-python GGUF 推理引擎：只负责物理加载与调用。"""

    _SAMPLING_KEY_MAP = {"repetition_penalty": "repeat_penalty", "mirostat": "mirostat_mode"}
    _EXTRA_SAMPLING_KEYS = (
        "min_p", "seed", "mirostat", "mirostat_eta", "mirostat_tau",
        "repeat_last_n", "tfs_z", "logit_bias", "frequency_penalty", "presence_penalty",
    )
    _FALLBACK_N_CTX = 8192
    _CONTEXT_SAFETY_MARGIN = 32
    _MIN_GENERATION_TOKENS = 16

    def __init__(self) -> None:
        super().__init__()
        self._saved_llm_kwargs: dict[str, Any] | None = None
        self._saved_draft_kwargs: dict[str, Any] | None = None
        self._draft_model = None

    @staticmethod
    def _resolve_gguf_file(model_path: str) -> Path:
        source = Path(model_path)
        if source.is_dir():
            files = sorted(source.glob("*.gguf"))
            if not files:
                raise FileNotFoundError(f"目录中没有 .gguf 文件: {model_path}")
            source = files[0]
        if not source.is_file() or source.suffix.lower() != ".gguf":
            raise ValueError(f"不是有效的 GGUF 文件: {model_path}")
        return source

    def load(
        self,
        model_path: str,
        *,
        config: dict[str, Any] | None = None,
        draft: str | None = None,
        lora: str | None = None,
        **kwargs: Any,
    ) -> "llama":
        if Llama is None:
            raise RuntimeError("GGUF 模型需要安装 llama-cpp-python")

        cfg = {**(config or {}), **kwargs}
        source = self._resolve_gguf_file(model_path)
        gpu_layers = int(cfg.get("gpu_offload_layers", -1))

        if gpu_layers != 0 and detect_gpu() and _gpu_offload_supported() is False:
            raise RuntimeError("当前 llama-cpp-python 未启用 CUDA，请安装 CUDA 构建版本。")

        mtp = cfg.pop("mtp", False)
        if draft and mtp:
            raise ValueError("draft 与 mtp 不能同时启用")

        llm_kwargs: dict[str, Any] = {
            "model_path": str(source),
            "n_gpu_layers": gpu_layers,
            "n_ctx": int(cfg.get("max_model_len") or llama._FALLBACK_N_CTX),
            "n_batch": int(cfg.get("batch_size", 512)),
            "verbose": bool(cfg.get("verbose", False)),
            "use_mlock": True,
        }
        if cfg.get("flash_attention") is not None:
            llm_kwargs["flash_attn"] = bool(cfg["flash_attention"])
        if cfg.get("gpu_split"):
            llm_kwargs["tensor_split"] = cfg["gpu_split"]

        optional_kwargs = self._accepted_engine_kwargs(
            Llama, cfg, {"model_path", "n_gpu_layers", "n_ctx", "n_batch", "verbose", "batch_size", "flash_attention", "gpu_split"}
        )
        llm_kwargs.update(optional_kwargs)

        if lora:
            llm_kwargs["lora_path"] = lora
            llm_kwargs["lora_scale"] = cfg.get("lora_scale", 1.0)
            if cfg.get("lora_base"):
                llm_kwargs["lora_base"] = cfg["lora_base"]

        self._saved_draft_kwargs = None
        if draft:
            draft_source = self._resolve_gguf_file(draft)
            self._saved_draft_kwargs = {
                "model_path": str(draft_source),
                "n_gpu_layers": int(cfg.get("draft_gpu_offload_layers", 0)),
            }
            self._draft_model = Llama(**self._saved_draft_kwargs)
            llm_kwargs["draft_model"] = self._draft_model

        self._saved_llm_kwargs = {k: v for k, v in llm_kwargs.items() if k != "draft_model"}
        self._model = Llama(**llm_kwargs)
        self._sleep_capable = True
        self._model_info = self._extract_model_info(str(source), dtype=cfg.get("dtype", "float16"))
        self._model_info.context_length = int(self._model.n_ctx())

        self._effective_load = {
            "engine": "llama",
            "dtype": self._model_info.dtype,
            "context_length": self._model_info.context_length,
            "gpu_offload_layers": llm_kwargs["n_gpu_layers"],
            "batch_size": llm_kwargs["n_batch"],
            "draft": draft,
            "lora": lora,
        }
        return self

    def count_tokens(self, text_or_messages: str | list[dict[str, Any]]) -> int:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        if isinstance(text_or_messages, str):
            return len(self._model.tokenize(text_or_messages.encode("utf-8")))

        try:
            prompt_str = self._model.chat_template(messages=text_or_messages) if hasattr(self._model, "chat_template") else None
            if not prompt_str:
                prompt_str = flatten_messages(text_or_messages)
        except Exception:
            prompt_str = flatten_messages(text_or_messages)
        return len(self._model.tokenize(prompt_str.encode("utf-8")))

    def _response(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[str, int, int, list[dict[str, Any]], str]:
        """纯粹物理调用：输入经过 baseInference 管道预清洗，此处保持极简。"""
        max_tokens = self._clamp_to_context(messages, sampling.get("max_tokens", 512))
        sampling["max_tokens"] = max_tokens

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
        if kwargs.get("tool_choice"):
            extra["tool_choice"] = kwargs["tool_choice"]

        try:
            result = self._model.create_chat_completion(messages=messages, **extra, **sampling)
            choice = result["choices"][0]
            message = choice["message"]
            text = str(message.get("content") or "")
            calls = message.get("tool_calls") or []
            finish_reason = str(choice.get("finish_reason") or ("tool_calls" if calls else "stop"))
            usage = result.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            tokens = int(usage.get("completion_tokens", 0))
        except Exception as exc:
            error("原生模板推理异常，回退纯文本补全", type(exc).__name__, exc)
            flat = flatten_messages(messages)
            result = self._model(flat, **sampling)
            text = str(result["choices"][0].get("text", ""))
            tokens = len(self._model.tokenize(text.encode("utf-8")))
            prompt_tokens = len(self._model.tokenize(flat.encode("utf-8")))
            finish_reason = "length" if tokens >= max_tokens else "stop"
            calls = []

        return text, tokens, prompt_tokens, calls, finish_reason

    def stream_generate(
        self,
        prompt: str,
        *,
        sampling: dict[str, Any] | None = None,
        system_prompt: str = "",
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        norm_messages = messages or ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
        sampling_params = self._build_sampling(sampling or {}, kwargs)
        max_new_tokens = self._clamp_to_context(norm_messages, sampling_params.get("max_tokens", 512))
        sampling_params["max_tokens"] = max_new_tokens
        sampling_params["stream"] = True

        try:
            stream_iter = self._model.create_chat_completion(messages=norm_messages, **sampling_params)
            for chunk in stream_iter:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        except Exception:
            flat = flatten_messages(norm_messages)
            for chunk in self._model(flat, **sampling_params):
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text

    def _clamp_to_context(self, messages: list[dict[str, Any]], max_new_tokens: int) -> int:
        n_ctx = self._model.n_ctx()
        prompt_tokens = self.count_tokens(messages)
        available = n_ctx - prompt_tokens - self._CONTEXT_SAFETY_MARGIN
        if available < self._MIN_GENERATION_TOKENS:
            raise ValueError(f"上下文不足: prompt约 {prompt_tokens} token, 窗口大小 {n_ctx} token")
        return min(max_new_tokens, available)

    def _engine_sleep(self) -> None:
        model, self._model = self._model, None
        if hasattr(model, "close"):
            model.close()
        draft, self._draft_model = self._draft_model, None
        if draft and hasattr(draft, "close"):
            draft.close()

    def _engine_wake(self) -> None:
        if self._saved_draft_kwargs:
            self._draft_model = Llama(**self._saved_draft_kwargs)
        if self._saved_llm_kwargs:
            kw = dict(self._saved_llm_kwargs)
            if self._draft_model:
                kw["draft_model"] = self._draft_model
            self._model = Llama(**kw)

    def sleep_holds_ram(self) -> bool:
        return False

    def _unload_engine(self) -> None:
        self._engine_sleep()
        self._mark_unloaded()
        self._saved_llm_kwargs = None
        self._saved_draft_kwargs = None
        self.release_cache()