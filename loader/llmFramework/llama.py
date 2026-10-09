from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ...utils.hardware import detect_gpu
from ...utils.common import error
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
    """llama-cpp-python GGUF 推理引擎"""

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

    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "llama":
        if Llama is None:
            raise RuntimeError("GGUF 模型需要安装 llama-cpp-python")

        source = self._resolve_gguf_file(model_path)
        gpu_layers = int(load_cfg.get("gpu_offload_layers", -1))
        if gpu_layers != 0 and detect_gpu() and _gpu_offload_supported() is False:
            raise RuntimeError("当前 llama-cpp-python 未启用 CUDA，请安装 CUDA 构建版本。")

        # 计算并对齐 context 长度
        calculated_n_ctx = load_cfg.get("context", 0)
        if calculated_n_ctx <= llama._FALLBACK_N_CTX:
            calculated_n_ctx = llama._FALLBACK_N_CTX

        print("~~~~模型加载:ctx~~~~~~",calculated_n_ctx)
        llm_kwargs: dict[str, Any] = {
            "model_path": str(source),
            "n_gpu_layers": gpu_layers,
            "n_ctx": calculated_n_ctx,
            "n_batch": int(load_cfg.get("batch_size", 512)),
            "verbose": False,
            "use_mlock": bool(load_cfg.get("use_mlock", True)),
        }
        if "flash_attention" in load_cfg:
            llm_kwargs["flash_attn"] = bool(load_cfg["flash_attention"])

        lora = load_cfg.get("lora")
        if lora:
            llm_kwargs["lora_path"] = str(lora)

        # draft 处理：[0] 为主草稿模型，[1] 为辅助推测解码
        draft_list = load_cfg.get("draft") or []
        if isinstance(draft_list, str):
            draft_list = [draft_list]
        self._saved_draft_kwargs = None

        if draft_list and len(draft_list) > 0 and draft_list[0]:
            main_draft = self._resolve_gguf_file(draft_list[0])
            self._saved_draft_kwargs = {
                "model_path": str(main_draft),
                "n_gpu_layers": int(load_cfg.get("gpu_offload_layers", 0)),
            }
            self._draft_model = Llama(**self._saved_draft_kwargs)
            llm_kwargs["draft_model"] = self._draft_model

        self._saved_llm_kwargs = {k: v for k, v in llm_kwargs.items() if k != "draft_model"}
        self._model = Llama(**llm_kwargs)
        self._sleep_capable = True

        self._effective_load = {
            "engine": "llama",
            "context_length": int(self._model.n_ctx()),
            "gpu_offload_layers": llm_kwargs["n_gpu_layers"],
            "batch_size": llm_kwargs["n_batch"],
            "draft": draft_list,
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
        gen_cfg: dict[str, Any],
    ) -> tuple[str, int, int, list[dict[str, Any]], str]:
        cfg = dict(gen_cfg)
        max_tokens = self._clamp_to_context(messages, int(cfg.pop("max_tokens", 512)))
        cfg["max_tokens"] = max_tokens

        # 1. 字段映射对齐（llama-cpp-python 专有键名）
        if "repetition_penalty" in cfg:
            cfg["repeat_penalty"] = cfg.pop("repetition_penalty")
        if "mirostat" in cfg:
            cfg["mirostat_mode"] = cfg.pop("mirostat")

        tools = cfg.pop("tools", None)
        tool_choice = cfg.pop("tool_choice", None)
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
        if tool_choice:
            extra["tool_choice"] = tool_choice

        # 2. 过滤底层不支持的参数，避免 TypeError 崩溃
        import inspect
        try:
            valid_params = inspect.signature(self._model.create_chat_completion).parameters
            filtered_cfg = {k: v for k, v in cfg.items() if k in valid_params}
        except Exception:
            filtered_cfg = cfg

        try:
            result = self._model.create_chat_completion(messages=messages, **extra, **filtered_cfg)
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
            try:
                call_params = inspect.signature(self._model.__call__).parameters
                call_cfg = {k: v for k, v in filtered_cfg.items() if k in call_params}
            except Exception:
                call_cfg = filtered_cfg
            result = self._model(flat, **call_cfg)
            text = str(result["choices"][0].get("text", ""))
            tokens = len(self._model.tokenize(text.encode("utf-8")))
            prompt_tokens = len(self._model.tokenize(flat.encode("utf-8")))
            finish_reason = "length" if tokens >= max_tokens else "stop"
            calls = []

        return text, tokens, prompt_tokens, calls, finish_reason

    def _clamp_to_context(self, messages: list[dict[str, Any]], max_new_tokens: int) -> int:
        n_ctx = self._model.n_ctx()
        prompt_tokens = self.count_tokens(messages)
        available = n_ctx - prompt_tokens - self._CONTEXT_SAFETY_MARGIN
        if available < self._MIN_GENERATION_TOKENS:
            raise ValueError(f"上下文不足: prompt 约 {prompt_tokens} token, 窗口大小 {n_ctx} token")
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

    def _unload_engine(self) -> None:
        self._engine_sleep()
        self._mark_unloaded()
        self._saved_llm_kwargs = None
        self._saved_draft_kwargs = None
        self.release_cache()