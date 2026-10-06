from __future__ import annotations

import gc,torch
from typing import Any, Iterator

from ...utils.common import info as log_info
from .baseInference import MemoryUsage, baseInference
import sglang as sgl


class sglang(baseInference):
    """SGLang 推理引擎"""

    _SAMPLING_KEY_MAP = {"max_tokens": "max_new_tokens"}
    _EXTRA_SAMPLING_KEYS = ("min_p", "frequency_penalty", "presence_penalty", "regex", "json_schema")

    def load(
        self,
        model_path: str,
        *,
        config: dict[str, Any] | None = None,
        draft: str | None = None,
        lora: str | None = None,
        **kwargs: Any,
    ) -> "sglang":
        if sgl is None:
            raise RuntimeError("SGLang 模型需要安装 sglang")

        cfg = {**(config or {}), **kwargs}
        engine_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "trust_remote_code": cfg.get("trust_remote_code", True),
            "tp_size": cfg.get("tensor_parallel_size", 1),
            "dtype": cfg.get("dtype", "float16"),
            "mem_fraction_static": cfg.get("gpu_memory_utilization", 0.9),
        }
        if cfg.get("quantization"):
            engine_kwargs["quantization"] = cfg["quantization"]
        if cfg.get("max_model_len"):
            engine_kwargs["context_length"] = cfg["max_model_len"]

        if draft:
            engine_kwargs.update({
                "speculative_algorithm": cfg.get("speculative_algorithm", "EAGLE"),
                "speculative_draft_model_path": draft,
                "speculative_num_steps": cfg.get("speculative_num_steps", 5),
            })
        if lora:
            engine_kwargs["lora_paths"] = [lora]

        enable_saver = cfg.get("enable_memory_saver", True)
        if enable_saver:
            engine_kwargs["enable_memory_saver"] = True

        try:
            self._model = sgl.Engine(**engine_kwargs)
            self._sleep_capable = bool(enable_saver)
        except TypeError:
            engine_kwargs.pop("enable_memory_saver", None)
            self._model = sgl.Engine(**engine_kwargs)
            self._sleep_capable = False

        self._model_info = self._extract_model_info(model_path, **cfg)
        self._effective_load = {"engine": "sglang", **engine_kwargs}
        return self

    def count_tokens(self, text_or_messages: str | list[dict[str, Any]]) -> int:
        tokenizer = self._get_tokenizer()
        if isinstance(text_or_messages, str):
            return len(tokenizer.encode(text_or_messages))
        try:
            rendered = tokenizer.apply_chat_template(text_or_messages, tokenize=False, add_generation_prompt=True)
            return len(tokenizer.encode(rendered))
        except Exception:
            flat = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in text_or_messages)
            return len(tokenizer.encode(flat))

    def _get_tokenizer(self) -> Any:
        return self._model.tokenizer_manager.tokenizer

    def _engine_sleep(self) -> None:
        self._model.release_memory_occupation()

    def _engine_wake(self) -> None:
        self._model.resume_memory_occupation()

    def _run_engine(self, rendered_prompt: str, sampling: dict[str, Any]) -> tuple[str, int, int]:
        result = self._model.generate(prompt=rendered_prompt, sampling_params=sampling)
        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        return (result["text"], int(meta.get("completion_tokens", 0)), int(meta.get("prompt_tokens", 0)))

    def _run_engine_stream(self, rendered_prompt: str, sampling: dict[str, Any]) -> Iterator[str]:
        for chunk in self._model.generate_stream(prompt=rendered_prompt, sampling_params=sampling):
            text = chunk.get("text", "")
            if text:
                yield text

    def _unload_engine(self) -> None:
        if self._model is not None:
            try:
                self._model.shutdown()
            except Exception:
                pass
        self._mark_unloaded()
        self.release_cache()