from __future__ import annotations

import gc
from typing import Any, Iterator
import torch

from ...utils.common import info as log_info
from ..chatDataFilter import flatten_messages
from .baseInference import baseInference

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    LLM = None
    SamplingParams = None
    LoRARequest = None


class vllm(baseInference):
    """vLLM 推理框架"""

    _EXTRA_SAMPLING_KEYS = ("min_p", "seed", "frequency_penalty", "presence_penalty", "logit_bias")

    def load(
        self,
        model_path: str,
        *,
        config: dict[str, Any] | None = None,
        draft: str | None = None,
        lora: str | None = None,
        **kwargs: Any,
    ) -> "vllm":
        if LLM is None:
            raise RuntimeError("vLLM 模型需要安装 vllm")

        cfg = {**(config or {}), **kwargs}
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": cfg.get("trust_remote_code", True),
            "tensor_parallel_size": cfg.get("tensor_parallel_size", 1),
            "dtype": cfg.get("dtype", "float16"),
            "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.9),
        }
        if cfg.get("quantization"):
            llm_kwargs["quantization"] = cfg["quantization"]
        if cfg.get("max_model_len"):
            llm_kwargs["max_model_len"] = cfg["max_model_len"]

        if draft:
            llm_kwargs["speculative_model"] = draft
            llm_kwargs["num_speculative_tokens"] = cfg.get("num_speculative_tokens", 5)
        if lora:
            llm_kwargs.update({"enable_lora": True, "max_loras": 1})
            self._lora_request = LoRARequest("configured-lora", 1, lora)
        else:
            self._lora_request = None

        enable_sleep = cfg.get("enable_sleep_mode", True)
        if enable_sleep:
            llm_kwargs["enable_sleep_mode"] = True

        try:
            self._model = LLM(**llm_kwargs)
            self._sleep_capable = bool(enable_sleep)
        except (TypeError, ValueError):
            llm_kwargs.pop("enable_sleep_mode", None)
            self._model = LLM(**llm_kwargs)
            self._sleep_capable = False

        self._model_info = self._extract_model_info(model_path, **cfg)
        self._effective_load = {"engine": "vllm", **llm_kwargs}
        return self

    def count_tokens(self, text_or_messages: str | list[dict[str, Any]]) -> int:
        tokenizer = self._get_tokenizer()
        if isinstance(text_or_messages, str):
            return len(tokenizer.encode(text_or_messages))
        try:
            rendered = tokenizer.apply_chat_template(text_or_messages, tokenize=False, add_generation_prompt=True)
            return len(tokenizer.encode(rendered))
        except Exception:
            return len(tokenizer.encode(flatten_messages(text_or_messages)))

    def _get_tokenizer(self) -> Any:
        return self._model.get_tokenizer()

    def _engine_sleep(self) -> None:
        self._model.sleep(level=1)

    def _engine_wake(self) -> None:
        self._model.wake_up()

    def _response(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[str, int, int, list[dict[str, Any]], str]:
        rendered = self._build_chat_prompt(messages)
        params = SamplingParams(**sampling)
        extra = {"lora_request": self._lora_request} if self._lora_request else {}
        output = self._model.generate([rendered], params, **extra)[0]
        choice = output.outputs[0]
        text = choice.text
        tokens = len(choice.token_ids)
        prompt_tokens = len(output.prompt_token_ids)
        finish_reason = choice.finish_reason or "stop"
        return text, tokens, prompt_tokens, [], finish_reason

    def _unload_engine(self) -> None:
        self._mark_unloaded()
        self._lora_request = None
        self.release_cache()