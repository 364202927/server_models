from __future__ import annotations

from typing import Any
from ...utils.common import info as log_info
from .baseInference import RawOutput, baseInference

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    LLM = None
    SamplingParams = None
    LoRARequest = None


class vllm(baseInference):
    """vLLM 推理框架"""

    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "vllm":
        if LLM is None:
            raise RuntimeError("vLLM 模型需要安装 vllm")

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": load_cfg.get("trust_remote_code", True),
            "tensor_parallel_size": load_cfg.get("tensor_parallel", 1),
            "dtype": load_cfg.get("dtype", "bfloat16"),
            "gpu_memory_utilization": load_cfg.get("gpu_memory_utilization", 0.9),
        }

        # 显存策略换算对齐
        context_val = load_cfg.get("context", 0)
        if "context_memory_mb" in load_cfg and load_cfg["context_memory_mb"] > 0:
            llm_kwargs["max_model_len"] = max(2048, int(load_cfg["context_memory_mb"] * 128 // 512 * 512))
        elif context_val and context_val > 0:
            llm_kwargs["max_model_len"] = int(context_val)

        # draft: [0] 主模型，[1] 辅助设置
        draft_list = load_cfg.get("draft") or []
        if isinstance(draft_list, str):
            draft_list = [draft_list]
        if draft_list and len(draft_list) > 0 and draft_list[0]:
            llm_kwargs["speculative_model"] = draft_list[0]
            llm_kwargs["num_speculative_tokens"] = 5

        lora = load_cfg.get("lora")
        if lora:
            llm_kwargs.update({"enable_lora": True, "max_loras": 1})
            self._lora_request = LoRARequest("configured-lora", 1, lora)
        else:
            self._lora_request = None

        enable_sleep = load_cfg.get("enable_sleep_mode", True)
        if enable_sleep:
            llm_kwargs["enable_sleep_mode"] = True

        try:
            self._model = LLM(**llm_kwargs)
            self._sleep_capable = bool(enable_sleep)
        except (TypeError, ValueError):
            llm_kwargs.pop("enable_sleep_mode", None)
            self._model = LLM(**llm_kwargs)
            self._sleep_capable = False

        self._tokenizer = self._model.get_tokenizer()
        # 以引擎实际生效的 max_model_len 为准（未显式配置时由模型 config 决定）
        max_len = self._model.llm_engine.model_config.max_model_len
        self._effective_load = {"engine": "vllm", **llm_kwargs, "context_length": int(max_len)}
        return self

    def _engine_sleep(self) -> None:
        self._model.sleep(level=1)

    def _engine_wake(self) -> None:
        self._model.wake_up()

    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> RawOutput:
        sampling = dict(gen_cfg)
        rendered = self._build_chat_prompt(messages, sampling.pop("tools", None))
        sampling.pop("tool_choice", None)
        params = SamplingParams(**sampling)

        extra = {"lora_request": self._lora_request} if self._lora_request else {}
        output = self._model.generate([rendered], params, **extra)[0]
        choice = output.outputs[0]
        cached = getattr(output, "num_cached_tokens", None)  # 前缀缓存(APC)命中的 prompt token 数
        return RawOutput(
            text=choice.text,
            tokens=len(choice.token_ids),
            prompt_tokens=len(output.prompt_token_ids),
            finish_reason=choice.finish_reason or "stop",
            cached_tokens=-1 if cached is None else int(cached),
        )

    def _unload_engine(self) -> None:
        self._mark_unloaded()
        self._lora_request = None
        self.release_cache()