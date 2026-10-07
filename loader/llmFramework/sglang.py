from __future__ import annotations

from typing import Any
from ...utils.common import info as log_info
from ..chatDataFilter import flatten_messages
from .baseInference import baseInference

try:
    import sglang as sgl
except ImportError:
    sgl = None


class sglang(baseInference):
    """SGLang 推理引擎"""

    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "sglang":
        if sgl is None:
            raise RuntimeError("SGLang 模型需要安装 sglang")

        engine_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "trust_remote_code": load_cfg.get("trust_remote_code", True),
            "tp_size": load_cfg.get("tensor_parallel", 1),
            "dtype": load_cfg.get("dtype", "bfloat16"),
            "mem_fraction_static": load_cfg.get("gpu_memory_utilization", 0.9),
        }

        context_val = load_cfg.get("context", 0)
        if "context_memory_mb" in load_cfg and load_cfg["context_memory_mb"] > 0:
            engine_kwargs["context_length"] = max(2048, int(load_cfg["context_memory_mb"] * 128 // 512 * 512))
        elif context_val and context_val > 0:
            engine_kwargs["context_length"] = int(context_val)

        # draft: [0] 主模型，[1] 辅助路径/算法设置
        draft_list = load_cfg.get("draft") or []
        if isinstance(draft_list, str):
            draft_list = [draft_list]
        if draft_list and len(draft_list) > 0 and draft_list[0]:
            engine_kwargs.update({
                "speculative_algorithm": "EAGLE",
                "speculative_draft_model_path": draft_list[0],
                "speculative_num_steps": 5,
            })

        lora = load_cfg.get("lora")
        if lora:
            engine_kwargs["lora_paths"] = [lora]

        enable_saver = load_cfg.get("enable_memory_saver", True)
        if enable_saver:
            engine_kwargs["enable_memory_saver"] = True

        try:
            self._model = sgl.Engine(**engine_kwargs)
            self._sleep_capable = bool(enable_saver)
        except TypeError:
            engine_kwargs.pop("enable_memory_saver", None)
            self._model = sgl.Engine(**engine_kwargs)
            self._sleep_capable = False

        self._effective_load = {"engine": "sglang", **engine_kwargs}
        return self

    def count_tokens(self, text_or_messages: str | list[dict[str, Any]]) -> int:
        tokenizer = self._model.tokenizer_manager.tokenizer
        if isinstance(text_or_messages, str):
            return len(tokenizer.encode(text_or_messages))
        try:
            rendered = tokenizer.apply_chat_template(text_or_messages, tokenize=False, add_generation_prompt=True)
            return len(tokenizer.encode(rendered))
        except Exception:
            return len(tokenizer.encode(flatten_messages(text_or_messages)))

    def _engine_sleep(self) -> None:
        self._model.release_memory_occupation()

    def _engine_wake(self) -> None:
        self._model.resume_memory_occupation()

    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> tuple[str, int, int, list[dict[str, Any]], str]:
        rendered = self._build_chat_prompt(messages)
        sampling = dict(gen_cfg)
        sampling.pop("tools", None)
        sampling.pop("tool_choice", None)
        if "max_tokens" in sampling:
            sampling["max_new_tokens"] = sampling.pop("max_tokens")

        result = self._model.generate(prompt=rendered, sampling_params=sampling)
        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        tokens = int(meta.get("completion_tokens", 0))
        prompt_tokens = int(meta.get("prompt_tokens", 0))
        finish_reason = meta.get("finish_reason", "stop")
        return result["text"], tokens, prompt_tokens, [], finish_reason

    def _unload_engine(self) -> None:
        if self._model is not None:
            try:
                self._model.shutdown()
            except Exception:
                pass
        self._mark_unloaded()
        self.release_cache()