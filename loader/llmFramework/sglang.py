"""
sglang.py
SGLang 推理框架

特点: RadixAttention 前缀缓存 + 高吞吐调度,适合多轮对话/共享前缀场景。
"""

import gc
from typing import Any

from ...utils.common import info as log_info
from .baseInference import MemoryUsage, baseInference

# sglang 自带 torch 依赖;在模块级尝试一次导入,缺失时统一置 None,
# 由 load() 在入口处报出清晰的 RuntimeError,而不是让 ImportError 直接冒出来。
try:
    import torch
    import sglang as sgl
except ImportError:
    torch = sgl = None


class sglang(baseInference):
    """SGLang 推理框架"""

    # sglang 的采样参数用 max_new_tokens,不是 vllm 的 max_tokens。
    _SAMPLING_KEY_MAP = {"max_tokens": "max_new_tokens", "repetition_penalty": "repetition_penalty"}
    _EXTRA_SAMPLING_KEYS = ("min_p", "frequency_penalty", "presence_penalty", "regex", "json_schema")

    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "sglang":
        if sgl is None:
            raise RuntimeError("SGLang 模型需要安装 sglang")

        draft, mtp, lora = kwargs.pop("draft", None), kwargs.pop("mtp", False), kwargs.pop("lora", None)
        if draft and mtp:
            raise ValueError("draft 与 mtp 不能同时启用")
        engine_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "trust_remote_code": trust_remote_code,
            "tp_size": tensor_parallel_size,
            "dtype": dtype,
            "mem_fraction_static": (kwargs.get("gpu_memory_utilization")
                                    if kwargs.get("gpu_memory_utilization") is not None else 0.9),
        }
        if quantization:
            engine_kwargs["quantization"] = quantization
        if max_model_len:
            engine_kwargs["context_length"] = max_model_len

        feature_kwargs: dict[str, Any] = {}
        if draft:
            feature_kwargs = {
                "speculative_algorithm": kwargs.get("speculative_algorithm") or "EAGLE",
                "speculative_draft_model_path": draft,
                "speculative_num_steps": (kwargs.get("speculative_num_steps")
                                           if kwargs.get("speculative_num_steps") is not None else 5),
            }
            log_info("启用SGLang Draft", draft)
        elif mtp:
            feature_kwargs = {
                "speculative_algorithm": kwargs.get("speculative_algorithm") or "EAGLE",
                "speculative_num_steps": (kwargs.get("speculative_num_steps")
                                           if kwargs.get("speculative_num_steps") is not None else 1),
            }
            log_info("启用SGLang MTP")
        if lora:
            feature_kwargs["lora_paths"] = [lora]
            if kwargs.get("max_loras_per_batch") is not None:
                feature_kwargs["max_loras_per_batch"] = kwargs["max_loras_per_batch"]
            log_info("启用SGLang LoRA", lora)
        enable_memory_saver = kwargs.get("enable_memory_saver", True)
        optional_kwargs = self._accepted_engine_kwargs(
            sgl.Engine, kwargs, {"model_path", "model", "trust_remote_code", "tp_size", "dtype",
                                 "context_length", "quantization", "mem_fraction_static",
                                 "gpu_memory_utilization", "enable_memory_saver", "enable_sleep_mode",
                                 "tool_parser"})
        engine_kwargs.update(optional_kwargs)
        # 顶层 draft/mtp/lora 是功能开关，派生出的构造参数优先于同名可选字段。
        engine_kwargs.update(feature_kwargs)
        if enable_memory_saver:
            engine_kwargs["enable_memory_saver"] = True
        try:
            self._model = sgl.Engine(**engine_kwargs)
            self._sleep_capable = bool(enable_memory_saver)
        except TypeError as exc:
            if not enable_memory_saver:
                raise
            # 旧版本 SGLang 不认显存让渡参数时退回普通加载。
            log_info("SGLang显存让渡不可用，按普通模式加载", type(exc).__name__, exc)
            engine_kwargs.pop("enable_memory_saver", None)
            self._model = sgl.Engine(**engine_kwargs)
            self._sleep_capable = False
        self._model_info = self._extract_model_info(model_path, quantization=quantization, dtype=dtype)
        self._tool_parser = kwargs.get("tool_parser")

        if max_model_len:
            self._model_info.context_length = max_model_len

        effective_load = dict(optional_kwargs)
        effective_load.update({
            "engine": "sglang", "dtype": dtype,
            "context_length": self._model_info.context_length,
            "tensor_parallel": tensor_parallel_size,
            "gpu_memory_utilization": engine_kwargs["mem_fraction_static"],
            "quantization": quantization,
            "enable_memory_saver": self._sleep_capable,
            "trust_remote_code": trust_remote_code,
            "tool_parser": self._tool_parser,
        })
        effective_load.update(feature_kwargs)
        self._effective_load = effective_load
        return self

    def _get_tokenizer(self) -> Any:
        return self._model.tokenizer_manager.tokenizer

    def _engine_sleep(self) -> None:
        # SGLang 内部有 is_fully_idle 断言，必须在引擎空闲时调用；
        # ModelsMgr 的 runtime.active 已保证同一时刻没有在跑的请求。
        self._model.release_memory_occupation()

    def _engine_wake(self) -> None:
        self._model.resume_memory_occupation()

    def _run_engine(self, rendered_prompt: str, sampling: dict[str, Any]) -> tuple[str, int, int]:
        result = self._model.generate(prompt=rendered_prompt, sampling_params=sampling)
        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        return (result["text"], int(meta.get("completion_tokens", 0)), int(meta.get("prompt_tokens", 0)))

    def release_cache(self) -> MemoryUsage:
        """SGLang特有: RadixAttention 的前缀缓存不经 torch 分配器统计,这里仅回收 Python 侧临时对象。"""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return self.memory_usage()

    def _unload_engine(self) -> None:
        """卸载模型,释放显存"""
        if self._model is not None:
            try:
                self._model.shutdown()
            except Exception:
                pass
        self._mark_unloaded()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
