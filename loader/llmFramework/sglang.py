"""
sglang.py
SGLang 推理框架

特点: RadixAttention 前缀缓存 + 高吞吐调度,适合多轮对话/共享前缀场景。
"""

import gc
from typing import Any

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
    _EXTRA_SAMPLING_KEYS = ("min_p", "frequency_penalty", "presence_penalty")

    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "sglang":
        if sgl is None:
            raise RuntimeError("SGLang 模型需要安装 sglang")

        engine_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "trust_remote_code": trust_remote_code,
            "tp_size": tensor_parallel_size,
            "dtype": dtype,
            "mem_fraction_static": kwargs.get("gpu_memory_utilization", 0.9),
        }
        if quantization:
            engine_kwargs["quantization"] = quantization
        if max_model_len:
            engine_kwargs["context_length"] = max_model_len

        self._model = sgl.Engine(**engine_kwargs)
        self._model_info = self._extract_model_info(model_path, quantization=quantization, dtype=dtype)
        self._tool_parser = kwargs.get("tool_parser")

        if max_model_len:
            self._model_info.context_length = max_model_len

        self._effective_load = {
            "engine": "sglang", "dtype": dtype,
            "context_length": self._model_info.context_length,
            "tensor_parallel": tensor_parallel_size,
            "gpu_memory_utilization": engine_kwargs["mem_fraction_static"],
            "trust_remote_code": trust_remote_code,
            "tool_parser": self._tool_parser,
        }
        return self

    def _get_tokenizer(self) -> Any:
        return self._model.tokenizer_manager.tokenizer

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

    def unload(self) -> None:
        """卸载模型,释放显存"""
        if self._model is not None:
            try:
                self._model.shutdown()
            except Exception:
                pass
        self._model = None
        self._model_info = None
        self._effective_load = {}
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
