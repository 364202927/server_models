"""
vllm.py
vLLM 推理框架

特点: 高性能推理,支持批量生成和张量并行
适用: 生产环境,需要高吞吐量场景
"""

import gc
import time
from typing import Any

from .baseInference import GenerationResult, MemoryUsage, baseInference

# vllm 自带 torch 依赖;在模块级尝试一次导入,缺失时统一置 None,
# 由 load() 在入口处报出清晰的 RuntimeError,而不是让 ImportError 直接冒出来。
try:
    import torch
    from vllm import LLM, SamplingParams
except ImportError:
    torch = LLM = SamplingParams = None


class vllm(baseInference):
    """vLLM 推理框架 - 高性能批量推理"""

    _EXTRA_SAMPLING_KEYS = ("min_p", "seed", "frequency_penalty", "presence_penalty", "logit_bias")

    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "vllm":
        if LLM is None:
            raise RuntimeError("vLLM 模型需要安装 vllm")

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
        }
        if quantization:
            llm_kwargs["quantization"] = quantization
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len

        self._model = LLM(**llm_kwargs)
        self._model_info = self._extract_model_info(model_path, quantization=quantization, dtype=dtype)
        self._tool_parser = kwargs.get("tool_parser")

        # 上下文长度是附加信息,读取失败不应阻断模型加载。
        try:
            self._apply_context_length()
        except Exception:
            pass

        self._effective_load = {
            "engine": "vllm", "dtype": dtype,
            "context_length": self._model_info.context_length if self._model_info else max_model_len,
            "tensor_parallel": tensor_parallel_size,
            "gpu_memory_utilization": llm_kwargs["gpu_memory_utilization"],
            "trust_remote_code": trust_remote_code,
            "tool_parser": self._tool_parser,
        }
        return self

    def _apply_context_length(self) -> None:
        """从 vLLM 引擎配置回填上下文长度;调用方用一次 try/except 包裹,读取失败就跳过。"""
        config = self._model.llm_engine.model_config
        if hasattr(config, "max_model_len"):
            self._model_info.context_length = config.max_model_len

    def _get_tokenizer(self) -> Any:
        return self._model.get_tokenizer()

    def _run_engine(self, rendered_prompt: str, sampling: dict[str, Any]) -> tuple[str, int, int]:
        sampling_params = SamplingParams(**sampling)
        outputs = self._model.generate([rendered_prompt], sampling_params)
        output = outputs[0]
        return (output.outputs[0].text, len(output.outputs[0].token_ids), len(output.prompt_token_ids))

    def generate_batch(self, prompts: list[str], *, max_new_tokens: int = 512, temperature: float = 0.3,
                       top_p: float = 0.95, **kwargs) -> list[GenerationResult]:
        """批量生成 - vLLM的核心优势,显著提升吞吐量。"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=max(temperature, 0.01),
                                         top_p=top_p)
        start_time = time.perf_counter()
        outputs = self._model.generate(prompts, sampling_params)
        total_time = time.perf_counter() - start_time

        # 按比例分配时间到各个输出
        per_output_time = total_time / len(outputs) if outputs else 0
        return [GenerationResult(
            text=output.outputs[0].text,
            tokens_generated=len(output.outputs[0].token_ids),
            time_seconds=per_output_time,
            tokens_per_second=len(output.outputs[0].token_ids) / per_output_time if per_output_time > 0 else 0,
            prompt_tokens=len(output.prompt_token_ids),
        ) for output in outputs]

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        """vLLM特有: 追加引擎级别的GPU/KV cache利用率"""
        usage = super().memory_usage(verbose)
        if verbose and self.is_loaded:
            usage.details = usage.details or {}
            try:
                engine = self._model.llm_engine
                usage.details["gpu_memory_utilization"] = getattr(
                    engine.model_config, "gpu_memory_utilization", "N/A")
                if hasattr(engine, "scheduler"):
                    stats = self._kv_cache_stats(engine)
                    if stats:
                        usage.details.update(stats)
            except Exception:
                pass
        return usage

    @staticmethod
    def _kv_cache_stats(engine: Any) -> dict[str, Any] | None:
        """遍历 vLLM 引擎的 scheduler(s),取第一个暴露了 block_manager 统计的 KV cache 块信息。"""
        schedulers = engine.scheduler if isinstance(engine.scheduler, list) else [engine.scheduler]
        for scheduler in schedulers:
            block_mgr = getattr(scheduler, "block_manager", None)
            if block_mgr is not None and hasattr(block_mgr, "get_num_free_gpu_blocks"):
                total = getattr(block_mgr, "get_num_total_gpu_blocks", lambda: "N/A")
                return {"kv_cache_free_blocks": block_mgr.get_num_free_gpu_blocks(),
                        "kv_cache_total_blocks": total()}
        return None

    def release_cache(self) -> MemoryUsage:
        """vLLM特有: 触发引擎级KV cache回收"""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return self.memory_usage()

    def unload(self) -> None:
        """卸载模型,释放显存"""
        self._model = None
        self._model_info = None
        self._effective_load = {}
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
