"""
vLLM 模型加载器

特点: 高性能推理，支持批量生成和张量并行
适用: 生产环境，需要高吞吐量场景
"""

import time
from typing import Any

from .base import ModelLoader, GenerationResult, MemoryUsage


class VLLMLoader(ModelLoader):
    """vLLM加载器 - 高性能批量推理"""

    def __init__(self):
        super().__init__()
        self._sampling_params = None

    def load(
        self,
        model_path: str,
        *,
        quantization: str | None = None,
        dtype: str = "float16",
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        **kwargs: Any
    ) -> "VLLMLoader":
        from vllm import LLM

        # 构建vLLM参数
        llm_kwargs: dict = {
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

        # 加载模型
        self._model = LLM(**llm_kwargs)
        self._model_info = self._extract_model_info(model_path, quantization=quantization, dtype=dtype)

        # 获取上下文长度
        try:
            config = self._model.llm_engine.model_config
            if hasattr(config, "max_model_len"):
                self._model_info.context_length = config.max_model_len
        except Exception:
            pass

        return self

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        stop_sequences: list[str] | None = None,
        system_prompt: str = "",
        **kwargs: Any
    ) -> GenerationResult:
        from vllm import SamplingParams

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # vLLM 的 LLM.generate 接受纯文本，没有 chat 接口可用，直接前置 system 段。
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences,
        )

        # 执行生成
        start_time = time.perf_counter()
        outputs = self._model.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - start_time

        # 解析结果
        output = outputs[0]
        tokens_generated = len(output.outputs[0].token_ids)

        return GenerationResult(
            text=output.outputs[0].text,
            tokens_generated=tokens_generated,
            time_seconds=elapsed,
            tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0,
            prompt_tokens=len(output.prompt_token_ids),
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        **kwargs
    ) -> list[GenerationResult]:
        """批量生成 - vLLM的核心优势，显著提升吞吐量"""
        from vllm import SamplingParams

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
        )

        start_time = time.perf_counter()
        outputs = self._model.generate(prompts, sampling_params)
        total_time = time.perf_counter() - start_time

        # 按比例分配时间到各个输出
        per_output_time = total_time / len(outputs) if outputs else 0
        return [
            GenerationResult(
                text=output.outputs[0].text,
                tokens_generated=len(output.outputs[0].token_ids),
                time_seconds=per_output_time,
                tokens_per_second=len(output.outputs[0].token_ids) / per_output_time if per_output_time > 0 else 0,
                prompt_tokens=len(output.prompt_token_ids),
            )
            for output in outputs
        ]

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        """vLLM特有: 追加引擎级别的GPU/KV cache利用率"""
        usage = super().memory_usage(verbose)

        if verbose and self.is_loaded:
            usage.details = usage.details or {}
            try:
                # vLLM引擎暴露的GPU利用率配置
                engine = self._model.llm_engine
                usage.details["gpu_memory_utilization"] = getattr(
                    engine.model_config, "gpu_memory_utilization", "N/A"
                )
                # KV cache块统计
                if hasattr(engine, "scheduler"):
                    for scheduler in (engine.scheduler if isinstance(engine.scheduler, list) else [engine.scheduler]):
                        block_mgr = getattr(scheduler, "block_manager", None)
                        if block_mgr and hasattr(block_mgr, "get_num_free_gpu_blocks"):
                            usage.details["kv_cache_free_blocks"] = block_mgr.get_num_free_gpu_blocks()
                            usage.details["kv_cache_total_blocks"] = getattr(
                                block_mgr, "get_num_total_gpu_blocks", lambda: "N/A"
                            )()
                            break
            except Exception:
                pass

        return usage

    def release_cache(self) -> MemoryUsage:
        """vLLM特有: 触发引擎级KV cache回收"""
        import gc
        import torch

        # vLLM内部管理KV cache块，调用empty_cache释放PyTorch层缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        return self.memory_usage()

    def unload(self) -> None:
        """卸载模型，释放显存"""
        import gc
        import torch

        if self._model is not None:
            del self._model
            self._model = None

        self._model_info = None

        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
