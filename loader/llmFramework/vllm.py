"""
vllm.py
vLLM 推理框架

特点: 高性能推理,支持批量生成和张量并行
适用: 生产环境,需要高吞吐量场景
"""

import gc
import time
from typing import Any

from ...utils.common import info as log_info
from .baseInference import GenerationResult, MemoryUsage, baseInference

# vllm 自带 torch 依赖;在模块级尝试一次导入,缺失时统一置 None,
# 由 load() 在入口处报出清晰的 RuntimeError,而不是让 ImportError 直接冒出来。
try:
    import torch
    from vllm import LLM, SamplingParams
except ImportError:
    torch = LLM = SamplingParams = None
try:
    from vllm.lora.request import LoRARequest
except ImportError:
    LoRARequest = None


class vllm(baseInference):
    """vLLM 推理框架 - 高性能批量推理"""

    _EXTRA_SAMPLING_KEYS = ("min_p", "seed", "frequency_penalty", "presence_penalty", "logit_bias")

    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "vllm":
        if LLM is None:
            raise RuntimeError("vLLM 模型需要安装 vllm")

        draft, mtp, lora = kwargs.pop("draft", None), kwargs.pop("mtp", False), kwargs.pop("lora", None)
        if draft and mtp:
            raise ValueError("draft 与 mtp 不能同时启用")
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": (kwargs.get("gpu_memory_utilization")
                                       if kwargs.get("gpu_memory_utilization") is not None else 0.9),
        }
        if quantization:
            llm_kwargs["quantization"] = quantization
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len

        feature_kwargs: dict[str, Any] = {}
        if draft:
            feature_kwargs = {
                "speculative_model": draft,
                "num_speculative_tokens": (kwargs.get("num_speculative_tokens")
                                            if kwargs.get("num_speculative_tokens") is not None else 5),
            }
            log_info("启用vLLM Draft", draft)
        elif mtp:
            feature_kwargs = {
                "speculative_model": "[INLINE]",
                "num_speculative_tokens": (kwargs.get("num_speculative_tokens")
                                            if kwargs.get("num_speculative_tokens") is not None else 1),
                "speculative_draft_tensor_parallel_size": (
                    kwargs.get("speculative_draft_tensor_parallel_size")
                    if kwargs.get("speculative_draft_tensor_parallel_size") is not None else 1),
            }
            log_info("启用vLLM MTP")
        if lora:
            feature_kwargs.update({
                "enable_lora": True,
                "max_loras": kwargs.get("max_loras") if kwargs.get("max_loras") is not None else 1,
                "max_lora_rank": (kwargs.get("max_lora_rank")
                                  if kwargs.get("max_lora_rank") is not None else 16),
            })
            if LoRARequest is None:
                raise RuntimeError("当前 vLLM 未提供 LoRARequest")
            self._lora_request = LoRARequest("configured-lora", 1, lora)
            log_info("启用vLLM LoRA", lora)
        else:
            self._lora_request = None
        enable_sleep_mode = kwargs.get("enable_sleep_mode", True)
        optional_kwargs = self._accepted_engine_kwargs(
            LLM, kwargs, {"model", "trust_remote_code", "tensor_parallel_size", "dtype",
                          "max_model_len", "quantization", "gpu_memory_utilization",
                          "enable_sleep_mode", "enable_memory_saver", "tool_parser"})
        llm_kwargs.update(optional_kwargs)
        # 顶层 draft/mtp/lora 是功能开关，派生出的构造参数优先于同名可选字段。
        llm_kwargs.update(feature_kwargs)
        if enable_sleep_mode:
            llm_kwargs["enable_sleep_mode"] = True
            try:
                self._model = LLM(**llm_kwargs)
                self._sleep_capable = True
            except (TypeError, ValueError) as exc:
                # 平台不支持 sleep mode 或旧版本不认该参数时退回普通加载。
                log_info("vLLM休眠模式不可用，按普通模式加载", type(exc).__name__, exc)
                llm_kwargs.pop("enable_sleep_mode", None)
                self._model = LLM(**llm_kwargs)
                self._sleep_capable = False
        else:
            self._model = LLM(**llm_kwargs)
            self._sleep_capable = False
        self._model_info = self._extract_model_info(model_path, quantization=quantization, dtype=dtype)
        self._tool_parser = kwargs.get("tool_parser")

        # 上下文长度是附加信息,读取失败不应阻断模型加载。
        try:
            self._apply_context_length()
        except Exception:
            pass

        effective_load = dict(optional_kwargs)
        effective_load.update({
            "engine": "vllm", "dtype": dtype,
            "context_length": self._model_info.context_length if self._model_info else max_model_len,
            "tensor_parallel": tensor_parallel_size,
            "gpu_memory_utilization": llm_kwargs["gpu_memory_utilization"],
            "quantization": quantization,
            "enable_sleep_mode": self._sleep_capable,
            "trust_remote_code": trust_remote_code,
            "tool_parser": self._tool_parser,
        })
        effective_load.update(feature_kwargs)
        self._effective_load = effective_load
        return self

    def _apply_context_length(self) -> None:
        """从 vLLM 引擎配置回填上下文长度;调用方用一次 try/except 包裹,读取失败就跳过。"""
        config = self._model.llm_engine.model_config
        if hasattr(config, "max_model_len"):
            self._model_info.context_length = config.max_model_len

    def _get_tokenizer(self) -> Any:
        return self._model.get_tokenizer()

    def _engine_sleep(self) -> None:
        # level=1：权重 offload 到 CPU 内存、丢弃 KV cache。
        # level=2 会连权重一起丢，唤醒后还得重新载权重，不符合"休眠到 RAM"的语义。
        self._model.sleep(level=1)

    def _engine_wake(self) -> None:
        self._model.wake_up()

    def _run_engine(self, rendered_prompt: str, sampling: dict[str, Any]) -> tuple[str, int, int]:
        sampling_params = SamplingParams(**sampling)
        generate_kwargs = ({"lora_request": self._lora_request}
                           if self._lora_request is not None else {})
        outputs = self._model.generate([rendered_prompt], sampling_params, **generate_kwargs)
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
        generate_kwargs = ({"lora_request": self._lora_request}
                           if self._lora_request is not None else {})
        outputs = self._model.generate(prompts, sampling_params, **generate_kwargs)
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

    def _unload_engine(self) -> None:
        """卸载模型,释放显存"""
        self._mark_unloaded()
        self._lora_request = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
