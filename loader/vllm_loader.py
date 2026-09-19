"""
vllm_loader.py
vLLM 模型加载器

特点: 高性能推理，支持批量生成和张量并行
适用: 生产环境，需要高吞吐量场景
"""

import gc
import copy
import json
import time
from typing import Any

from .base import GenerationResult, MemoryUsage, ModelLoader, ToolCapabilityError
from .tool_format import parse_hermes_tool_calls

# vllm 自带 torch 依赖；在模块级尝试一次导入，缺失时统一置 None，
# 由 load() 在入口处报出清晰的 RuntimeError，而不是让 ImportError 直接冒出来。
try:
    import torch
    from vllm import LLM, SamplingParams
except ImportError:
    torch = LLM = SamplingParams = None

_EXTRA_SAMPLING_KEYS = ("min_p", "seed", "frequency_penalty", "presence_penalty", "logit_bias")


def _build_sampling(max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                    repetition_penalty: float, stop_sequences: list[str] | None,
                    extra: dict[str, Any]) -> dict[str, Any]:
    """组装 vLLM 采样参数，附加已知的可选采样字段。"""
    sampling = {"max_tokens": max_new_tokens, "temperature": max(temperature, 0.01),
               "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty,
               "stop": stop_sequences}
    sampling.update({key: extra[key] for key in _EXTRA_SAMPLING_KEYS if key in extra})
    return sampling


class VLLMLoader(ModelLoader):
    """vLLM加载器 - 高性能批量推理"""

    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "VLLMLoader":
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

        # 上下文长度是附加信息，读取失败不应阻断模型加载。
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
        """从 vLLM 引擎配置回填上下文长度；调用方用一次 try/except 包裹，读取失败就跳过。"""
        config = self._model.llm_engine.model_config
        if hasattr(config, "max_model_len"):
            self._model_info.context_length = config.max_model_len

    def generate(self, prompt: str, *, max_new_tokens: int = 512, temperature: float = 0.3,
                top_p: float = 0.95, top_k: int = 50, repetition_penalty: float = 1.05,
                stop_sequences: list[str] | None = None, system_prompt: str = "",
                **kwargs: Any) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = kwargs.get("messages")
        if messages is None:
            # 兼容绕过 MsgHandler 直接调用 Loader 的场景（脚本/测试）；
            # 正常链路里 MsgHandler 已经把 prompt 统一成 messages。
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            messages = messages + [{"role": "user", "content": prompt}]
        tools = kwargs.get("tools") or []
        tool_choice = kwargs.get("tool_choice")
        tool_mode = bool(tools and tool_choice != "none")

        # 分支键是 tools，不是 messages：和 HF/GGUF 两个 Loader 保持一致。
        rendered = (self._build_tool_prompt(messages, tools, tool_choice)
                   if tool_mode else self._build_chat_prompt(messages))

        sampling = _build_sampling(max_new_tokens, temperature, top_p, top_k,
                                   repetition_penalty, stop_sequences, kwargs)
        sampling_params = SamplingParams(**sampling)

        start_time = time.perf_counter()
        outputs = self._model.generate([rendered], sampling_params)
        elapsed = time.perf_counter() - start_time

        output = outputs[0]
        text = output.outputs[0].text
        tokens_generated = len(output.outputs[0].token_ids)
        content, tool_calls = self._parse_tool_output(text) if tool_mode else (text, [])
        return GenerationResult(
            text=content, tokens_generated=tokens_generated, time_seconds=elapsed,
            tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0,
            prompt_tokens=len(output.prompt_token_ids), tool_calls=tool_calls,
            finish_reason=("tool_calls" if tool_calls else
                          "length" if tokens_generated >= max_new_tokens else "stop"),
        )

    def _build_tool_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                           tool_choice: Any) -> str:
        if self._tool_parser != "hermes_json":
            raise ToolCapabilityError("vLLM 工具调用需要配置 load.tool_parser=hermes_json")
        prepared = copy.deepcopy(messages)
        for message in prepared:
            for call in message.get("tool_calls", []):
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        if tool_choice == "none":
            tools = []
        elif tool_choice == "required":
            prepared.insert(0, {"role": "system", "content": "你必须调用至少一个可用工具。"})
        elif isinstance(tool_choice, dict):
            name = tool_choice["function"]["name"]
            tools = [tool for tool in tools if tool["function"]["name"] == name]
            prepared.insert(0, {"role": "system", "content": f"你必须调用工具 {name}。"})
        try:
            tokenizer = self._model.get_tokenizer()
            return tokenizer.apply_chat_template(prepared, tools=tools, tokenize=False,
                                                 add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError) as exc:
            raise ToolCapabilityError("当前 vLLM tokenizer 缺少可用的工具聊天模板") from exc

    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        """无工具场景：优先用 tokenizer 的 chat template 渲染完整多轮对话，不支持时退化为逐条拼接。"""
        try:
            tokenizer = self._model.get_tokenizer()
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError):
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)

    def _parse_tool_output(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        if self._tool_parser != "hermes_json":
            raise ToolCapabilityError("vLLM 工具调用需要配置 load.tool_parser=hermes_json")
        content, calls = parse_hermes_tool_calls(text)
        return content.strip(), calls

    def generate_batch(self, prompts: list[str], *, max_new_tokens: int = 512, temperature: float = 0.3,
                       top_p: float = 0.95, **kwargs) -> list[GenerationResult]:
        """批量生成 - vLLM的核心优势，显著提升吞吐量。"""
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
        """遍历 vLLM 引擎的 scheduler(s)，取第一个暴露了 block_manager 统计的 KV cache 块信息。"""
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
        """卸载模型，释放显存"""
        self._model = None
        self._model_info = None
        self._effective_load = {}
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
