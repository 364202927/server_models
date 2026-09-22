"""
gguf_loader.py
llama.cpp GGUF 模型加载器。
该模块延迟导入 ``llama_cpp``，因此未安装可选依赖时不会影响 HF/vLLM 的导入。
"""

from __future__ import annotations

import importlib
import re
import time
from pathlib import Path
from typing import Any

from .base import GenerationResult, MemoryUsage, ModelLoader, ToolCapabilityError, ToolOutputError
from .tool_format import parse_hermes_tool_calls, render_tool_system_prompt
from ..hardware import detect_gpu
from ..utils.common import info as log_info, warn, err

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


def _find_gpu_offload_probe() -> Any:
    """``llama_supports_gpu_offload`` 的导出位置随 llama-cpp-python 版本变化，依次尝试已知位置。"""
    for module_name in ("llama_cpp", "llama_cpp.llama_cpp"):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "llama_supports_gpu_offload")
        except (ImportError, AttributeError):
            continue
    return None


_llama_supports_gpu_offload = _find_gpu_offload_probe() if Llama is not None else None


def _gpu_offload_supported() -> bool | None:
    """探测当前 llama-cpp-python 是否编译了 CUDA 支持；
    探测函数不可用、或旧版本签名不兼容时返回 None（未知，不阻断加载）。"""
    if not callable(_llama_supports_gpu_offload):
        return None
    try:
        return bool(_llama_supports_gpu_offload())
    except TypeError:
        # 旧版本探测函数签名不同，视为未知，交给构造器处理。
        return None


_EXTRA_SAMPLING_KEYS = ("min_p", "seed", "mirostat", "mirostat_eta", "mirostat_tau",
                        "repeat_last_n", "tfs_z", "logit_bias", "frequency_penalty",
                        "presence_penalty")


def _build_sampling(max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                    repetition_penalty: float, stop_sequences: list[str] | None,
                    extra: dict[str, Any]) -> dict[str, Any]:
    """组装 llama.cpp 采样参数，附加已知的可选采样字段。"""
    sampling = {"max_tokens": max_new_tokens, "temperature": max(temperature, 0.01),
               "top_p": top_p, "top_k": top_k, "repeat_penalty": repetition_penalty,
               "stop": stop_sequences}
    sampling.update({key: extra[key] for key in _EXTRA_SAMPLING_KEYS if key in extra})
    if "mirostat" in sampling:
        # OpenAI/Ollama 协议里这个参数叫 mirostat；llama-cpp-python 的
        # Llama.__call__/create_chat_completion 实际接收的形参名是
        # mirostat_mode，直传 "mirostat" 会被当成未知关键字参数拒绝。
        sampling["mirostat_mode"] = sampling.pop("mirostat")
    return sampling


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """把多轮 messages 拍平成纯文本，仅用于 create_chat_completion 不可用时的兜底补全接口。"""
    return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)


def _repair_think_open(text: str) -> str:
    """补全被截断的 ``<think>`` 开标签。

    Qwen3 等模型的 chat template 会把 ``<think>\\n`` 作为 assistant 段的生成起点
    拼进 prompt 里，模型只需要续写、最后吐出 ``</think>`` 收尾——所以
    completion 文本天然是"有尾没头"。不补回开标签的话，客户端（如 OpenWebUI）
    按 ``<think>...</think>`` 完整标签对识别推理块，会认不出来直接整段当正文
    显示，看不到折叠/变暗效果。这里只是补标签，不改变、不删除任何内容。"""
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        return "<think>\n" + text
    return text


class GGUFLoader(ModelLoader):
    """使用 llama-cpp-python 加载单文件或目录中的 GGUF 模型。"""

    # 模型没配置 context_length 时的兜底窗口；不能省略 n_ctx，llama.cpp 的
    # 构造器默认值是 512，历史消息一多就会静默截断或直接报错。
    _FALLBACK_N_CTX = 8192
    # 生成前给 prompt 之外预留的安全余量，以及至少要留出的生成空间；
    # 余量不够时直接报错，而不是让 create_chat_completion 抛异常后被
    # 误判成"没有 chat template"走进拍平兜底。
    _CONTEXT_SAFETY_MARGIN = 32
    _MIN_GENERATION_TOKENS = 16

    @staticmethod
    def _resolve_gguf_file(model_path: str) -> Path:
        """若传入目录，取目录下按名称排序的第一个 .gguf 文件；否则要求路径本身就是 .gguf 文件。"""
        source = Path(model_path)
        if source.is_dir():
            files = sorted(source.glob("*.gguf"))
            if not files:
                raise FileNotFoundError(f"目录中没有 .gguf 文件: {model_path}")
            source = files[0]
        if not source.is_file() or source.suffix.lower() != ".gguf":
            raise ValueError(f"不是有效的 GGUF 文件: {model_path}")
        return source

    @staticmethod
    def _build_llm_kwargs(source: Path, gpu_layers: int, max_model_len: int | None,
                          **kwargs: Any) -> dict[str, Any]:
        """组装传给 ``Llama()`` 构造器的参数。"""
        llm_kwargs: dict[str, Any] = {
            "model_path": str(source),
            # n_gpu_layers 决定有多少层放入 GPU；-1 表示尽可能全部 offload。
            "n_gpu_layers": gpu_layers,
            # 未配置时不能省略 n_ctx——llama.cpp 构造器默认只有 512。
            "n_ctx": int(max_model_len) if max_model_len else GGUFLoader._FALLBACK_N_CTX,
            "n_batch": int(kwargs.get("batch_size", 512)),
            "verbose": bool(kwargs.get("verbose", False)),
        }
        if kwargs.get("flash_attention") is not None:
            llm_kwargs["flash_attn"] = bool(kwargs["flash_attention"])
        if kwargs.get("gpu_split"):
            # tensor_split 用每张卡的相对分配比例；None 表示 llama.cpp 自动分配。
            llm_kwargs["tensor_split"] = kwargs["gpu_split"]
        if kwargs.get("chat_format"):
            llm_kwargs["chat_format"] = kwargs["chat_format"]
        return llm_kwargs

    def _apply_metadata(self, source: Path, max_model_len: int | None) -> dict[str, Any]:
        """从已加载的 Llama 对象读取 metadata，回填 context_length/quantization 到
        model_info，返回原始 metadata 供调用方记日志用。

        context_length 以 ``Llama.n_ctx()``（构造器实际生效值）为准，不能再信
        metadata 里的训练上下文——不同架构键名不一致（如 Qwen3 是
        ``qwen35.context_length`` 而不是通用的 ``llama.context_length``），
        取不到时会静默退回 dataclass 默认值 4096，把真实只有 512 的窗口掩盖掉。
        """
        metadata = getattr(self._model, "metadata", {}) or {}
        self._model_info.context_length = int(self._model.n_ctx())

        arch = metadata.get("general.architecture")
        trained = metadata.get(f"{arch}.context_length") if arch else None
        if trained and int(trained) > self._model_info.context_length:
            log_info("上下文窗口小于模型训练长度", source.name,
                     f"当前={self._model_info.context_length}", f"训练={trained}")

        # 转换器通常把 ``general.file_type`` 写成数字枚举（例如 30），
        # 它不是用户可读的量化名称；优先使用 metadata 字符串或文件名标记。
        quantization = metadata.get("general.quantization") or self._guess_quantization(source.name)
        file_type = metadata.get("general.file_type")
        if not quantization and isinstance(file_type, str) and not file_type.isdigit():
            quantization = file_type
        if quantization:
            self._model_info.quantization = str(quantization)
        return metadata

    def load(self, model_path: str, *, dtype: str = "float16", max_model_len: int | None = None,
             tensor_parallel_size: int = 1, trust_remote_code: bool = True,
             **kwargs: Any) -> "GGUFLoader":
        if Llama is None:
            raise RuntimeError("GGUF 模型需要安装 llama-cpp-python（建议按 CUDA 架构安装）")

        source = self._resolve_gguf_file(model_path)
        gpu_offload_layers = kwargs.get("gpu_offload_layers")
        gpu_layers = int(gpu_offload_layers) if gpu_offload_layers is not None else -1
        # n_gpu_layers 只有 CUDA 编译版本才真正生效；CPU 版会静默把 GGUF
        # 留在 RAM，因此在检测到 NVIDIA GPU 时提前给出明确错误。
        if gpu_layers != 0 and detect_gpu() and _gpu_offload_supported() is False:
            raise RuntimeError(
                "当前 llama-cpp-python 未启用 CUDA，GGUF 将加载到系统内存；"
                "请安装带 CUDA 支持的构建版本。"
            )

        self._chat_format = kwargs.get("chat_format")
        llm_kwargs = self._build_llm_kwargs(source, gpu_layers, max_model_len, **kwargs)
        self._model = Llama(**llm_kwargs)
        self._model_info = self._extract_model_info(str(source), dtype=dtype)

        self._apply_metadata(source, max_model_len)
        log_info("GGUF加载完成", source.name,
                "n_ctx=", self._model_info.context_length,
                "n_gpu_layers=", llm_kwargs["n_gpu_layers"],
                "quantization=", self._model_info.quantization or "未检测到")
        self._effective_load = {
            "dtype": dtype,
            "context_length": self._model_info.context_length,
            "gpu_offload_layers": llm_kwargs["n_gpu_layers"],
            "batch_size": llm_kwargs["n_batch"],
            "flash_attention": bool(kwargs.get("flash_attention", True)),
            "draft_model": kwargs.get("draft_model"),
            "speculative_decoding": bool(kwargs.get("speculative_decoding", False)),
            "tensor_parallel": tensor_parallel_size,
            "gpu_split": kwargs.get("gpu_split"),
            "trust_remote_code": trust_remote_code,
            "tool_parser": kwargs.get("tool_parser"),
            "chat_format": self._chat_format,
        }
        return self

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

        # 请求的 max_new_tokens 可能超过上下文剩余空间（历史消息越攒越长）；
        # 在这里提前按 n_ctx 裁掉，而不是让 create_chat_completion 抛异常后
        # 被误判成"没有 chat template"走进拍平兜底、吐出垃圾续写。
        max_new_tokens = self._clamp_to_context(messages, max_new_tokens)
        sampling = _build_sampling(max_new_tokens, temperature, top_p, top_k, repetition_penalty,
                                   stop_sequences, kwargs)

        # 分支键是 tools，不是 messages：没有工具时走普通聊天，
        # 和改造前的普通对话行为一致，只是现在用的是完整多轮 messages。
        tools = kwargs.get("tools") or []
        if not tools:
            return self._chat(messages, sampling, max_new_tokens)
        if self._chat_format == "chatml-function-calling":
            return self._chat_with_backend_tools(messages, tools, kwargs.get("tool_choice"), sampling)
        # 未配置 chatml-function-calling 不再直接报错：先尝试模型自带模板
        # 支持的 Hermes 风格 <tool_call> 兜底（如 Qwen3 原生就是这个格式）。
        return self._chat_with_hermes_tools(messages, tools, kwargs.get("tool_choice"),
                                            sampling, max_new_tokens)

    def _clamp_to_context(self, messages: list[dict[str, Any]], max_new_tokens: int) -> int:
        """按 ``n_ctx`` 与已用 prompt token 数裁剪本次可生成的 token 数；
        剩余空间连最小生成长度都不够时直接报错，不再尝试硬生成。"""
        n_ctx = self._model.n_ctx()
        prompt_tokens = len(self._model.tokenize(_flatten_messages(messages).encode("utf-8")))
        available = n_ctx - prompt_tokens - self._CONTEXT_SAFETY_MARGIN
        if available < self._MIN_GENERATION_TOKENS:
            raise ValueError(
                f"上下文不足: prompt 约 {prompt_tokens} token，窗口 {n_ctx} token，"
                "请精简历史消息或调大 context_length"
            )
        return min(max_new_tokens, available)

    def _chat(self, messages: list[dict[str, Any]], sampling: dict[str, Any],
             max_new_tokens: int) -> GenerationResult:
        """无工具场景：用后端聊天接口生成完整多轮对话；模型没有 chat template 时
        退化为普通补全接口。上下文溢出等请求级错误已经在 generate() 里提前拦截，
        这里只处理"接口本身不可用"，不再吞掉别的异常。"""
        start = time.perf_counter()
        try:
            result = self._model.create_chat_completion(messages=messages, **sampling)
            choice = result["choices"][0]
            text = str(choice["message"]["content"])
            finish_reason = str(choice.get("finish_reason") or "stop")
            usage = result.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            tokens = int(usage.get("completion_tokens", 0))
        except (AttributeError, TypeError) as exc:
            # 打印异常原文 + 触发这次调用的消息结构（角色序列/是否缺 content），
            # 这是目前唯一能定位"是哪条消息、哪个字段让模型自带的 chat template
            # 渲染失败"的线索，不能只留异常类名。
            err("GGUF聊天模板渲染失败，回退补全接口", type(exc).__name__, exc,
                "messages=", [(m.get("role"), "content" in m, bool(m.get("tool_calls")))
                              for m in messages])
            # 没有 chat template 时只能手工拍平多轮消息。
            flat = _flatten_messages(messages)
            result = self._model(flat, **sampling)
            text = str(result["choices"][0].get("text", ""))
            tokens = len(self._model.tokenize(text.encode("utf-8")))
            finish_reason = str(result["choices"][0].get("finish_reason") or
                                ("length" if tokens >= max_new_tokens else "stop"))
            prompt_tokens = len(self._model.tokenize(flat.encode("utf-8")))
        elapsed = time.perf_counter() - start
        if finish_reason == "length":
            warn("生成被截断", "tokens=", tokens, "max_new_tokens=", max_new_tokens)
        return GenerationResult(_repair_think_open(text), tokens, elapsed,
                                tokens / elapsed if elapsed else 0.0,
                                prompt_tokens, finish_reason=finish_reason)

    def _chat_with_backend_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                                 tool_choice: Any, sampling: dict[str, Any]) -> GenerationResult:
        """chat_format=chatml-function-calling 时，走 llama-cpp-python 内置的工具调用支持。"""
        messages = [dict(item) for item in messages]
        backend_choice = tool_choice
        if tool_choice == "required":
            backend_choice = "auto"
            messages.insert(0, {"role": "system", "content": "你必须调用至少一个可用工具。"})

        start = time.perf_counter()
        try:
            result = self._model.create_chat_completion(messages=messages, tools=tools,
                                                        tool_choice=backend_choice, **sampling)
            choice = result["choices"][0]
            message = choice["message"]
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            raise ToolCapabilityError(f"GGUF 工具聊天接口不可用: {exc}") from exc

        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ToolOutputError("GGUF 后端返回的 tool_calls 不是数组")
        usage = result.get("usage", {})
        elapsed = time.perf_counter() - start
        tokens = int(usage.get("completion_tokens", 0))
        return GenerationResult(
            _repair_think_open(str(message.get("content") or "")), tokens, elapsed,
            tokens / elapsed if elapsed else 0.0,
            int(usage.get("prompt_tokens", 0)), calls,
            str(choice.get("finish_reason") or ("tool_calls" if calls else "stop")),
        )

    def _chat_with_hermes_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                                tool_choice: Any, sampling: dict[str, Any],
                                max_new_tokens: int) -> GenerationResult:
        """未配置 chatml-function-calling 时的兜底：把工具 schema 渲染进 system 段，
        用模型自带聊天模板生成，再解析 Hermes 风格的 <tool_call> 标签。
        适用于原生就输出这种格式的模型（如 Qwen3）。

        注意：这里不把 tools= 传给 create_chat_completion，完全靠 system 段
        里的 schema 说明 + 模型自身的 Jinja 模板渲染历史消息（包括之前轮次
        的 assistant.tool_calls / role=tool），依赖模型模板本身支持这些字段。
        """
        if tool_choice == "none":
            return self._chat(messages, sampling, max_new_tokens)

        instruction, tools = render_tool_system_prompt(tools, tool_choice)
        prepared = [dict(item) for item in messages]
        if prepared and prepared[0].get("role") == "system":
            existing = str(prepared[0].get("content") or "")
            prepared[0]["content"] = f"{instruction}\n\n{existing}" if existing else instruction
        else:
            prepared.insert(0, {"role": "system", "content": instruction})

        result = self._chat(prepared, sampling, max_new_tokens)
        content, calls = parse_hermes_tool_calls(result.text)
        return GenerationResult(content, result.tokens_generated, result.time_seconds,
                                result.tokens_per_second, result.prompt_tokens, calls,
                                "tool_calls" if calls else result.finish_reason)

    @staticmethod
    def _guess_quantization(filename: str) -> str | None:
        """部分 GGUF 转换器不会写量化 metadata，从文件名补充常见量化标记。"""
        match = re.search(r"(?i)(IQ\d+[_A-Z]*|Q\d+[_A-Z0-9]*|F16|F32|BF16)", filename)
        return match.group(1) if match else None

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        usage = super().memory_usage(verbose)
        if verbose:
            usage.details = usage.details or {}
            usage.details["format"] = "gguf"
            # llama.cpp 的 CUDA 分配不经过 torch，单模型占用由 ModelsMgr 用
            # 加载前后的 nvidia-smi 差值归属，这里不重复测量。
            usage.details["gpu_memory_source"] = "models_mgr(delta)"
        return usage

    def sleep_to_ram(self) -> bool:
        # llama.cpp 的上下文和 mmap 状态不能可靠地迁移到 CPU 后再恢复，交由管理器卸载。
        return False

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_info = None
        self._effective_load = {}
        self.release_cache()