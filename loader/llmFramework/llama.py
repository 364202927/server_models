"""
llama.py
llama.cpp GGUF 模型加载器。
该模块延迟导入 ``llama_cpp``，因此未安装可选依赖时不会影响 vllm/sglang 的导入。
"""

from __future__ import annotations

import importlib
import re
import time
from pathlib import Path
from typing import Any

from ...hardware import detect_gpu
from ...utils.common import info as log_info, warn, error
from .baseInference import GenerationResult, MemoryUsage, baseInference

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


class llama(baseInference):
    """使用 llama-cpp-python 加载单文件或目录中的 GGUF 模型。"""

    def __init__(self) -> None:
        super().__init__()
        # 唤醒时重建 Llama 对象要用的原始构造参数；unload()/_mark_unloaded() 之外单独清空。
        self._saved_llm_kwargs: dict[str, Any] | None = None
        self._saved_draft_kwargs: dict[str, Any] | None = None
        self._draft_model = None

    _SAMPLING_KEY_MAP = {"repetition_penalty": "repeat_penalty", "mirostat": "mirostat_mode"}
    _EXTRA_SAMPLING_KEYS = ("min_p", "seed", "mirostat", "mirostat_eta", "mirostat_tau",
                            "repeat_last_n", "tfs_z", "logit_bias", "frequency_penalty",
                            "presence_penalty")
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
            "n_ctx": int(max_model_len) if max_model_len else llama._FALLBACK_N_CTX,
            "n_batch": int(kwargs.get("batch_size", 512)),
            "verbose": bool(kwargs.get("verbose", False)),
            'use_mlock':True             #内存常驻锁定,休眠时保持在oom(ulimit -l:检查可存放无限页)
        }
        if kwargs.get("flash_attention") is not None:
            llm_kwargs["flash_attn"] = bool(kwargs["flash_attention"])
        if kwargs.get("gpu_split"):
            # tensor_split 用每张卡的相对分配比例；None 表示 llama.cpp 自动分配。
            llm_kwargs["tensor_split"] = kwargs["gpu_split"]
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

    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "llama":
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

        draft, mtp, lora = kwargs.pop("draft", None), kwargs.pop("mtp", False), kwargs.pop("lora", None)
        if draft and mtp:
            raise ValueError("draft 与 mtp 不能同时启用")
        llm_kwargs = self._build_llm_kwargs(source, gpu_layers, max_model_len, **kwargs)
        optional_kwargs = self._accepted_engine_kwargs(
            Llama, kwargs, {"model_path", "n_gpu_layers", "n_ctx", "n_batch", "verbose",
                            "gpu_memory_utilization", "gpu_offload_layers", "batch_size",
                            "flash_attention", "gpu_split",
                            "enable_memory_saver",
                            "enable_sleep_mode"})
        llm_kwargs.update(optional_kwargs)
        if lora:
            llm_kwargs["lora_path"] = lora
            llm_kwargs["lora_scale"] = kwargs.get("lora_scale", 1.0)
            if kwargs.get("lora_base") is not None:
                llm_kwargs["lora_base"] = kwargs["lora_base"]
            log_info("启用llama.cpp LoRA", lora)
        self._saved_draft_kwargs = None
        if draft:
            draft_source = self._resolve_gguf_file(draft)
            draft_layers = kwargs.get("draft_gpu_offload_layers")
            self._saved_draft_kwargs = {
                "model_path": str(draft_source),
                "n_gpu_layers": 0 if draft_layers is None else int(draft_layers),
            }
            self._saved_draft_kwargs.update(self._accepted_engine_kwargs(
                Llama, kwargs, {"model_path", "n_gpu_layers", "draft_gpu_offload_layers"}))
            self._draft_model = Llama(**self._saved_draft_kwargs)
            llm_kwargs["draft_model"] = self._draft_model
            log_info("启用llama.cpp Draft", str(draft_source))
        elif mtp:
            log_info("启用llama.cpp MTP；由 GGUF/backend 自动识别")
        self._saved_llm_kwargs = {key: value for key, value in llm_kwargs.items()
                                  if key != "draft_model"}
        self._model = Llama(**llm_kwargs)
        self._sleep_capable = True
        self._model_info = self._extract_model_info(str(source), dtype=dtype)

        self._apply_metadata(source, max_model_len)
        log_info("GGUF加载完成", source.name,
                "n_ctx=", self._model_info.context_length,
                "n_gpu_layers=", llm_kwargs["n_gpu_layers"],
                "quantization=", self._model_info.quantization or "未检测到")
        self._effective_load = {
            "engine": "llama", "dtype": dtype,
            "context_length": self._model_info.context_length,
            "gpu_offload_layers": llm_kwargs["n_gpu_layers"],
            "batch_size": llm_kwargs["n_batch"],
            "flash_attention": bool(kwargs.get("flash_attention", True)),
            "draft": draft,
            "mtp": bool(mtp),
            "lora": lora,
            **({"draft_gpu_offload_layers": self._saved_draft_kwargs["n_gpu_layers"]}
               if self._saved_draft_kwargs else {}),
            **({"lora_scale": llm_kwargs["lora_scale"]} if lora else {}),
            **({"lora_base": llm_kwargs["lora_base"]} if lora and "lora_base" in llm_kwargs else {}),
            "tensor_parallel": tensor_parallel_size,
            "gpu_split": kwargs.get("gpu_split"),
            "trust_remote_code": trust_remote_code,
            **optional_kwargs,
        }
        return self

    def generate(self, prompt: str, *, max_new_tokens: int = 512, temperature: float = 0.3,
                top_p: float = 0.95, top_k: int = 50, repetition_penalty: float = 1.05,
                stop_sequences: list[str] | None = None, system_prompt: str = "",
                **kwargs: Any) -> GenerationResult:
        """llama.cpp 的后端是消息级 chat-completions 接口，形状与 vllm/sglang 的
        整段 prompt 进/整段文本出不同，完整覆盖 generate() 而不用模板方法。"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        messages = self._normalize_messages(prompt, system_prompt, kwargs)

        # 请求的 max_new_tokens 可能超过上下文剩余空间（历史消息越攒越长）；
        # 在这里提前按 n_ctx 裁掉，而不是让 create_chat_completion 抛异常后
        # 被误判成"没有 chat template"走进拍平兜底、吐出垃圾续写。
        max_new_tokens = self._clamp_to_context(messages, max_new_tokens)
        sampling = self._build_sampling(max_new_tokens, temperature, top_p, top_k, repetition_penalty,
                                        stop_sequences, kwargs)

        # tools/tool_choice 纯透传：客户端怎么传就怎么交给 create_chat_completion，
        # 由模型自带 chat_template 自己处理，这里不做任何校验/渲染/解析。
        return self._chat(messages, sampling, max_new_tokens,
                          tools=kwargs.get("tools") or None, tool_choice=kwargs.get("tool_choice"))

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
             max_new_tokens: int, tools: list[dict[str, Any]] | None = None,
             tool_choice: Any = None) -> GenerationResult:
        """用后端聊天接口生成完整多轮对话；模型没有 chat template 时退化为普通补全接口。

        tools/tool_choice 有值时原样透传给 create_chat_completion，由模型自带
        chat_template 自己决定怎么处理，这里不做任何校验/渲染/解析。上下文溢出
        等请求级错误已经在 generate() 里提前拦截，这里只处理"接口本身不可用"，
        不再吞掉别的异常。"""
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
        if tool_choice is not None:
            extra["tool_choice"] = tool_choice

        start = time.perf_counter()
        try:
            result = self._model.create_chat_completion(messages=messages, **extra, **sampling)
            choice = result["choices"][0]
            message = choice["message"]
            text = str(message.get("content") or "")
            calls = message.get("tool_calls") or []
            finish_reason = str(choice.get("finish_reason") or ("tool_calls" if calls else "stop"))
            usage = result.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            tokens = int(usage.get("completion_tokens", 0))
        except (AttributeError, TypeError) as exc:
            # 打印异常原文 + 触发这次调用的消息结构（角色序列/是否缺 content），
            # 这是目前唯一能定位"是哪条消息、哪个字段让模型自带的 chat template
            # 渲染失败"的线索，不能只留异常类名。
            error("GGUF聊天模板渲染失败，回退补全接口", type(exc).__name__, exc,
                "messages=", [(m.get("role"), "content" in m, bool(m.get("tool_calls")))
                              for m in messages])
            # 没有 chat template 时只能手工拍平多轮消息；tools 信息在这条兜底
            # 路径上没有承载的地方，直接丢弃，不会产生 tool_calls。
            flat = _flatten_messages(messages)
            result = self._model(flat, **sampling)
            text = str(result["choices"][0].get("text", ""))
            tokens = len(self._model.tokenize(text.encode("utf-8")))
            finish_reason = str(result["choices"][0].get("finish_reason") or
                                ("length" if tokens >= max_new_tokens else "stop"))
            prompt_tokens = len(self._model.tokenize(flat.encode("utf-8")))
            calls = []
        elapsed = time.perf_counter() - start
        if finish_reason == "length":
            warn("生成被截断", "tokens=", tokens, "max_new_tokens=", max_new_tokens)
        return GenerationResult(_repair_think_open(text), tokens, elapsed,
                                tokens / elapsed if elapsed else 0.0,
                                prompt_tokens, calls, finish_reason)

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

    def _engine_sleep(self) -> None:
        """销毁 Llama 实例放掉显存；GGUF 文件仍在 OS 页缓存里，唤醒时重建会命中缓存。

        llama.cpp 没有"显存->内存"的权重迁移 API，这是能做到的最接近的形状。
        close() 走 llama-cpp-python 的 ExitStack 清理，比只丢引用更可靠地释放显存。
        """
        model, self._model = self._model, None
        if hasattr(model, "close"):
            model.close()
        draft, self._draft_model = self._draft_model, None
        if draft is not None and hasattr(draft, "close"):
            draft.close()

    def _engine_wake(self) -> None:
        if self._saved_draft_kwargs:
            self._draft_model = Llama(**self._saved_draft_kwargs)
        if self._saved_llm_kwargs is not None:
            llm_kwargs = dict(self._saved_llm_kwargs)
            if self._draft_model is not None:
                llm_kwargs["draft_model"] = self._draft_model
            self._model = Llama(**llm_kwargs)

    def sleep_holds_ram(self) -> bool:
        # 权重待在 OS 页缓存里(算可回收内存)，不占本进程 RSS；
        # 卸载它腾不出 MemAvailable，所以不参与 RAM 回收。
        return False

    def _unload_engine(self) -> None:
        model, self._model = self._model, None
        if model is not None and hasattr(model, "close"):
            model.close()
        draft, self._draft_model = self._draft_model, None
        if draft is not None and hasattr(draft, "close"):
            draft.close()
        self._mark_unloaded()
        self._saved_llm_kwargs = None
        self._saved_draft_kwargs = None
        self.release_cache()
