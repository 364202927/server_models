"""llama.cpp GGUF 模型加载器。

该模块延迟导入 ``llama_cpp``，因此未安装可选依赖时不会影响 HF/vLLM 的导入。
"""

from __future__ import annotations

import time
import re
from pathlib import Path
from typing import Any

from .base import GenerationResult, MemoryUsage, ModelLoader
from ..utils.common import info as log_info


class GGUFLoader(ModelLoader):
    """使用 llama-cpp-python 加载单文件或目录中的 GGUF 模型。"""

    def load(
        self,
        model_path: str,
        *,
        dtype: str = "float16",
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "GGUFLoader":
        try:
            from llama_cpp import Llama
            try:
                from llama_cpp import llama_supports_gpu_offload
            except ImportError:
                try:
                    from llama_cpp.llama_cpp import llama_supports_gpu_offload
                except ImportError:
                    llama_supports_gpu_offload = None
        except ImportError as exc:
            raise RuntimeError("GGUF 模型需要安装 llama-cpp-python（建议按 CUDA 架构安装）") from exc

        source = Path(model_path)
        if source.is_dir():
            files = sorted(source.glob("*.gguf"))
            if not files:
                raise FileNotFoundError(f"目录中没有 .gguf 文件: {model_path}")
            source = files[0]
        if not source.is_file() or source.suffix.lower() != ".gguf":
            raise ValueError(f"不是有效的 GGUF 文件: {model_path}")

        gpu_layers = int(
            kwargs.get("gpu_offload_layers")
            if kwargs.get("gpu_offload_layers") is not None else -1
        )
        # n_gpu_layers 只有 CUDA 编译版本才真正生效；CPU 版会静默把 GGUF
        # 留在 RAM，因此在检测到 NVIDIA GPU 时提前给出明确错误。
        if gpu_layers != 0:
            try:
                from ..hardware import detect_gpu
                gpu_present = bool(detect_gpu())
            except ImportError:
                gpu_present = False
            if gpu_present and callable(llama_supports_gpu_offload):
                try:
                    if not llama_supports_gpu_offload():
                        raise RuntimeError(
                            "当前 llama-cpp-python 未启用 CUDA，GGUF 将加载到系统内存；"
                            "请安装带 CUDA 支持的构建版本。"
                        )
                except TypeError:
                    # 旧版本检测函数签名不同，交给构造器处理。
                    pass

        llm_kwargs: dict[str, Any] = {
            "model_path": str(source),
            # n_gpu_layers 决定有多少层放入 GPU；-1 表示尽可能全部 offload。
            "n_gpu_layers": gpu_layers,
            "n_batch": int(kwargs.get("batch_size", 1)),
            "verbose": bool(kwargs.get("verbose", False)),
        }
        if max_model_len is not None:
            llm_kwargs["n_ctx"] = int(max_model_len)
        if kwargs.get("flash_attention") is not None:
            llm_kwargs["flash_attn"] = bool(kwargs["flash_attention"])
        if kwargs.get("gpu_split"):
            # tensor_split 用每张卡的相对分配比例；None 表示 llama.cpp 自动分配。
            llm_kwargs["tensor_split"] = kwargs["gpu_split"]
        log_info("GGUF加载参数", str(source), llm_kwargs)
        self._model = Llama(**llm_kwargs)
        self._model_info = self._extract_model_info(str(source), dtype=dtype)

        metadata = getattr(self._model, "metadata", {}) or {}
        context = max_model_len or metadata.get("llama.context_length") or metadata.get("n_ctx_train")
        if context:
            self._model_info.context_length = int(context)
        # 转换器通常把 ``general.file_type`` 写成数字枚举（例如 30），
        # 它不是用户可读的量化名称；优先使用 metadata 字符串或文件名标记。
        quantization = metadata.get("general.quantization") or self._guess_quantization(source.name)
        if not quantization:
            file_type = metadata.get("general.file_type")
            if isinstance(file_type, str) and not file_type.isdigit():
                quantization = file_type
        if quantization:
            self._model_info.quantization = str(quantization)
        log_info("GGUF加载结果", str(source),
                 "metadata_keys=", list(metadata)[:20],
                 "context=", self._model_info.context_length,
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
        }
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
        **kwargs: Any,
    ) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        try:
            result = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens, temperature=max(temperature, 0.01), top_p=top_p,
                top_k=top_k, repeat_penalty=repetition_penalty, stop=stop_sequences,
            )
            text = str(result["choices"][0]["message"]["content"])
            usage = result.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            tokens = int(usage.get("completion_tokens", 0))
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            log_info("GGUF聊天接口不可用，回退普通生成", type(exc).__name__, exc)
            # 没有 chat template 时只能手工前置 system 段。
            prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            result = self._model(
                prompt, max_tokens=max_new_tokens, temperature=max(temperature, 0.01),
                top_p=top_p, top_k=top_k, repeat_penalty=repetition_penalty, stop=stop_sequences,
            )
            text = str(result["choices"][0].get("text", ""))
            tokens = len(self._model.tokenize(text.encode("utf-8")))
            prompt_tokens = len(self._model.tokenize(prompt.encode("utf-8")))
        elapsed = time.perf_counter() - start
        return GenerationResult(text, tokens, elapsed, tokens / elapsed if elapsed else 0.0, prompt_tokens)

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
