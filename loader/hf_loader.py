"""
hf_loader.py
HuggingFace Transformers 模型加载器

特点: 兼容性好，适合测试和小规模推理
局限: 单batch推理，速度较vLLM慢
"""

import copy
import gc
import json
import time
from pathlib import Path
from typing import Any

from .base import GenerationResult, MemoryUsage, ModelLoader, ToolCapabilityError, ToolOutputError
from .tool_format import parse_hermes_tool_calls
from ..hardware import detect_gpu

# torch/transformers 是 HF 后端的硬依赖；在模块级尝试一次导入，
# 缺失时统一置 None，由 load() 在入口处报出清晰的 RuntimeError，
# 而不是在每次调用相关方法时都重复 try/except ImportError。
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None


class HFLoader(ModelLoader):
    """HuggingFace Transformers加载器"""

    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        """若传入的是单个 .safetensors 文件，定位其所在目录并校验配套的 config/tokenizer 文件齐全。"""
        source = Path(model_path)
        if not (source.is_file() and source.suffix.lower() == ".safetensors"):
            return model_path
        # Transformers 需要 config/tokenizer 元数据；单独权重文件不能直接完成文本生成。
        directory = source.parent
        if not (directory / "config.json").is_file():
            raise ValueError(".safetensors 文件旁缺少 config.json，无法按 HuggingFace 模型加载")
        tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                           "special_tokens_map.json", "vocab.json")
        if not any((directory / name).is_file() for name in tokenizer_files):
            raise ValueError(".safetensors 文件旁缺少 tokenizer 元数据，无法完成文本生成")
        return str(directory)

    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "HFLoader":
        if torch is None:
            raise RuntimeError("HF 模型需要安装 torch、transformers（以及 safetensors）")

        model_path = self._resolve_model_dir(model_path)
        cuda_available = bool(torch.cuda.is_available())
        # nvidia-smi 可能仍能发现 GPU，但 CPU 版 PyTorch 无法把权重放入显存；
        # 继续静默加载会导致权重落到 RAM，直到推理时才暴露问题。
        if not cuda_available and detect_gpu():
            raise RuntimeError(
                "检测到 NVIDIA GPU，但当前 PyTorch 未启用 CUDA；请安装 CUDA 版 torch，"
                "否则模型会加载到系统内存。"
            )
        torch_dtype = getattr(torch, dtype, torch.float16)

        # 加载tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, **kwargs.get("tokenizer_kwargs", {}))

        # 构建模型加载参数
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            # 优先读取 safetensors，避免 pickle 权重反序列化风险并提升加载稳定性。
            "use_safetensors": True,
        }
        if quantization in ("4bit", "8bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantization == "4bit"),
                load_in_8bit=(quantization == "8bit"),
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            # auto 可能因显存估算不足而把层悄悄放到 CPU；固定到首张 GPU。
            model_kwargs["device_map"] = {"": "cuda:0"} if cuda_available else {"": "cpu"}
        elif cuda_available:
            # 使用映射字典显式整模型放入显存，避免字符串设备触发自动分层。
            model_kwargs["device_map"] = {"": "cuda:0"}

        # 加载模型
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if cuda_available:
            self._ensure_fully_on_gpu()

        self._model_info = self._extract_model_info(model_path, quantization=quantization, dtype=dtype)
        self._tool_parser = kwargs.get("tool_parser")

        # 参数量/上下文长度是附加信息，读取失败不应阻断模型加载。
        try:
            self._update_model_metadata()
        except Exception:
            pass

        try:
            actual_dtype = str(next(self._model.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            actual_dtype = dtype

        gpu_offload_layers = kwargs.get("gpu_offload_layers")
        self._effective_load = {
            "engine": "hf", "dtype": actual_dtype,
            "context_length": self._model_info.context_length if self._model_info else max_model_len,
            "gpu_offload_layers": gpu_offload_layers if gpu_offload_layers is not None else -1,
            "batch_size": kwargs.get("batch_size", 1),
            "flash_attention": kwargs.get("flash_attention", True),
            "draft_model": kwargs.get("draft_model"),
            "speculative_decoding": kwargs.get("speculative_decoding", False),
            "tensor_parallel": tensor_parallel_size,
            "gpu_split": kwargs.get("gpu_split"),
            "trust_remote_code": trust_remote_code,
            "tool_parser": self._tool_parser,
            "chat_format": kwargs.get("chat_format"),
        }
        return self

    def _ensure_fully_on_gpu(self) -> None:
        """确认模型权重已完整加载到 GPU；检测到 CPU/disk 分层时释放模型并报错。"""
        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            # accelerate 会在 hf_device_map 中标记被卸载到 CPU/disk 的层。
            offloaded = {str(location).lower() for location in device_map.values()
                        if str(location).lower() in {"cpu", "disk", "meta"}}
            if offloaded:
                self._model = None
                raise MemoryError(
                    "模型无法完整加载到 GPU，检测到 CPU/disk 分层；"
                    "请释放显存或降低 context_length/batch_size。"
                )

        try:
            devices = {str(device).lower() for param in self._model.parameters()
                      if (device := getattr(param, "device", None)) is not None}
        except StopIteration:
            return
        if any(name.startswith(("cpu", "meta")) for name in devices):
            self._model = None
            raise MemoryError("模型包含位于 CPU/RAM 的权重，未完整加载到 GPU 显存")

    def _update_model_metadata(self) -> None:
        """更新模型参数量和上下文长度信息"""
        total_params = sum(p.numel() for p in self._model.parameters())
        self._model_info.parameters = (f"{total_params / 1e9:.1f}B" if total_params >= 1e9
                                       else f"{total_params / 1e6:.0f}M")
        if hasattr(self._model.config, "max_position_embeddings"):
            self._model_info.context_length = self._model.config.max_position_embeddings

    def generate(self, prompt: str, *, max_new_tokens: int = 512, temperature: float = 0.3,
                top_p: float = 0.95, top_k: int = 50, repetition_penalty: float = 1.05,
                stop_sequences: list[str] | None = None, system_prompt: str = "",
                **kwargs: Any) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if "seed" in kwargs:
            torch.manual_seed(int(kwargs["seed"]))

        messages = kwargs.get("messages")
        if messages is None:
            # 兼容绕过 MsgHandler 直接调用 Loader 的场景（脚本/测试）；
            # 正常链路里 MsgHandler 已经把 prompt 统一成 messages。
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            messages = messages + [{"role": "user", "content": prompt}]
        tools = kwargs.get("tools") or []
        tool_choice = kwargs.get("tool_choice")
        tool_mode = bool(tools and tool_choice != "none")

        # 分支键是 tools，不是 messages：没有工具时走普通聊天模板，
        # 和只有一条 user 消息时的行为完全一致。
        rendered = (self._build_tool_prompt(messages, tools, tool_choice)
                   if tool_mode else self._build_chat_prompt(messages))
        inputs = self._tokenizer(rendered, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        prompt_tokens = input_ids.shape[1]

        # 构建生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.01),  # 避免除零
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if "min_p" in kwargs:
            gen_kwargs["min_p"] = kwargs["min_p"]

        # 处理停止序列
        if stop_sequences:
            stop_ids = [self._tokenizer.encode(s, add_special_tokens=False) for s in stop_sequences]
            gen_kwargs["eos_token_id"] = [self._tokenizer.eos_token_id] + [ids[0] for ids in stop_ids if ids]

        # 执行生成
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(input_ids, **gen_kwargs)
        elapsed = time.perf_counter() - start_time

        # 解析结果
        generated_ids = outputs[0][prompt_tokens:]
        tokens_generated = len(generated_ids)
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=not tool_mode)
        content, tool_calls = self._parse_tool_output(text) if tool_mode else (text, [])
        return GenerationResult(
            text=content, tokens_generated=tokens_generated, time_seconds=elapsed,
            tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0,
            prompt_tokens=prompt_tokens, tool_calls=tool_calls,
            finish_reason=("tool_calls" if tool_calls else
                          "length" if tokens_generated >= max_new_tokens else "stop"),
        )

    def _build_tool_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                           tool_choice: Any) -> str:
        if self._tool_parser != "hermes_json":
            raise ToolCapabilityError("HF 工具调用需要配置 load.tool_parser=hermes_json")
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
            return self._tokenizer.apply_chat_template(
                prepared, tools=tools, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError) as exc:
            raise ToolCapabilityError("当前 HF tokenizer 缺少可用的工具聊天模板") from exc

    def _parse_tool_output(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        if self._tool_parser != "hermes_json":
            raise ToolCapabilityError("HF 工具调用需要配置 load.tool_parser=hermes_json")
        content, calls = parse_hermes_tool_calls(text)
        for token in ("<|im_end|>", "<|endoftext|>"):
            content = content.replace(token, "")
        return content.strip(), calls

    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        """无工具场景：优先用 tokenizer 的 chat template 渲染完整多轮对话，不支持时退化为逐条拼接。"""
        try:
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError):
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        """HF特有: 追加模型参数内存占用明细"""
        usage = super().memory_usage(verbose)
        if verbose and self.is_loaded:
            usage.details = usage.details or {}
            try:
                param_bytes = sum(p.nelement() * p.element_size() for p in self._model.parameters())
                usage.details["model_param_mb"] = round(param_bytes / (1024 ** 2), 1)
                # 推理临时占用 = 总分配 - 参数占用
                usage.details["inference_temp_mb"] = round(
                    max(0, usage.gpu_allocated_mb - param_bytes / (1024 ** 2)), 1)
            except Exception:
                pass
        return usage

    def sleep_to_ram(self) -> bool:
        """HF 权重可移动到 CPU RAM；之后 ``wake`` 再移动回原设备。

        这会释放 GPU 参数显存，但 CPU 推理会明显变慢，因此只作为显存回收手段，
        不在正常请求路径中使用。
        """
        if not self.is_loaded:
            return False
        try:
            self._model.to("cpu")
            self.release_cache()
            return True
        except Exception:
            return False

    def wake(self) -> None:
        """将 RAM 中的 HF 模型恢复到 CUDA（无 CUDA 时保持 CPU）。"""
        if not self.is_loaded:
            raise RuntimeError("模型尚未加载")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model.to(device)

    def _get_gpu_device_index(self) -> int:
        """获取HF模型主设备索引"""
        if self.is_loaded and hasattr(self._model, "device"):
            device = self._model.device
            if hasattr(device, "index") and device.index is not None:
                return device.index
        return 0

    def release_cache(self) -> MemoryUsage:
        """HF特有: 清理past_key_values等推理缓存"""
        if self.is_loaded and hasattr(self._model, "_cache"):
            self._model._cache = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return self.memory_usage()

    def unload(self) -> None:
        """卸载模型，释放显存"""
        self._model = None
        self._tokenizer = None
        self._model_info = None
        self._effective_load = {}
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
