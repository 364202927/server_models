"""
HuggingFace Transformers 模型加载器

特点: 兼容性好，适合测试和小规模推理
局限: 单batch推理，速度较vLLM慢
"""

import time
from pathlib import Path
from typing import Any

from .base import ModelLoader, ModelInfo, GenerationResult, MemoryUsage


class HFLoader(ModelLoader):
    """HuggingFace Transformers加载器"""

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
    ) -> "HFLoader":
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("HF 模型需要安装 torch、transformers（以及 safetensors）") from exc

        source = Path(model_path)
        if source.is_file() and source.suffix.lower() == ".safetensors":
            # Transformers 需要 config/tokenizer 元数据；单独权重文件不能直接完成文本生成。
            source = source.parent
            if not (source / "config.json").is_file():
                raise ValueError(".safetensors 文件旁缺少 config.json，无法按 HuggingFace 模型加载")
            tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                               "special_tokens_map.json", "vocab.json")
            if not any((source / name).is_file() for name in tokenizer_files):
                raise ValueError(".safetensors 文件旁缺少 tokenizer 元数据，无法完成文本生成")
            model_path = str(source)

        cuda_available = bool(torch.cuda.is_available())
        if not cuda_available:
            # nvidia-smi 可能仍能发现 GPU，但 CPU 版 PyTorch 无法把权重放入显存；
            # 继续静默加载会导致权重落到 RAM，直到推理时才暴露问题。
            try:
                from ..hardware import detect_gpu
                if detect_gpu():
                    raise RuntimeError(
                        "检测到 NVIDIA GPU，但当前 PyTorch 未启用 CUDA；请安装 CUDA 版 torch，"
                        "否则模型会加载到系统内存。"
                    )
            except ImportError:
                pass
        device = "cuda:0" if cuda_available else "cpu"
        torch_dtype = getattr(torch, dtype, torch.float16)

        # 加载tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **kwargs.get("tokenizer_kwargs", {})
        )

        # 构建模型加载参数
        model_kwargs: dict = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            # 优先读取 safetensors，避免 pickle 权重反序列化风险并提升加载稳定性。
            "use_safetensors": True,
        }

        # BitsAndBytes量化配置
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
            # accelerate 会在 hf_device_map 中标记被卸载到 CPU/disk 的层。
            device_map = getattr(self._model, "hf_device_map", None)
            if isinstance(device_map, dict):
                offloaded = {
                    str(location).lower() for location in device_map.values()
                    if str(location).lower() in {"cpu", "disk", "meta"}
                }
                if offloaded:
                    del self._model
                    self._model = None
                    raise MemoryError(
                        "模型无法完整加载到 GPU，检测到 CPU/disk 分层；"
                        "请释放显存或降低 context_length/batch_size。"
                    )
            try:
                devices = {
                    str(device).lower()
                    for param in self._model.parameters()
                    if (device := getattr(param, "device", None)) is not None
                }
                if devices and any(name.startswith("cpu") or name.startswith("meta") for name in devices):
                    del self._model
                    self._model = None
                    raise MemoryError("模型包含位于 CPU/RAM 的权重，未完整加载到 GPU 显存")
            except StopIteration:
                pass
        self._model_info = self._extract_model_info(model_path, quantization=quantization, dtype=dtype)

        # 提取模型元信息
        self._update_model_metadata()
        try:
            actual_dtype = str(next(self._model.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            actual_dtype = dtype
        self._effective_load = {
            "engine": "hf", "dtype": actual_dtype,
            "context_length": self._model_info.context_length if self._model_info else max_model_len,
            "gpu_offload_layers": kwargs.get("gpu_offload_layers") if kwargs.get("gpu_offload_layers") is not None else -1,
            "batch_size": kwargs.get("batch_size", 1),
            "flash_attention": kwargs.get("flash_attention", True),
            "draft_model": kwargs.get("draft_model"),
            "speculative_decoding": kwargs.get("speculative_decoding", False),
            "tensor_parallel": tensor_parallel_size,
            "gpu_split": kwargs.get("gpu_split"),
            "trust_remote_code": trust_remote_code,
        }

        return self

    def _update_model_metadata(self) -> None:
        """更新模型参数量和上下文长度信息"""
        try:
            total_params = sum(p.numel() for p in self._model.parameters())
            self._model_info.parameters = f"{total_params/1e9:.1f}B" if total_params >= 1e9 else f"{total_params/1e6:.0f}M"
        except Exception:
            pass

        try:
            if hasattr(self._model.config, "max_position_embeddings"):
                self._model_info.context_length = self._model.config.max_position_embeddings
        except Exception:
            pass

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
        import torch

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 编码输入
        inputs = self._tokenizer(self._build_prompt(prompt, system_prompt), return_tensors="pt")
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

        return GenerationResult(
            text=self._tokenizer.decode(generated_ids, skip_special_tokens=True),
            tokens_generated=tokens_generated,
            time_seconds=elapsed,
            tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0,
            prompt_tokens=prompt_tokens,
        )

    def _build_prompt(self, prompt: str, system_prompt: str) -> str:
        """优先走 tokenizer 的 chat template 注入 system 段，没有则手工前置。"""
        if not system_prompt:
            return prompt
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError):
            return f"{system_prompt}\n\n{prompt}"

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        """HF特有: 追加模型参数内存占用明细"""
        usage = super().memory_usage(verbose)

        if verbose and self.is_loaded:
            usage.details = usage.details or {}
            # HF模型可直接统计参数占用
            try:
                param_bytes = sum(
                    p.nelement() * p.element_size() for p in self._model.parameters()
                )
                usage.details["model_param_mb"] = round(param_bytes / (1024 ** 2), 1)
                # 推理临时占用 = 总分配 - 参数占用
                usage.details["inference_temp_mb"] = round(
                    max(0, usage.gpu_allocated_mb - param_bytes / (1024 ** 2)), 1
                )
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
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model.to(device)

    def _get_gpu_device_index(self) -> int:
        """获取HF模型主设备索引"""
        if self.is_loaded and hasattr(self._model, 'device'):
            device = self._model.device
            if hasattr(device, 'index') and device.index is not None:
                return device.index
        return 0

    def release_cache(self) -> MemoryUsage:
        """HF特有: 清理past_key_values等推理缓存"""
        import torch

        # 清除HF generate产生的KV cache引用
        if self.is_loaded and hasattr(self._model, '_cache'):
            self._model._cache = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

        return self.memory_usage()

    def unload(self) -> None:
        """卸载模型，释放显存"""
        import gc
        import torch

        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._model_info = None
        self._effective_load = {}

        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
