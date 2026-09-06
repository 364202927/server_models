"""
模型加载器抽象基类

定义统一的模型加载和推理接口，支持:
- HuggingFace Transformers (适合测试)
- vLLM (高性能推理，适合生产)
"""

import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import GenerationParams, detect_model_type


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    model_type: str
    quantization: str | None = None
    dtype: str = "float16"
    parameters: str = "unknown"         # 参数量 e.g. "7B"
    context_length: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    prompt_tokens: int = 0


@dataclass
class MemoryUsage:
    """模型内存/显存占用信息"""
    gpu_allocated_mb: float = 0       # GPU已分配显存 (模型+推理临时)
    gpu_reserved_mb: float = 0        # GPU预留显存 (含缓存池)
    gpu_total_mb: float = 0           # GPU总显存
    gpu_free_mb: float = 0            # GPU空闲显存
    process_rss_mb: float = 0         # 进程物理内存占用
    system_available_mb: float = 0    # 系统可用内存
    details: dict[str, Any] | None = None  # verbose模式下的详细信息


class ModelLoader(ABC):
    """
    模型加载器抽象基类

    使用方式:
        with HFLoader() as loader:
            loader.load("model_path")
            result = loader.generate("prompt")
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_info: ModelInfo | None = None
        self._effective_load: dict[str, Any] = {}

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def effective_load(self) -> dict[str, Any]:
        """返回本次加载实际采用的参数，供 ModelsMgr 同步到 models.json。"""
        return dict(self._effective_load)

    @abstractmethod
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
    ) -> "ModelLoader":
        """加载模型，返回self支持链式调用"""
        pass

    @abstractmethod
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
        """生成文本，返回GenerationResult；system_prompt 非空时作为系统角色注入"""
        pass

    def generate_with_params(
        self,
        prompt: str,
        params: GenerationParams,
        **kwargs
    ) -> GenerationResult:
        """使用GenerationParams配置生成"""
        return self.generate(
            prompt,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            **kwargs
        )

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放显存"""
        pass

    def sleep_to_ram(self) -> bool:
        """将模型权重移出 GPU 保留在 RAM；引擎不支持时返回 False。"""
        return False

    def wake(self) -> None:
        """唤醒 RAM 中的模型；不支持休眠的 Loader 无需实现。"""
        return None

    def supports_state_snapshot(self) -> bool:
        return True

    def supports_prompt_cache(self) -> bool:
        return False

    def supports_kv_cache_persistence(self) -> bool:
        return False

    def load_lora(self, paths: list[str]) -> None:
        """加载适配器；具体引擎不支持时显式报告，避免静默误用。"""
        raise NotImplementedError("当前模型引擎不支持 LoRA")

    def unload_lora(self) -> None:
        """卸载已加载的适配器。"""
        return None

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        """
        检测当前模型内存/显存占用

        Args:
            verbose: True时返回详细信息(各GPU设备、模型参数量等)
        """
        usage = MemoryUsage()

        # GPU显存 - 通过torch.cuda获取
        try:
            import torch
            if torch.cuda.is_available():
                device_index = self._get_gpu_device_index()
                usage.gpu_allocated_mb = round(torch.cuda.memory_allocated(device_index) / (1024 ** 2), 1)
                usage.gpu_reserved_mb = round(torch.cuda.memory_reserved(device_index) / (1024 ** 2), 1)
                total = torch.cuda.get_device_properties(device_index).total_memory
                usage.gpu_total_mb = round(total / (1024 ** 2), 1)
                usage.gpu_free_mb = round(usage.gpu_total_mb - usage.gpu_reserved_mb, 1)

                if verbose:
                    usage.details = usage.details or {}
                    # 多GPU时列出各设备占用
                    if torch.cuda.device_count() > 1:
                        gpu_details = []
                        for i in range(torch.cuda.device_count()):
                            gpu_details.append({
                                "device": i,
                                "name": torch.cuda.get_device_properties(i).name,
                                "allocated_mb": round(torch.cuda.memory_allocated(i) / (1024 ** 2), 1),
                                "reserved_mb": round(torch.cuda.memory_reserved(i) / (1024 ** 2), 1),
                            })
                        usage.details["gpu_devices"] = gpu_details
        except ImportError:
            pass

        # 进程内存 - 通过psutil或/proc获取
        usage.process_rss_mb = round(self._get_process_rss_mb(), 1)
        usage.system_available_mb = round(self._get_system_available_mb(), 1)

        if verbose:
            usage.details = usage.details or {}
            usage.details["model_loaded"] = self.is_loaded
            if self._model_info:
                usage.details["model_name"] = self._model_info.name
                usage.details["parameters"] = self._model_info.parameters

        return usage

    def release_cache(self) -> MemoryUsage:
        """
        释放推理过程中产生的临时显存/内存占用 (KV cache, 临时张量等)
        保持模型本身已加载状态不变

        Returns:
            清理后的MemoryUsage
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
        return self.memory_usage()

    def _get_gpu_device_index(self) -> int:
        """获取模型所在的GPU设备索引，子类可覆盖"""
        return 0

    @staticmethod
    def _get_process_rss_mb() -> float:
        """获取当前进程物理内存占用(MB)"""
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except ImportError:
            pass
        # psutil不可用，Linux下读/proc/self/status
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_system_available_mb() -> float:
        """获取系统可用内存(MB)"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 2)
        except ImportError:
            pass
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except Exception:
            pass
        return 0

    def _extract_model_info(self, model_path: str, **kwargs) -> ModelInfo:
        """从路径提取模型信息"""
        path = Path(model_path)
        name = path.name if path.exists() else model_path.split("/")[-1]
        return ModelInfo(
            name=name,
            path=model_path,
            model_type=detect_model_type(name),
            quantization=kwargs.get("quantization"),
            dtype=kwargs.get("dtype", "float16"),
        )

    def __enter__(self) -> "ModelLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()
