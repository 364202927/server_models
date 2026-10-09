from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 汇总 detector（硬件探测）与 process（进程资源）的公共接口，
# 本模块是硬件相关的唯一对外的门面。
from .detector import (
    CPUInfo,
    GPUInfo,
    HardwareInfo,
    MemoryInfo,
    detect_gpu,
    detect_hardware,
    detect_memory,
    print_hardware_info,
)
from .process import process_memory_mb, process_summary


# ==========================================
# 内存（RAM）准入判定
# ==========================================

@dataclass(frozen=True)
class RamDecision:
    allowed: bool
    available_mb: int
    required_mb: int
    reason: str = ""


def check_ram(required_mb: int, reserve_mb: int = 8192) -> RamDecision:
    available = detect_memory().available_mb
    required = max(0, required_mb) + max(0, reserve_mb)
    return RamDecision(
        available >= required,
        available,
        required_mb,
        "RAM 不足" if available < required else "",
    )


# ==========================================
# 显存（VRAM）准入判定
# ==========================================

@dataclass(frozen=True)
class GpuMemoryDecision:
    allowed: bool
    required_mb: int
    available_mb: int
    reason: str = ""


def gpu_memory() -> list[GPUInfo]:
    return detect_gpu()


def check_gpu_memory(required_mb: int, reserve_mb: int = 512) -> GpuMemoryDecision:
    """检查所有可见 GPU 的总空闲显存是否满足模型需求。"""
    available = sum(gpu.memory_free_mb for gpu in detect_gpu())
    required = max(0, required_mb) + max(0, reserve_mb)
    return GpuMemoryDecision(
        available >= required,
        required_mb,
        available,
        "显存不足" if available < required else "",
    )


# 显卡已用显存
def query_gpu_used_mb() -> int:
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        total_used_bytes = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_used_bytes += mem_info.used
        pynvml.nvmlShutdown()
        return total_used_bytes // (1024 * 1024)
    except Exception:
        return 0


def gpu_summary() -> dict[str, Any]:
    return {"gpus": [gpu.__dict__ for gpu in detect_gpu()]}

def query_gpu_free_mb() -> int:
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        total_free_bytes = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_free_bytes += mem_info.free
        pynvml.nvmlShutdown()
        return total_free_bytes // (1024 * 1024)
    except Exception:
        # 如果 NVML 不可用，回退使用 detector 探测出的显存信息
        return sum(gpu.memory_free_mb for gpu in detect_gpu())

__all__ = [
    # detector：硬件探测
    "CPUInfo",
    "GPUInfo",
    "HardwareInfo",
    "MemoryInfo",
    "detect_gpu",
    "detect_hardware",
    "detect_memory",
    "print_hardware_info",
    # 显存准入
    "GpuMemoryDecision",
    "gpu_memory",
    "check_gpu_memory",
    "query_gpu_used_mb",
    "gpu_summary",
    # 内存准入
    "RamDecision",
    "check_ram",
    # process：进程资源
    "process_memory_mb",
    "process_summary",
    #
    'query_gpu_free_mb',
]