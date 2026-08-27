"""NVIDIA GPU 查询和显存准入检查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .detector import GPUInfo, detect_gpu


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
    return GpuMemoryDecision(available >= required, required_mb, available,
                             "显存不足" if available < required else "")


def gpu_summary() -> dict[str, Any]:
    return {"gpus": [gpu.__dict__ for gpu in detect_gpu()]}
