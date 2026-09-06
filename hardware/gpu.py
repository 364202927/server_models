"""NVIDIA GPU 查询和显存准入检查。"""

from __future__ import annotations

import subprocess
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


def query_gpu_used_mb() -> int:
    """读取整卡已用显存（MB），查不到时返回 0。

    llama.cpp 使用独立 CUDA 上下文，``torch.cuda`` 统计不到它的分配；而
    ``torch.cuda.memory_allocated()`` 又是进程级整卡累计，多模型同时驻留时无法
    归属到单个模型。因此统一用驱动级的 nvidia-smi，配合加载前后取差值。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode != 0:
            return 0
        return sum(int(float(line.strip())) for line in result.stdout.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def gpu_summary() -> dict[str, Any]:
    return {"gpus": [gpu.__dict__ for gpu in detect_gpu()]}
