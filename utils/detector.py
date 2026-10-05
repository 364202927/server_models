"""WSL2/Windows 主机硬件与显存检测。"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    memory_allocated_mb: int
    memory_reserved_mb: int
    driver_version: str = "N/A"
    cuda_version: str = "N/A"
    compute_capability: str = "N/A"


@dataclass(frozen=True)
class MemoryInfo:
    total_mb: int
    available_mb: int

    @property
    def total_gb(self) -> float:
        return round(self.total_mb / 1024, 1)

    @property
    def available_gb(self) -> float:
        return round(self.available_mb / 1024, 1)


@dataclass(frozen=True)
class CPUInfo:
    name: str
    cores: int
    threads: int
    architecture: str


@dataclass(frozen=True)
class HardwareInfo:
    gpus: list[GPUInfo]
    memory: MemoryInfo
    platform: str
    cpu: CPUInfo

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cpu_cores(self) -> int:
        return self.cpu.cores


def detect_gpu() -> list[GPUInfo]:
    """优先通过 torch CUDA API，失败时回退到 nvidia-smi。"""
    try:
        import torch
        if torch.cuda.is_available():
            result: list[GPUInfo] = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total = int(props.total_memory / (1024 ** 2))
                allocated = int(torch.cuda.memory_allocated(index) / (1024 ** 2))
                reserved = int(torch.cuda.memory_reserved(index) / (1024 ** 2))
                # allocated 是本进程张量；reserved 还包含 PyTorch 缓存池，二者不能混淆。
                result.append(GPUInfo(index, props.name, total, max(0, total - reserved),
                                      allocated, reserved, cuda_version=torch.version.cuda or "N/A",
                                      compute_capability=f"{props.major}.{props.minor}"))
            return result
    except (ImportError, RuntimeError):
        pass
    return _detect_gpu_via_smi()


def _detect_gpu_via_smi() -> list[GPUInfo]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0:
            return []
        gpus: list[GPUInfo] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(GPUInfo(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3])),
                                int(float(parts[4])), int(float(parts[4])), driver_version=parts[5]))
        return gpus
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def detect_memory() -> MemoryInfo:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return MemoryInfo(int(mem.total / (1024 ** 2)), int(mem.available / (1024 ** 2)))
    except ImportError:
        pass
    if platform.system() == "Linux":
        try:
            values: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as file:
                for line in file:
                    key, value = line.split(":", 1)
                    if key in {"MemTotal", "MemAvailable"}:
                        values[key] = int(value.split()[0])
            return MemoryInfo(values.get("MemTotal", 0), values.get("MemAvailable", 0))
        except (OSError, ValueError):
            pass
    return MemoryInfo(0, 0)


def detect_hardware() -> HardwareInfo:
    threads = os.cpu_count() or 1
    cpu = CPUInfo(platform.processor() or "Unknown", threads, threads, platform.machine())
    return HardwareInfo(detect_gpu(), detect_memory(), f"{platform.system()} {platform.release()}", cpu)


def print_hardware_info(info: HardwareInfo | None = None) -> None:
    info = info or detect_hardware()
    print(info.to_dict())
