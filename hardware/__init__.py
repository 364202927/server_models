"""硬件检测公共接口。"""

from .detector import CPUInfo, GPUInfo, HardwareInfo, MemoryInfo, detect_gpu, detect_hardware, detect_memory, print_hardware_info
from .gpu import GpuMemoryDecision, check_gpu_memory, gpu_memory, gpu_summary
from .memory import RamDecision, check_ram
from .process import process_memory_mb, process_summary

__all__ = ["CPUInfo", "GPUInfo", "HardwareInfo", "MemoryInfo", "detect_gpu", "detect_hardware", "detect_memory", "print_hardware_info",
           "GpuMemoryDecision", "check_gpu_memory", "gpu_memory", "gpu_summary", "RamDecision", "check_ram",
           "process_memory_mb", "process_summary"]
