# AI Model Benchmark for Quantitative Trading
# 量化交易AI模型评估框架

__version__ = "0.1.0"

from .utils.config import BenchmarkConfig, TestDimension
from .hardware import detect_hardware, print_hardware_info, HardwareInfo
from .loader import ModelLoader, MemoryUsage, HFLoader, GGUFLoader, VLLMLoader
from .msgHandler import MsgHandler
from .benchmark import BenchmarkRunner, BenchmarkResult, Scorer, ReportGenerator

__all__ = [
    # Config
    "BenchmarkConfig",
    "TestDimension",
    # Hardware
    "detect_hardware",
    "print_hardware_info",
    "HardwareInfo",
    # Loader
    "ModelLoader",
    "MemoryUsage",
    "HFLoader",
    "GGUFLoader",
    "VLLMLoader",
    "MsgHandler",
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
    "Scorer",
    "ReportGenerator",
]
