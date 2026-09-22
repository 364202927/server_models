# AI 模型服务与量化交易能力评估框架
# server_models package

__version__ = "0.1.0"

from .hardware import detect_hardware, print_hardware_info, HardwareInfo
from .loader import (
    MemoryUsage, ModelSpec, ModelsMgr, RuntimeModel, baseInference,
    create_loader, load_model_specs, normalize_model_path,
)

__all__ = [
    # Hardware
    "detect_hardware",
    "print_hardware_info",
    "HardwareInfo",
    # Loader
    "baseInference",
    "MemoryUsage",
    "ModelSpec",
    "load_model_specs",
    "normalize_model_path",
    "create_loader",
    "ModelsMgr",
    "RuntimeModel",
]
