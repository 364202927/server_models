__version__ = "0.1.0"
from .utils.hardware import detect_hardware, print_hardware_info, HardwareInfo
from .loader import (
    MemoryUsage, ModelSpec, ModelsMgr, RuntimeModel, baseInference,
    load_model_specs, normalize_model_path,
)

from enum import Enum

class adim_ID(Enum):
    eStatus = 1001
    eHardwareInfo = 1002
    eModelsList = 1003
    eSleep = 1004           #休眠
    eUnload = 1005          #卸载模型
    eEnsureLoaded = 1006    #加载模型
    eUpdateGeneration = 1007#更新Generation参数

__all__ = [
    # Hardware
    "detect_hardware",
    "print_hardware_info",
    "HardwareInfo",
    "adim_ID",
    # Loader
    "baseInference",
    "MemoryUsage",
    "ModelSpec",
    "load_model_specs",
    "normalize_model_path",
    "ModelsMgr",
    "RuntimeModel",
]