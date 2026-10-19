__version__ = "0.1.0"
from enum import Enum
from .utils.hardware import detect_hardware, print_hardware_info, HardwareInfo
from .loader import (
    MemoryUsage, baseInference,
    load_model_specs, modelSpec, models_mgr, normalize_model_path,
)


class adim_ID(Enum):
    eHandshake = 1000          # 握手探活
    eStatus = 1001             # 系统/模型状态
    eHardwareInfo = 1002       # 硬件信息
    eModelsList = 1003         # 注册模型列表
    eSleep = 1004              # 指定模型休眠
    eUnload = 1005             # 指定模型卸载
    eEnsureLoaded = 1006       # 指定模型加载
    eUpdateGeneration = 1007   # 更新Generation参数
    eLogs = 1008               # 增量日志获取
    eAutolaunch = 1009         # 触发全局autolaunch加载
    eUnloadAll = 1010          # 触发全局卸载释放显存
    eStopped = 1099            # 停机保存


__all__ = [
    # Hardware
    "detect_hardware",
    "print_hardware_info",
    "HardwareInfo",
    "adim_ID",
    # Loader
    "baseInference",
    "MemoryUsage",
    "modelSpec",
    "models_mgr",
    "load_model_specs",
    "normalize_model_path",
]