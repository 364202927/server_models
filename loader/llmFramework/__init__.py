"""推理框架实现：``baseInference`` 抽象基类 + vllm/sglang/llama 三个子类。

子类通过 ``utils.common.require()`` 按模块路径最后一段的类名动态创建
（见 ``loader/__init__.py`` 的 ``create_loader``），因此不在此处显式导入
三个子类模块——按需 import 才能保持"未安装的推理框架不影响其它框架"的
可选依赖隔离。
"""

from .baseInference import (
    GenerationResult, MemoryUsage, ModelInfo, ToolCapabilityError, ToolOutputError,
    baseInference, detect_model_type,
)

__all__ = [
    "baseInference", "ModelInfo", "GenerationResult", "MemoryUsage",
    "ToolCapabilityError", "ToolOutputError", "detect_model_type",
]
