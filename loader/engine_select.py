"""推理框架选择与创建。

按 ``spec.load.engine`` 显式选择;未配置且路径后缀也没有默认映射时,
打印错误并返回 ``None``(不抛异常)——调用方(``models_mgr._load``)据此
把模型标记为未加载,而不是让整条加载链路带着一个不存在的推理框架往下走。

不经过独立的 factory 模块——直接用 ``utils.common.require()`` 按模块路径
最后一段类名动态创建子类(``vllm``/``sglang``/``llama``),新增推理框架只需在
``llmFramework/`` 下加一个同名模块,不需要改这里的分支。

独立成模块(不放进 ``loader/__init__.py``)是为了避免 ``models_mgr`` 反向
导入包 ``__init__`` 造成循环导入。
"""

from __future__ import annotations

from ..utils.common import aContainB, err, require
from .llmFramework.baseInference import baseInference
from .model_spec import ModelSpec

# __package__ 就是本模块所在的 "loader" 包的完整点分路径(例如 "ai.loader" 或
# 直接执行时的 "server_models.loader")；用它拼出 llmFramework 子模块路径，
# 而不是硬编码顶层包名——repo 顶层目录名在不同部署方式下并不总是 "ai"
# (main.py 也是这样按 __package__ 二选一处理导入的)。
_ENGINE_MODULE = f"{__package__}.llmFramework.{{engine}}"
# 路径名里出现这些标记时默认用 vllm(量化格式对 llama.cpp 没有意义)。
_VLLM_PATH_HINTS = ("fp8", "awq", "gptq")


def detect_engine(path) -> str | None:
    """按文件/目录后缀推断默认推理框架:

    - ``.gguf`` 文件或含 ``*.gguf`` 的目录 -> llama
    - 含 ``config.json`` + ``*.safetensors`` 的标准 HF 目录 -> vllm
    - 路径名包含 fp8/awq/gptq -> vllm
    - 其余情况没有默认值,返回 None(调用方按此打印错误,不再继续加载)。
    """
    if path.suffix.lower() == ".gguf":
        return "llama"
    if path.is_dir():
        if any(path.glob("*.gguf")):
            return "llama"
        if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
            return "vllm"
    if aContainB(path.name.lower(), _VLLM_PATH_HINTS):
        return "vllm"
    return None


def create_loader(spec: ModelSpec) -> baseInference | None:
    """按 ``spec.load.engine`` 或路径后缀选择并创建推理框架实例。

    没有配置 engine 且路径也推断不出默认值时,打印错误并返回 None——
    不抛异常。"""
    engine = spec.engine or detect_engine(spec.path_obj)
    if not engine:
        err("无法确定推理框架，请在 models.json 配置 load.engine：model=", spec.model_id, " path=", spec.path)
        return None
    cls = require(_ENGINE_MODULE.format(engine=engine))
    return cls() if cls else None
