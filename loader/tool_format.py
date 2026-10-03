"""
tool_format.py
模型快照缓存、推理框架选择——两块合并到一个模块。
"""


import json
from pathlib import Path
from typing import Any

from ..utils.common import aContainB, error, require
from .llmFramework.baseInference import baseInference
from .model_spec import ModelSpec


# ---------------------------------------------------------------------------
# 模型状态快照（原 cache.py）：快照只记录模型自身的身份和运行时状态，
# 缓存策略在 defaults 里配置。
# ---------------------------------------------------------------------------


def snapshot_path(root: str | Path, model_id: str) -> Path:
    return Path(root) / "runtime" / "snapshots" / f"{model_id}.json"


def save_snapshot(root: str | Path, spec: ModelSpec, **runtime: Any) -> Path:
    path = snapshot_path(root, spec.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": spec.model_id, "path": spec.path,
               "dtype": spec.load.dtype, "runtime": runtime}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_snapshot(root: str | Path, model_id: str) -> dict[str, Any] | None:
    path = snapshot_path(root, model_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# 推理框架选择与创建（原 engine_select.py）：按 spec.load.engine 显式选择；
# 未配置且路径后缀也没有默认映射时，打印错误并返回 None（不抛异常）——
# 调用方（models_mgr._load）据此把模型标记为未加载，而不是让整条加载链路
# 带着一个不存在的推理框架往下走。
#
# 不经过独立的 factory 模块——直接用 utils.common.require() 按模块路径
# 最后一段类名动态创建子类（vllm/sglang/llama），新增推理框架只需在
# llmFramework/ 下加一个同名模块，不需要改这里的分支。
# ---------------------------------------------------------------------------

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
        error("无法确定推理框架，请在 models.json 配置 load.engine：model=", spec.model_id, " path=", spec.path)
        return None
    cls = require(_ENGINE_MODULE.format(engine=engine))
    return cls() if cls else None
