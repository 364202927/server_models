"""模型注册配置与首次加载参数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import sys
from pathlib import Path
from typing import Any


CACHE_DEFAULTS: dict[str, Any] = {
    "state_snapshot_enabled": False,
    "prompt_cache_enabled": False,
    "kv_cache_enabled": False,
    "cache_dir": "assets/cache",
}


@dataclass
class ModelLoadConfig:
    """模型第一次加载时使用的参数。

    HF 工具调用显式配置 ``tool_parser: hermes_json``；GGUF 配置
    ``chat_format: chatml-function-calling``。未配置的模型只支持文本生成。
    """

    dtype: str = "float16"
    context_length: int | None = None
    # None 表示使用引擎默认；GGUF 默认会尽可能把层放到 GPU。
    gpu_offload_layers: int | None = None
    batch_size: int = 1
    flash_attention: bool = True
    draft_model: str | None = None
    speculative_decoding: bool = False
    tensor_parallel: int = 1
    gpu_split: list[float] | None = None
    trust_remote_code: bool = True
    tool_parser: str | None = None
    chat_format: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ModelLoadConfig":
        values = raw if isinstance(raw, dict) else {}
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def loader_kwargs(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "max_model_len": self.context_length,
            "tensor_parallel_size": self.tensor_parallel,
            "trust_remote_code": self.trust_remote_code,
            "gpu_offload_layers": self.gpu_offload_layers,
            "batch_size": self.batch_size,
            "flash_attention": self.flash_attention,
            "draft_model": self.draft_model,
            "speculative_decoding": self.speculative_decoding,
            "gpu_split": self.gpu_split,
            "tool_parser": self.tool_parser,
            "chat_format": self.chat_format,
        }


# 会影响模型创建、改动后必须重新加载的参数。
LOAD_KEYS = frozenset(ModelLoadConfig.__dataclass_fields__)

# 权重文件后缀；估算值缺失时按文件大小推算显存下界。
WEIGHT_SUFFIXES = ("*.gguf", "*.safetensors", "*.bin")


@dataclass
class ModelSpec:
    """来自 ``assets/models.json`` 的模型定义。"""

    model_id: str
    path: str
    source_path: str | None = field(default=None, repr=False, compare=False)
    estimated_vram_mb: int | None = None
    lora: list[dict[str, Any]] = field(default_factory=list)
    load: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    generation: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    load_present: bool = field(default=False, repr=False, compare=False)
    generation_present: bool = field(default=False, repr=False, compare=False)
    # JSON 中实际出现的 load 字段；不把 dataclass 默认值误认为用户配置。
    load_fields: set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def load_configured(self) -> bool:
        """兼容旧调用：只要至少有一个 load 字段即视为已配置。"""
        return bool(self.load_fields)

    @property
    def path_obj(self) -> Path:
        return Path(self.path)

    def size_on_disk_mb(self) -> int | None:
        """按权重文件大小估算显存下界，用于 ``estimated_vram_mb`` 缺失时的准入兜底。

        没有它的话首次加载会直接跳过显存检查，撑不下时表现为硬 OOM 而不是先回收。
        """
        path = self.path_obj
        try:
            if path.is_file():
                total = path.stat().st_size
            elif path.is_dir():
                files: list[Path] = []
                for pattern in WEIGHT_SUFFIXES:
                    files = sorted(path.glob(pattern))
                    if files:
                        break
                total = sum(item.stat().st_size for item in files)
            else:
                return None
        except OSError:
            return None
        # 留 10% 给上下文和推理临时分配。
        return int(total / (1024 ** 2) * 1.1) if total > 0 else None


def load_model_specs(config: dict[str, Any]) -> dict[str, ModelSpec]:
    """读取 ``models``，仅从每个模型自己的 ``load``/``generation`` 节点取配置。"""
    result: dict[str, ModelSpec] = {}
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models.json 的 models 必须是对象")
    known = {"path", "estimated_vram_mb", "lora", "load", "generation"}
    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"模型 {model_id} 配置必须是对象")
        if not raw.get("path"):
            raise ValueError(f"模型 {model_id} 缺少 path 配置")
        raw_load = raw.get("load")
        raw_generation = raw.get("generation")
        result[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            path=normalize_model_path(str(raw["path"])),
            source_path=str(raw["path"]),
            estimated_vram_mb=raw.get("estimated_vram_mb"),
            lora=list(raw.get("lora", [])),
            load=ModelLoadConfig.from_dict(raw_load),
            generation=dict(raw_generation) if isinstance(raw_generation, dict) else {},
            load_present=isinstance(raw_load, dict),
            generation_present=isinstance(raw_generation, dict),
            load_fields={
                key for key, value in (raw_load.items() if isinstance(raw_load, dict) else [])
                if key in LOAD_KEYS and value is not None
            },
            # 未识别的键原样保留，持久化时不会被丢弃。
            extra={key: value for key, value in raw.items() if key not in known},
        )
    return result


def normalize_model_path(path: str) -> str:
    """将配置中的 Windows 路径转换为当前平台可访问的路径。

    WSL/Linux 下 ``D:/x`` 或 ``D:\\x`` 映射为 ``/mnt/d/x``；Windows 保留盘符路径。
    其他 POSIX/UNC 路径不做猜测性修改。
    """
    value = os.path.expandvars(path.strip())
    if sys.platform == "win32":
        return value.replace("/", "\\") if len(value) >= 2 and value[1] == ":" else value
    if len(value) >= 2 and value[1] == ":":
        drive = value[0].lower()
        tail = value[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{tail}"
    if value.startswith("\\\\"):
        return "/" + value.lstrip("\\").replace("\\", "/")
    return value.replace("\\", "/") if "\\" in value else value
