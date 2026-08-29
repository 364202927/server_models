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
    """模型第一次加载时使用的参数。缺少字段时使用模型/后端默认值。"""

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
        }


@dataclass
class ModelSpec:
    """来自 ``assets/models.json`` 的模型定义。"""

    model_id: str
    path: str
    source_path: str | None = field(default=None, repr=False, compare=False)
    estimated_vram_mb: int | None = None
    lora: list[dict[str, Any]] = field(default_factory=list)
    load: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    cache: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    load_present: bool = field(default=False, repr=False, compare=False)
    cache_present: bool = field(default=False, repr=False, compare=False)
    load_data: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    cache_data: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    # JSON 中实际出现的 load 字段；不把 dataclass 默认值误认为用户配置。
    load_fields: set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def load_configured(self) -> bool:
        """兼容旧调用：只要至少有一个 load 字段即视为已配置。"""
        return bool(self.load_fields)

    @property
    def path_obj(self) -> Path:
        return Path(self.path)

    def to_config_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.source_path or self.path, "estimated_vram_mb": self.estimated_vram_mb}
        result["load"] = dict(self.load_data) if self.load_present else self.load.to_dict()
        result["cache"] = dict(self.cache_data) if self.cache_present else dict(self.cache)
        if self.lora:
            result["lora"] = self.lora
        result.update(self.extra)
        return result


def load_model_specs(config: dict[str, Any]) -> dict[str, ModelSpec]:
    """读取 ``models``，仅从每个模型自己的 ``load`` 节点取加载配置。"""
    result: dict[str, ModelSpec] = {}
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models.json 的 models 必须是对象")
    known = {
        "path", "estimated_vram_mb", "lora", "load", "cache", "model_type", "quantization", "engine",
        "dtype", "context_length", "gpu_offload_layers", "batch_size",
        "flash_attention", "draft_model", "speculative_decoding", "tensor_parallel",
        "gpu_split", "trust_remote_code",
    }
    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"模型 {model_id} 配置必须是对象")
        if not raw.get("path"):
            raise ValueError(f"模型 {model_id} 缺少 path 配置")
        raw_load = raw.get("load")
        result[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            path=normalize_model_path(str(raw["path"])),
            source_path=str(raw["path"]),
            estimated_vram_mb=raw.get("estimated_vram_mb"),
            lora=list(raw.get("lora", [])),
            load=ModelLoadConfig.from_dict(raw_load),
            cache={**CACHE_DEFAULTS, **(raw.get("cache", {}) if isinstance(raw.get("cache", {}), dict) else {})},
            load_present="load" in raw,
            cache_present="cache" in raw,
            load_data=dict(raw_load) if isinstance(raw_load, dict) else {},
            cache_data=dict(raw.get("cache", {})) if isinstance(raw.get("cache", {}), dict) else {},
            load_fields={
                key for key, value in (raw_load.items() if isinstance(raw_load, dict) else [])
                if key in ModelLoadConfig.__dataclass_fields__ and value is not None
            },
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
