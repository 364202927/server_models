"""模型注册配置与首次加载参数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelLoadConfig:
    """模型第一次加载时使用的参数。缺少字段时使用引擎默认值。"""

    engine: str = "hf"
    dtype: str = "float16"
    context_length: int | None = None
    gpu_offload_layers: int = 0
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
    model_type: str = "auto"
    quantization: str | None = None
    estimated_vram_mb: int | None = None
    lora: list[dict[str, Any]] = field(default_factory=list)
    load: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def path_obj(self) -> Path:
        return Path(self.path)

    def to_config_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path, "load": self.load.to_dict()}
        if self.model_type != "auto":
            result["model_type"] = self.model_type
        if self.quantization is not None:
            result["quantization"] = self.quantization
        if self.estimated_vram_mb is not None:
            result["estimated_vram_mb"] = self.estimated_vram_mb
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
        "path", "model_type", "quantization", "estimated_vram_mb", "lora", "load",
        # 旧版本曾把这些字段放在模型根节点；读取时不再沿用，也不写回旧结构。
        "engine", "dtype", "context_length", "gpu_offload_layers", "batch_size",
        "flash_attention", "draft_model", "speculative_decoding", "tensor_parallel",
        "gpu_split", "trust_remote_code",
    }
    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"模型 {model_id} 配置必须是对象")
        if not raw.get("path"):
            raise ValueError(f"模型 {model_id} 缺少 path 配置")
        result[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            path=str(raw["path"]),
            model_type=str(raw.get("model_type", "auto")),
            quantization=raw.get("quantization"),
            estimated_vram_mb=raw.get("estimated_vram_mb"),
            lora=list(raw.get("lora", [])),
            load=ModelLoadConfig.from_dict(raw.get("load")),
            extra={key: value for key, value in raw.items() if key not in known},
        )
    return result
