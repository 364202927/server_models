from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os, sys
from pathlib import Path
from typing import Any
from ..utils.common import aContainB


@dataclass
class ModelLoadConfig:
    dtype: str | None = None
    context_length: int | None = None
    gpu_memory_utilization: float | None = None
    quantization: str | None = None
    gpu_offload_layers: int | None = None
    batch_size: int | None = None
    flash_attention: bool | None = None
    enable_memory_saver: bool | None = None
    enable_sleep_mode: bool | None = None
    tensor_parallel: int | None = None
    gpu_split: list[float] | None = None
    trust_remote_code: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ModelLoadConfig":
        values = raw if isinstance(raw, dict) else {}
        known = set(cls.__dataclass_fields__) - {"extra"}
        return cls(
            **{key: value for key, value in values.items() if key in known},
            extra={
                key: value
                for key, value in values.items()
                if key not in known and key not in {"draft_model", "speculative_decoding"}
            },
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        extra = values.pop("extra")
        extra.pop("draft_model", None)
        extra.pop("speculative_decoding", None)
        values.update(extra)
        return values

    def loader_kwargs(self) -> dict[str, Any]:
        values = {
            "dtype": self.dtype,
            "max_model_len": self.context_length,
            "tensor_parallel_size": self.tensor_parallel,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization,
            "gpu_offload_layers": self.gpu_offload_layers,
            "batch_size": self.batch_size,
            "flash_attention": self.flash_attention,
            "enable_memory_saver": self.enable_memory_saver,
            "enable_sleep_mode": self.enable_sleep_mode,
            "gpu_split": self.gpu_split,
        }
        values.update({
            key: value
            for key, value in self.extra.items()
            if key not in {"draft_model", "speculative_decoding"}
        })
        return values


LOAD_KEYS = frozenset(ModelLoadConfig.__dataclass_fields__) - {"extra"}
WEIGHT_SUFFIXES = ("*.gguf", "*.safetensors", "*.bin")

LOAD_DEFAULTS: dict[str, Any] = {
    "dtype": "float16",
    "gpu_offload_layers": -1,
    "batch_size": 1,
    "flash_attention": True,
    "tensor_parallel_size": 1,
    "gpu_split": None,
    "trust_remote_code": True,
}
ENGINE_LOAD_DEFAULTS: dict[str, dict[str, Any]] = {
    "vllm": {"gpu_memory_utilization": 0.9, "enable_sleep_mode": True},
    "sglang": {"gpu_memory_utilization": 0.9, "enable_memory_saver": True},
}
LOAD_ARG_FIELDS = {"max_model_len": "context_length", "tensor_parallel_size": "tensor_parallel"}


@dataclass
class ModelSpec:
    model_id: str
    path: str
    source_path: str | None = field(default=None, repr=False, compare=False)
    estimated_vram_mb: int | None = None
    engine: str | None = None
    draft: str | None = None
    mtp: bool = False
    lora: str | None = None
    load: ModelLoadConfig = field(default_factory=ModelLoadConfig)
    generation: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    load_present: bool = field(default=False, repr=False, compare=False)
    generation_present: bool = field(default=False, repr=False, compare=False)
    load_fields: set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def load_configured(self) -> bool:
        return bool(self.load_fields)

    @property
    def path_obj(self) -> Path:
        return Path(self.path)

    def detect_engine(self) -> str | None:
        if self.engine:
            return self.engine.strip().lower()
        path = self.path_obj
        if path.suffix.lower() == ".gguf":
            return "llama"
        if path.is_dir():
            if any(path.glob("*.gguf")):
                return "llama"
            if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
                return "vllm"
        hints = ("fp8", "awq", "gptq")
        if aContainB(path.name.lower(), hints):
            return "vllm"
        return None

    def resolve_loader_kwargs(self, engine_name: str) -> dict[str, Any]:
        load_kwargs = self.load.loader_kwargs()
        defaults = {**LOAD_DEFAULTS, **ENGINE_LOAD_DEFAULTS.get(engine_name, {})}
        for name, value in defaults.items():
            if name not in load_kwargs or load_kwargs[name] is None:
                load_kwargs[name] = value
            field_name = LOAD_ARG_FIELDS.get(name, name)
            if field_name in LOAD_KEYS and getattr(self.load, field_name) is None:
                setattr(self.load, field_name, load_kwargs[name])
        return load_kwargs

    def apply_effective_load(self, effective: dict[str, Any], model_info: Any = None) -> None:
        if model_info:
            effective.setdefault("context_length", model_info.context_length)
            effective.setdefault("dtype", model_info.dtype)
        for name, value in effective.items():
            if name in LOAD_KEYS:
                setattr(self.load, name, value)
        accepted_extra = set(effective) - LOAD_KEYS - {
            "engine", "draft", "mtp", "lora", "speculative_model",
            "speculative_draft_model_path", "lora_paths", "enable_lora",
        }
        self.load.extra = {k: v for k, v in self.load.extra.items() if k in accepted_extra}
        self.load.extra.update({k: effective[k] for k in accepted_extra})
        self.load_fields.update(name for name in effective if name in LOAD_KEYS or name in self.load.extra)
        self.load_present = True

    def update_estimated_vram(self, used_before: int, current_used: int, force: bool = False) -> bool:
        if self.estimated_vram_mb is not None and not force:
            return False
        delta = max(0, current_used - used_before)
        if delta <= 0:
            return False
        self.estimated_vram_mb = int(delta)
        return True

    def required_vram_mb(self) -> int | None:
        """显存准入估算：在静态实测值或磁盘大小基础上留出 1024MB 运行时余量。"""
        if self.estimated_vram_mb is not None:
            return int(self.estimated_vram_mb + 1024)
        return self.size_on_disk_mb()

    def size_on_disk_mb(self) -> int | None:
        """根据权重文件大小按 1.25x 估算显存占用，包含上下文开销与基础 KV Cache。"""
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
        return int(total / (1024 ** 2) * 1.25) if total > 0 else None

    def to_config_node(self) -> dict[str, Any]:
        node: dict[str, Any] = {
            "path": self.source_path or self.path,
            "mtp": self.mtp,
            "load": self.load.to_dict(),
            "generation": self.generation,
        }
        if self.estimated_vram_mb is not None:
            node["estimated_vram_mb"] = self.estimated_vram_mb
        if self.engine is not None:
            node["engine"] = self.engine
        if self.draft is not None:
            node["draft"] = self.draft
        if self.lora is not None:
            node["lora"] = self.lora
        node.update(self.extra)
        return node


def load_model_specs(config: dict[str, Any]) -> dict[str, ModelSpec]:
    result: dict[str, ModelSpec] = {}
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models.json 的 models 必须是对象")
    known = {"path", "estimated_vram_mb", "engine", "draft", "mtp", "lora", "load", "generation"}
    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"模型 {model_id} 配置必须是对象")
        if not raw.get("path"):
            raise ValueError(f"模型 {model_id} 缺少 path 配置")
        raw_load = raw.get("load")
        raw_generation = raw.get("generation")
        raw_engine = raw.get("engine")
        raw_draft = raw.get("draft")
        raw_mtp = raw.get("mtp", False)
        raw_lora = raw.get("lora")
        if raw_draft is not None and not isinstance(raw_draft, str):
            raise ValueError(f"模型 {model_id} 的 draft 必须是地址字符串或 null")
        if not isinstance(raw_mtp, bool):
            raise ValueError(f"模型 {model_id} 的 mtp 必须是布尔值")
        if raw_lora is not None and not isinstance(raw_lora, str):
            raise ValueError(f"模型 {model_id} 的 lora 必须是文件路径字符串或 null")
        result[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            path=normalize_model_path(str(raw["path"])),
            source_path=str(raw["path"]),
            estimated_vram_mb=raw.get("estimated_vram_mb"),
            engine=str(raw_engine).strip().lower() if raw_engine else None,
            draft=normalize_model_path(raw_draft) if raw_draft else None,
            mtp=raw_mtp,
            lora=normalize_model_path(raw_lora) if raw_lora else None,
            load=ModelLoadConfig.from_dict(raw_load),
            generation=dict(raw_generation) if isinstance(raw_generation, dict) else {},
            load_present=isinstance(raw_load, dict),
            generation_present=isinstance(raw_generation, dict),
            load_fields={
                key for key, value in (raw_load.items() if isinstance(raw_load, dict) else [])
                if value is not None
            },
            extra={key: value for key, value in raw.items() if key not in known},
        )
    return result


def normalize_model_path(path: str) -> str:
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