from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ..utils.common import aContainB


@dataclass
class ModelSpec:
    model_id: str
    path: str
    source_path: str | None = field(default=None, repr=False, compare=False)
    estimated_vram_mb: int | None = None
    engine: str | None = None
    load: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

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

    def update_estimated_vram(self, used_before: int, current_used: int, force: bool = False) -> bool:
        if self.estimated_vram_mb is not None and not force:
            return False
        delta = max(0, current_used - used_before)
        if delta <= 0:
            return False
        self.estimated_vram_mb = int(delta)
        return True

    def required_vram_mb(self) -> int | None:
        if self.estimated_vram_mb is not None:
            return int(self.estimated_vram_mb + 1024)
        return self.size_on_disk_mb()

    def size_on_disk_mb(self) -> int | None:
        path = self.path_obj
        try:
            if path.is_file():
                total = path.stat().st_size
            elif path.is_dir():
                total = sum(item.stat().st_size for item in path.glob("*") if item.is_file())
            else:
                return None
        except OSError:
            return None
        return int(total / (1024 ** 2) * 1.25) if total > 0 else None

    def to_config_node(self) -> dict[str, Any]:
        node: dict[str, Any] = {
            "path": self.source_path or self.path,
            "engine": self.engine,
            "load": self.load,
            "generation": self.generation,
        }
        if self.estimated_vram_mb is not None:
            node["estimated_vram_mb"] = self.estimated_vram_mb
        return node


def load_model_specs(config: dict[str, Any]) -> dict[str, ModelSpec]:
    result: dict[str, ModelSpec] = {}
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models.json 的 models 必须是对象")

    for model_id, raw in models.items():
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        result[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            path=normalize_model_path(str(raw["path"])),
            source_path=str(raw["path"]),
            estimated_vram_mb=raw.get("estimated_vram_mb"),
            engine=str(raw.get("engine", "")).strip().lower() or None,
            load=dict(raw.get("load", {})),
            generation=dict(raw.get("generation", {})),
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