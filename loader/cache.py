"""模型状态快照和缓存能力的安全封装。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_spec import ModelSpec


def snapshot_path(root: str | Path, model_id: str) -> Path:
    return Path(root) / "runtime" / "snapshots" / f"{model_id}.json"


def save_snapshot(root: str | Path, spec: ModelSpec, **runtime: Any) -> Path:
    path = snapshot_path(root, spec.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": spec.model_id, "path": spec.path, "engine": spec.load.engine,
               "model_type": spec.model_type, "dtype": spec.load.dtype,
               "quantization": spec.quantization, "runtime": runtime}
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
