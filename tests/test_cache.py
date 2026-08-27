from pathlib import Path

from ai.loader.cache import load_snapshot, save_snapshot
from ai.loader.model_spec import ModelSpec


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    spec = ModelSpec(model_id="x", path="/models/x")
    save_snapshot(tmp_path, spec, state="RUNNING")
    assert load_snapshot(tmp_path, "x")["runtime"]["state"] == "RUNNING"
