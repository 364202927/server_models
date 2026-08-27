"""ModelsMgr 的无 GPU 单元测试骨架；真实模型测试需在 WSL2 GPU 环境执行。"""

from pathlib import Path

from ai.loader.models_mgr import ModelsMgr


def test_empty_registry(tmp_path: Path) -> None:
    config = tmp_path / "models.json"
    config.write_text('{"defaults": {}, "models": {}}', encoding="utf-8")
    assert ModelsMgr(str(config)).status()["models"] == []
