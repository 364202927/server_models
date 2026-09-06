"""无 GPU 单元测试的公共夹具。

真实显存行为只能在目标机（Win11 + NVIDIA）验证，这里用假 Loader + 打桩的显存
查询覆盖 ModelsMgr 的准入、释放、配置读写逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.loader import models_mgr as mgr_module
from ai.loader.base import GenerationResult, ModelLoader
from ai.loader.models_mgr import ModelsMgr


class FakeLoader(ModelLoader):
    """按固定显存占用记账的假 Loader。"""

    # 由 FakeGpu 在 load/unload 时读取。
    vram_mb = 0
    can_sleep = True

    def __init__(self, gpu: "FakeGpu", vram_mb: int, can_sleep: bool = True) -> None:
        super().__init__()
        self._gpu = gpu
        self.vram_mb = vram_mb
        self.can_sleep = can_sleep
        self.wake_calls = 0

    def load(self, model_path: str, **kwargs: Any) -> "FakeLoader":
        self._model = object()
        self._effective_load = {"dtype": "float16", "context_length": 4096}
        self._gpu.used += self.vram_mb
        return self

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        self.last_generate_kwargs = kwargs
        return GenerationResult(f"echo:{prompt}", 1, 0.1, 10.0, 1)

    def sleep_to_ram(self) -> bool:
        if not self.can_sleep:
            return False
        self._gpu.used -= self.vram_mb
        return True

    def wake(self) -> None:
        self.wake_calls += 1
        self._gpu.used += self.vram_mb

    def unload(self) -> None:
        if self._model is not None:
            self._gpu.used -= self.vram_mb
        self._model = None


class FakeGpu:
    """可控的显存账本，替换 nvidia-smi 查询。"""

    def __init__(self, total_mb: int = 32768) -> None:
        self.total = total_mb
        self.used = 0

    @property
    def free(self) -> int:
        return self.total - self.used


@pytest.fixture
def gpu() -> FakeGpu:
    return FakeGpu()


@pytest.fixture
def make_mgr(tmp_path: Path, gpu: FakeGpu, monkeypatch):
    """构造一个所有硬件查询都被打桩的 ModelsMgr。"""

    created: dict[str, FakeLoader] = {}

    def _build(config: dict[str, Any], vram: dict[str, int] | None = None,
               can_sleep: bool = True, ram_available_mb: int = 1_000_000) -> ModelsMgr:
        path = tmp_path / "models.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        sizes = vram or {}

        def fake_create_loader(spec):
            loader = FakeLoader(gpu, sizes.get(spec.model_id, 1024), can_sleep)
            created[spec.model_id] = loader
            return loader

        def fake_detect_gpu():
            return [type("G", (), {"memory_free_mb": gpu.free, "index": 0})()]

        monkeypatch.setattr(mgr_module, "create_loader", fake_create_loader)
        monkeypatch.setattr(mgr_module, "query_gpu_used_mb", lambda: gpu.used)
        monkeypatch.setattr(mgr_module, "detect_gpu", fake_detect_gpu)
        monkeypatch.setattr(
            mgr_module, "check_gpu_memory",
            lambda required, reserve=512: type("D", (), {"allowed": gpu.free >= required + reserve})(),
        )
        monkeypatch.setattr(
            mgr_module, "check_ram",
            lambda required, reserve=8192: type("D", (), {"allowed": ram_available_mb >= required + reserve})(),
        )
        return ModelsMgr(str(path))

    _build.loaders = created
    _build.gpu = gpu
    return _build


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "models.json"
