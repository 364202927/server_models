"""多模型生命周期管理器：显存准入、RAM 休眠、卸载和空闲回收。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from ..hardware import check_gpu_memory, check_ram, detect_hardware
from ..utils.common import readFile
from .base import GenerationResult, ModelLoader
from .factory import create_loader
from .cache import save_snapshot
from .lora import load_lora
from .model_spec import ModelSpec, load_model_specs


@dataclass
class RuntimeModel:
    spec: ModelSpec
    loader: ModelLoader | None = None
    state: str = "UNLOADED"
    last_used_at: float = field(default_factory=time.time)
    active: bool = False
    error: str | None = None
    sleep_location: str | None = None

    def public_dict(self) -> dict[str, Any]:
        usage = self.loader.memory_usage() if self.loader else None
        return {
            "id": self.spec.model_id,
            "state": self.state,
            "engine": self.spec.engine,
            "path": self.spec.path,
            "last_used_at": datetime.fromtimestamp(self.last_used_at, tz=timezone.utc).isoformat(),
            "gpu_memory_mb": usage.gpu_allocated_mb if usage else 0,
            "sleep_location": self.sleep_location,
            "error": self.error,
        }


class ModelsMgr:
    """串行场景下的模型注册表和运行时实例管理器。"""

    def __init__(self, config_path: str = "assets/models.json") -> None:
        config = readFile(config_path) or {}
        self.settings: dict[str, Any] = config.get("defaults", {})
        self.specs: dict[str, ModelSpec] = load_model_specs(config)
        self.runtime: dict[str, RuntimeModel] = {}
        self.asset_root = str(config_path).rsplit("/", 1)[0] or "."
        self._lock = threading.RLock()

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.runtime.get(model_id, RuntimeModel(spec)).public_dict()
                    for model_id, spec in self.specs.items()]

    def status(self) -> dict[str, Any]:
        return {"models": self.list_models(), "queue_length": 0}

    def _get_runtime(self, model_id: str) -> RuntimeModel:
        if model_id not in self.specs:
            raise KeyError(f"models.json 中不存在模型: {model_id}")
        return self.runtime.setdefault(model_id, RuntimeModel(self.specs[model_id]))

    def _free_for(self, spec: ModelSpec) -> bool:
        required = spec.estimated_vram_mb
        if required is None:
            return True
        reserve = int(self.settings.get("sleep", {}).get("gpu_reserve_mb", 512))
        return check_gpu_memory(required, reserve).allowed

    def _reclaim(self, exclude: str) -> None:
        candidates = [item for key, item in self.runtime.items()
                      if key != exclude and item.loader and not item.active and item.state == "RUNNING"]
        candidates.sort(key=lambda item: (item.last_used_at, -(item.loader.memory_usage().gpu_allocated_mb)))
        for item in candidates:
            ram_reserve = int(self.settings.get("sleep", {}).get("ram_reserve_mb", 8192))
            if item.loader and self.settings.get("sleep", {}).get("ram_enabled", True) and check_ram(int(item.spec.estimated_vram_mb or 0), ram_reserve).allowed and item.loader.sleep_to_ram():
                item.state = "SLEEPING_RAM"
                item.sleep_location = "ram"
            else:
                self._unload_runtime(item.spec.model_id)
            if self._free_for(self.specs[exclude]):
                return

    def ensure_loaded(self, model_id: str) -> RuntimeModel:
        with self._lock:
            runtime = self._get_runtime(model_id)
            if runtime.loader and runtime.state in {"RUNNING", "SLEEPING_RAM"}:
                if runtime.state == "SLEEPING_RAM":
                    runtime.loader.wake()
                runtime.state = "RUNNING"
                runtime.last_used_at = time.time()
                return runtime
            runtime.state = "LOADING"
            try:
                if not self._free_for(runtime.spec):
                    self._reclaim(model_id)
                if not self._free_for(runtime.spec):
                    raise MemoryError(f"模型 {model_id} 预计显存不足，拒绝加载")
                runtime.loader = create_loader(runtime.spec)
                runtime.loader.load(
                    runtime.spec.path,
                    quantization=runtime.spec.quantization,
                    dtype=runtime.spec.dtype,
                    max_model_len=runtime.spec.max_model_len,
                    tensor_parallel_size=runtime.spec.tensor_parallel_size,
                    trust_remote_code=runtime.spec.trust_remote_code,
                )
                load_lora(runtime.loader, runtime.spec.lora)
                runtime.state, runtime.error = "RUNNING", None
                runtime.last_used_at = time.time()
                return runtime
            except Exception as exc:
                if runtime.loader:
                    runtime.loader.unload()
                self.runtime.pop(model_id, None)
                raise RuntimeError(f"模型 {model_id} 加载失败: {exc}") from exc

    def generate(self, model_id: str, prompt: str, **kwargs: Any) -> GenerationResult:
        with self._lock:
            runtime = self.ensure_loaded(model_id)
            if runtime.loader is None:
                raise RuntimeError(f"模型 {model_id} 未加载")
            runtime.active = True
            try:
                result = runtime.loader.generate(prompt, **kwargs)
                runtime.last_used_at = time.time()
                return result
            finally:
                runtime.active = False

    def reconfigure(self, model_id: str, changes: dict[str, Any]) -> RuntimeModel:
        """应用会影响模型创建的参数；如模型正在运行则先安全卸载再加载。"""
        load_keys = {"engine", "dtype", "quantization", "max_model_len", "tensor_parallel_size",
                     "trust_remote_code"}
        if not changes or not load_keys.intersection(changes):
            return self._get_runtime(model_id)
        with self._lock:
            runtime = self._get_runtime(model_id)
            if runtime.active:
                raise RuntimeError("模型当前正在生成，不能修改加载参数")
            self._unload_runtime(model_id)
            spec = self.specs[model_id]
            self.specs[model_id] = replace(spec, **{key: changes[key] for key in load_keys if key in changes})
            try:
                return self.ensure_loaded(model_id)
            except Exception:
                self.specs[model_id] = spec
                raise

    def sleep(self, model_id: str) -> bool:
        with self._lock:
            runtime = self._get_runtime(model_id)
            if runtime.active or not runtime.loader:
                return False
            if runtime.loader.sleep_to_ram():
                runtime.state, runtime.sleep_location = "SLEEPING_RAM", "ram"
                return True
            return self.unload(model_id)

    def _unload_runtime(self, model_id: str) -> None:
        runtime = self.runtime.get(model_id)
        if not runtime or runtime.active:
            return
        if runtime.loader:
            if self.settings.get("cache", {}).get("state_snapshot_enabled", True):
                save_snapshot(self.asset_root, runtime.spec, state=runtime.state,
                              last_used_at=runtime.last_used_at)
            runtime.loader.unload()
        self.runtime.pop(model_id, None)

    def unload(self, model_id: str) -> bool:
        with self._lock:
            if model_id not in self.specs:
                return False
            runtime = self.runtime.get(model_id)
            if not runtime or runtime.active:
                return False
            self._unload_runtime(model_id)
            return True

    def reap_idle(self) -> list[str]:
        sleep_cfg = self.settings.get("sleep", {})
        if not sleep_cfg.get("enabled", True):
            return []
        timeout = int(sleep_cfg.get("idle_timeout_sec", 600))
        now = time.time()
        changed: list[str] = []
        for model_id, runtime in list(self.runtime.items()):
            if not runtime.active and now - runtime.last_used_at >= timeout and self.sleep(model_id):
                changed.append(model_id)
        return changed
