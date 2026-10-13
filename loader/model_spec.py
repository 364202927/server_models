from __future__ import annotations

import copy
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..utils.common import aContainB, info, require, vm2tokens
from ..utils.hardware import query_gpu_used_mb
from .llmFramework.baseInference import GenerationResult, baseInference


# def clean_gpu_cache() -> None:
#     """回收主进程内的残留显存（推理引擎在子进程里，这里只处理本进程的 torch 缓存）。"""
#     gc.collect()
#     try:
#         import torch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.ipc_collect()
#     except ImportError:
#         pass


@dataclass(eq=False)
class modelSpec:
    """单个模型：配置 + 状态机（加载/休眠/唤醒/卸载）+ 生成调用。跨模型的显存腾挪交给 modelsMgr 注入的 admit。"""

    # ---- 静态配置（models.json）
    model_id: str
    path: str
    source_path: str | None = field(default=None, repr=False, compare=False)
    estimated_vram_mb: int | None = None
    engine: str | None = None
    launch: bool = False
    data_compression: str = "default"
    load: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    # ---- 运行状态
    loader: baseInference | None = field(default=None, repr=False)
    state: str = "UNLOADED"          # UNLOADED / LOADING / RUNNING / SLEEPING_RAM
    last_used_at: float = field(default_factory=time.time)
    active_requests: int = 0
    error: str | None = None
    sleep_location: str | None = None
    needs_remeasure: bool = False

    # ---- 由 modelsMgr 注入，spec 不依赖 mgr
    settings: dict[str, Any] = field(default_factory=dict, repr=False)               # modelSetting.json 全量
    admit: Callable[[modelSpec], None] = field(default=lambda spec: None, repr=False)  # 腾显存，不足抛 MemoryError
    persist: Callable[[], None] = field(default=lambda: None, repr=False)              # 写回 models.json
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------ 状态查询
    @property
    def active(self) -> bool:
        return self.active_requests > 0

    @property
    def isRunning(self) -> bool:
        return self.state == "RUNNING"

    @property
    def isAlive(self) -> bool:
        """已加载（含休眠/加载中），请求到达时可直接唤醒或等待，无需重建。"""
        return self.state != "UNLOADED"

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
        if aContainB(path.name.lower(), ("fp8", "awq", "gptq")):
            return "vllm"
        return None

    # ------------------------------------------------------------------ 配置合并
    def _merged(self, section: str) -> dict[str, Any]:
        """模板 < 引擎默认 < 本模型，三层合并。"""
        engine = self.detect_engine() or "llama"
        merged = copy.deepcopy(self.settings.get("template", {}).get(section, {}))
        merged.update(self.settings.get(engine, {}).get(section, {}))
        merged.update(self.load if section == "load" else self.generation)
        return merged

    def full_load(self) -> dict[str, Any]:
        return self._merged("load")

    def full_generation(self) -> dict[str, Any]:
        return self._merged("generation")

    # ------------------------------------------------------------------ 显存估算
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

    # ------------------------------------------------------------------ 状态机
    def ensure(self, load_override: dict[str, Any] | None = None) -> modelSpec:
        """保证可用：运行中直接返回；休眠则唤醒；未加载则冷启动。同一模型的并发调用在锁上串行。"""
        with self._lock:
            if self.state == "RUNNING":
                self.last_used_at = time.time()
                return self
            if self.state == "SLEEPING_RAM" and self._wake():
                return self
            self._start(load_override or {})
            return self

    def _wake(self) -> bool:
        """唤醒休眠的引擎；失败则物理卸载，交由调用方冷启动。"""
        try:
            self.admit(self)
            self.loader.wake()
        except (RuntimeError, MemoryError, OSError):
            self._drop()
            return False
        self.state, self.sleep_location, self.last_used_at = "RUNNING", None, time.time()
        return True

    def _start(self, load_override: dict[str, Any]) -> None:
        self.state = "LOADING"
        try:
            self.admit(self)
            engine = self.detect_engine()
            if not engine:
                raise ValueError(f"未指定引擎: {self.model_id}")
            self.loader = require(f"{__package__}.llmFramework.{engine}")()

            full_load = self.full_load()
            full_load.update({k: v for k, v in load_override.items() if k in full_load})
            full_load["useVram"] = vm2tokens(
                self.engine, self.path, float(full_load["useVram"]), full_load["dtype"], int(full_load["tensor_parallel"])
            )

            used_before = query_gpu_used_mb()
            self.loader.load(self.path, full_load)
            self.update_estimated_vram(used_before, query_gpu_used_mb(), force=self.needs_remeasure)
            self.needs_remeasure = False
            self.persist()
        except Exception as exc:  # 清理后原样上抛，保证失败不残留子进程
            self._drop()
            self.error = str(exc)
            raise RuntimeError(f"模型 {self.model_id} 加载失败: {exc}") from exc
        self.state, self.error, self.last_used_at = "RUNNING", None, time.time()

    def _drop(self) -> None:
        """终止引擎子进程并回到未加载态（调用方已持锁）。"""
        loader, self.loader = self.loader, None
        self.state, self.sleep_location = "UNLOADED", None
        if loader is not None:
            loader.unload()
        gc.collect()
        # clean_gpu_cache()

    def sleep(self, blocking: bool = True) -> bool:
        """休眠到内存；引擎不支持休眠则直接卸载以交还显存。忙碌/未运行/拿不到锁返回 False。"""
        if not self._lock.acquire(blocking=blocking):
            return False
        try:
            if self.active or self.loader is None or self.state != "RUNNING":
                return False
            if self.loader.sleep_to_ram():
                self.state, self.sleep_location = "SLEEPING_RAM", "ram"
                gc.collect()
                # clean_gpu_cache()
            else:
                self._drop()
            return True
        finally:
            self._lock.release()

    def unload(self, force: bool = False) -> bool:
        with self._lock:
            if self.state == "UNLOADED" or (self.active and not force):
                return False
            self._drop()
            return True

    # ------------------------------------------------------------------ 生成
    def generate(self, messages: list[dict[str, Any]], overrides: dict[str, Any] | None = None,
                 prune_on_miss: bool = False) -> GenerationResult:
        """ensure + 参数合并 + 引擎生成。ensure 与计数同锁，避免被并发的 sleep 抢在中间。"""
        overrides = overrides or {}
        with self._lock:
            self.ensure()
            self.active_requests += 1
        try:
            cfg = self.full_generation()
            cfg.update({k: v for k, v in overrides.items() if k in cfg and v is not None})
            cfg.update({k: overrides[k] for k in ("tools", "tool_choice", "id_slot") if k in overrides})
            result = self.loader.generate(messages, cfg, prune_on_miss)
            self.last_used_at = time.time()
            return result
        finally:
            with self._lock:
                self.active_requests = max(0, self.active_requests - 1)

    # ------------------------------------------------------------------ 展示/序列化
    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "state": self.state,
            "path": self.path,
            "estimated_vram_mb": self.estimated_vram_mb,
            "data_compression": self.data_compression,
            "last_used_at": datetime.fromtimestamp(self.last_used_at, tz=timezone.utc).isoformat(),
            "gpu_memory_mb": self.estimated_vram_mb if self.state == "RUNNING" else 0,
            "sleep_location": self.sleep_location,
            "error": self.error,
        }

    def to_config_node(self) -> dict[str, Any]:
        node: dict[str, Any] = {
            "path": self.source_path or self.path,
            "launch": self.launch,
            "engine": self.engine,
            "data_compression": self.data_compression,
            "load": self.load,
            "generation": self.generation,
        }
        if self.estimated_vram_mb is not None:
            node["estimated_vram_mb"] = self.estimated_vram_mb
        return node


def load_model_specs(config: dict[str, Any]) -> dict[str, modelSpec]:
    result: dict[str, modelSpec] = {}
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models.json 的 models 必须是对象")

    for model_id, raw in models.items():
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        result[str(model_id)] = modelSpec(
            model_id=str(model_id),
            path=normalize_model_path(str(raw["path"])),
            source_path=str(raw["path"]),
            estimated_vram_mb=raw.get("estimated_vram_mb"),
            engine=str(raw.get("engine", "")).strip().lower() or None,
            launch=raw.get("launch") is True,
            data_compression=str(raw.get("data_compression") or "default"),
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