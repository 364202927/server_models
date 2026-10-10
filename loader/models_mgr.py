from __future__ import annotations

import copy
import gc
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.hardware import check_gpu_memory, check_ram, detect_gpu, query_gpu_used_mb, query_gpu_free_mb
from ..utils.common import info as log_info, readFile, writeFile, error, require, info, vm2tokens
from .llmFramework.baseInference import GenerationResult, baseInference
from .model_spec import ModelSpec, load_model_specs

@dataclass
class RuntimeModel:
    spec: ModelSpec
    loader: baseInference | None = None
    state: str = "UNLOADED"
    last_used_at: float = field(default_factory=time.time)
    active: bool = False
    error: str | None = None
    sleep_location: str | None = None
    needs_remeasure: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.model_id,
            "state": self.state,
            "path": self.spec.path,
            "estimated_vram_mb": self.spec.estimated_vram_mb,
            "think_strategy": self.spec.think_strategy,
            "last_used_at": datetime.fromtimestamp(self.last_used_at, tz=timezone.utc).isoformat(),
            "gpu_memory_mb": self.spec.estimated_vram_mb if self.state == "RUNNING" else 0,
            "sleep_location": self.sleep_location,
            "error": self.error,
        }


class ModelsMgr:
    """模型实例管理器：双 JSON 闭包参数驱动。"""

    def __init__(
        self,
        config_path: str = "assets/models.json",
        settings_path: str = "assets/modelSetting.json",
    ) -> None:
        self.config_path = Path(config_path)
        self.settings_path = Path(settings_path)

        self._config: dict[str, Any] = readFile(str(self.config_path)) or {}
        self._model_settings: dict[str, Any] = readFile(str(self.settings_path)) or {}

        defaults = self._config.get("defaults", {})
        self.settings: dict[str, Any] = copy.deepcopy(defaults) if isinstance(defaults, dict) else {}
        self.specs: dict[str, ModelSpec] = load_model_specs(self._config)
        self.runtime: dict[str, RuntimeModel] = {}
        self._meta_lock = threading.Lock()

        self.ram_reserve_mb = 8192
        self.gpu_reserve_mb = 1024
        self.sleep_time = int(self.settings.get("sleepTime", 60))
    #load和tink策略写入配置
    def _reconfigure(self) -> None:
        template_load = self._model_settings.get("template", {}).get("load", {})
        models = self._config.setdefault("models", {})
        modified = False

        for model_id, model_cfg in models.items():
            if not isinstance(model_cfg, dict):
                continue

            # 1. 检查并补齐 think_strategy
            if "think_strategy" not in model_cfg:
                model_cfg["think_strategy"] = "default"
                if model_id in self.specs:
                    self.specs[model_id].think_strategy = "default"
                modified = True

            # 2. 检查并补齐 model.load
            spec = self.specs.get(model_id)
            if spec and not spec.load:
                spec.load = copy.deepcopy(template_load)
                model_cfg["load"] = copy.deepcopy(template_load)
                modified = True
            elif not model_cfg.get("load"):
                model_cfg["load"] = copy.deepcopy(template_load)
                if spec:
                    spec.load = copy.deepcopy(template_load)
                modified = True

        if modified:
            writeFile(self._config, str(self.config_path))
            log_info("已自动补齐 models.json 中的 load 与 think_strategy 并完成持久化")

    def get_full_load_config(self, model_id: str) -> dict[str, Any]:
        """完整加载参数闭包 = template.load + engine.load + model.load"""
        spec = self.specs[model_id]
        engine = spec.detect_engine() or "llama"

        full_load = copy.deepcopy(self._model_settings.get("template", {}).get("load", {}))
        full_load.update(self._model_settings.get(engine, {}).get("load", {}))
        full_load.update(spec.load)
        return full_load

    def get_full_generation_config(self, model_id: str) -> dict[str, Any]:
        """完整生成参数闭包 = template.generation + engine.generation + model.generation"""
        spec = self.specs.get(model_id)
        engine = (spec.detect_engine() if spec else None) or "llama"

        full_gen = copy.deepcopy(self._model_settings.get("template", {}).get("generation", {}))
        full_gen.update(self._model_settings.get(engine, {}).get("generation", {}))
        if spec:
            full_gen.update(spec.generation)
        return full_gen

    def list_models(self) -> list[dict[str, Any]]:
        with self._meta_lock:
            return [(self.runtime.get(model_id) or RuntimeModel(spec)).public_dict()
                    for model_id, spec in self.specs.items()]

    def status(self) -> dict[str, Any]:
        return {"models": self.list_models(), "queue_length": 0}

    def _get_runtime(self, model_id: str) -> RuntimeModel:
        if model_id not in self.specs:
            raise KeyError(f"models.json 中不存在模型: {model_id}")
        with self._meta_lock:
            return self.runtime.setdefault(model_id, RuntimeModel(self.specs[model_id]))

    def _free_for(self, spec: ModelSpec) -> bool:
        required = spec.required_vram_mb()
        if required is None or not detect_gpu():
            return True
        return check_gpu_memory(required, self.gpu_reserve_mb).allowed

    def _clean_gpu_cache(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    def _admit(self, spec: ModelSpec) -> None:
        if self._free_for(spec):
            return
        self._reclaim_to_ram(exclude=spec.model_id)
        self._clean_gpu_cache()
        if self._free_for(spec):
            return
        raise MemoryError(f"模型 {spec.model_id} 显存不足")

    def _reclaim_to_ram(self, exclude: str) -> None:
        with self._meta_lock:
            candidates = [
                item for model_id, item in self.runtime.items()
                if model_id != exclude and item.loader and item.state == "RUNNING"
            ]
        candidates.sort(key=lambda item: item.last_used_at)
        for item in candidates:
            self._release(item)
            if self._free_for(self.specs[exclude]):
                return

    def _release(self, runtime: RuntimeModel) -> None:
        model_id = runtime.spec.model_id
        if runtime.loader is not None and runtime.loader.sleep_to_ram():
            with self._meta_lock:
                runtime.state, runtime.sleep_location = "SLEEPING_RAM", "ram"
            return
        self._unload_runtime(model_id)

    def ensure_loaded(self, model_id: str, load_override: dict[str, Any] | None = None) -> RuntimeModel:
        runtime = self._get_runtime(model_id)
        if runtime.loader and runtime.state in {"RUNNING", "SLEEPING_RAM"}:
            return self._resume(runtime)
        return self._load(runtime, load_override or {})

    def _resume(self, runtime: RuntimeModel) -> RuntimeModel:
        if runtime.state == "RUNNING":
            runtime.last_used_at = time.time()
            return runtime
        try:
            self._admit(runtime.spec)
            runtime.loader.wake()
        except Exception:
            self._unload_runtime(runtime.spec.model_id)
            return self._load(runtime, {})
        with self._meta_lock:
            runtime.state, runtime.sleep_location = "RUNNING", None
            runtime.last_used_at = time.time()
        return runtime

    def _load(self, runtime: RuntimeModel, load_override: dict[str, Any]) -> RuntimeModel:
        spec = runtime.spec
        model_id = spec.model_id
        with self._meta_lock:
            runtime.state = "LOADING"
        try:
            self._admit(spec)
            engine = spec.detect_engine()
            if not engine:
                raise ValueError(f"未指定引擎: {model_id}")
            cls = require(f"{__package__}.llmFramework.{engine}")
            runtime.loader = cls()
            # 完整加载参数闭包 + 仅接收存在于完整参数的覆盖值
            full_load = self.get_full_load_config(model_id)
            for k, v in load_override.items():
                if k in full_load:
                    full_load[k] = v
            # 显存->token
            context_val = vm2tokens(spec.engine, spec.path, float(full_load['context']), full_load['dtype'], int(full_load['tensor_parallel']))
            full_load['context'] = context_val
            used_before = query_gpu_used_mb()
            # 加载模型
            runtime.loader.load(spec.path, full_load)
            spec.update_estimated_vram(used_before, query_gpu_used_mb(), force=runtime.needs_remeasure)
            runtime.needs_remeasure = False
            # 检查并自动补齐配置后持久化
            self._reconfigure()
            with self._meta_lock:
                runtime.state, runtime.error = "RUNNING", None
                runtime.last_used_at = time.time()
            return runtime
        except Exception as exc:
            if runtime.loader:
                try:
                    runtime.loader.unload()
                except Exception:
                    pass
            with self._meta_lock:
                self.runtime.pop(model_id, None)
            self._clean_gpu_cache()
            raise RuntimeError(f"模型 {model_id} 加载失败: {exc}") from exc

    def generate(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any] | None = None,
    ) -> GenerationResult:
        runtime = self.ensure_loaded(model_id)
        if runtime.loader is None:
            raise RuntimeError(f"模型 {model_id} 未就绪")

        runtime.active = True
        try:
            result = runtime.loader.generate(messages, gen_cfg)
            runtime.last_used_at = time.time()
            return result
        finally:
            runtime.active = False

    def sleep(self, model_id: str) -> bool:
        runtime = self._get_runtime(model_id)
        if runtime.active or not runtime.loader or runtime.state != "RUNNING":
            return False
        self._release(runtime)
        self._clean_gpu_cache()
        return True

    def _unload_runtime(self, model_id: str) -> None:
        with self._meta_lock:
            runtime = self.runtime.pop(model_id, None)
        if not runtime or runtime.active:
            return
        if runtime.loader:
            runtime.state, runtime.sleep_location = "UNLOADED", None
            runtime.loader.unload()
        self._clean_gpu_cache()

    def unload(self, model_id: str) -> bool:
        if model_id not in self.specs:
            return False
        runtime = self.runtime.get(model_id)
        if not runtime or runtime.active:
            return False
        self._unload_runtime(model_id)
        return True
    
    def reap_idle(self) -> list[str]:
        timeout = self.sleep_time
        if timeout <= 0:
            return []
        now = time.time()
        changed: list[str] = []
        with self._meta_lock:
            candidates = list(self.runtime.items())
        for model_id, runtime in candidates:
            if runtime.state != "RUNNING" or runtime.active:
                continue
            if now - runtime.last_used_at >= timeout and self.sleep(model_id):
                changed.append(model_id)
        return changed