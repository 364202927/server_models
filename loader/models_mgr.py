from __future__ import annotations

import copy, gc, threading, time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.hardware import check_gpu_memory, check_ram, detect_gpu, query_gpu_used_mb
from ..utils.common import info as log_info, readFile, writeFile, error, require
from .llmFramework.baseInference import GenerationResult, baseInference
from .model_spec import LOAD_KEYS, ModelSpec, load_model_specs

GENERATION_DEFAULTS: dict[str, Any] = {
    "system_prompt": "",
    "presence_penalty": 0,
    "frequency_penalty": 0,
}

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
            "last_used_at": datetime.fromtimestamp(self.last_used_at, tz=timezone.utc).isoformat(),
            "gpu_memory_mb": self.spec.estimated_vram_mb if self.state == "RUNNING" else 0,
            "sleep_location": self.sleep_location,
            "error": self.error,
        }


class ModelsMgr:
    """模型实例管理器（针对单卡串行任务优化）。"""

    def __init__(self, config_path: str = "assets/models.json") -> None:
        self.config_path = Path(config_path)
        config = readFile(str(self.config_path)) or {}
        if not isinstance(config, dict):
            raise ValueError("models.json 根节点必须是对象")
        self._config: dict[str, Any] = config
        defaults = config.get("defaults", {})
        self.settings: dict[str, Any] = copy.deepcopy(defaults) if isinstance(defaults, dict) else {}
        self.specs: dict[str, ModelSpec] = load_model_specs(config)
        self.runtime: dict[str, RuntimeModel] = {}
        self._meta_lock = threading.Lock()

        self.ram_reserve_mb = 8192
        self.gpu_reserve_mb = 1024

        server = self.settings.get("server", {})
        self.sleep_time = int(server.get("sleepTime", 600)) if isinstance(server, dict) else 600

    def generation_params(self, model_id: str) -> dict[str, Any]:
        params = dict(self.settings.get("generation", {}))
        for name, value in GENERATION_DEFAULTS.items():
            params.setdefault(name, value)
        spec = self.specs.get(model_id)
        if spec:
            params.update(spec.generation)
        return params

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
        raise MemoryError(
            f"模型 {spec.model_id} 需要约 {spec.required_vram_mb()} MB 显存，"
            f"移入 RAM 并释放显存后仍不足"
        )

    def _reclaim_to_ram(self, exclude: str) -> None:
        with self._meta_lock:
            candidates = [
                item for model_id, item in self.runtime.items()
                if model_id != exclude and item.loader and item.state == "RUNNING"
            ]
        candidates.sort(key=lambda item: item.last_used_at)
        log_info("显存不足，开始将闲置模型腾退至 RAM", "exclude=", exclude,
                 "candidates=", [item.spec.model_id for item in candidates])
        for item in candidates:
            self._release(item)
            if self._free_for(self.specs[exclude]):
                return

    def _reclaim_ram(self, exclude: str, required_mb: int) -> bool:
        with self._meta_lock:
            candidates = [
                item for model_id, item in self.runtime.items()
                if model_id != exclude and item.loader and item.state == "SLEEPING_RAM"
            ]
        candidates.sort(key=lambda item: item.last_used_at)
        log_info("RAM 不足，开始按 LRU 卸载休眠模型", "exclude=", exclude,
                 "candidates=", [item.spec.model_id for item in candidates])
        for item in candidates:
            self._unload_runtime(item.spec.model_id)
            if check_ram(required_mb, self.ram_reserve_mb).allowed:
                return True
        return check_ram(required_mb, self.ram_reserve_mb).allowed

    def _release(self, runtime: RuntimeModel) -> None:
        model_id = runtime.spec.model_id
        required_mb = int(runtime.spec.required_vram_mb() or 0)
        ram_ok = check_ram(required_mb, self.ram_reserve_mb).allowed
        if not ram_ok:
            ram_ok = self._reclaim_ram(model_id, required_mb)

        if ram_ok and runtime.loader is not None and runtime.loader.sleep_to_ram():
            with self._meta_lock:
                runtime.state, runtime.sleep_location = "SLEEPING_RAM", "ram"
            log_info("模型已成功休眠至 RAM:", model_id)
            return

        log_info("RAM 不足或不支持休眠，彻底卸载:", model_id)
        self._unload_runtime(model_id)

    def ensure_loaded(self, model_id: str) -> RuntimeModel:
        runtime = self._get_runtime(model_id)
        if runtime.loader and runtime.state in {"RUNNING", "SLEEPING_RAM"}:
            return self._resume(runtime)
        return self._load(runtime)

    def _resume(self, runtime: RuntimeModel) -> RuntimeModel:
        model_id = runtime.spec.model_id
        if runtime.state == "RUNNING":
            runtime.last_used_at = time.time()
            return runtime
        try:
            self._admit(runtime.spec)
            log_info("从 RAM 唤醒模型:", model_id)
            runtime.loader.wake()
        except Exception as exc:
            log_info("模型唤醒失败，降级为卸载冷重载:", model_id, exc)
            self._unload_runtime(model_id)
            return self._load(runtime)

        with self._meta_lock:
            runtime.state, runtime.sleep_location = "RUNNING", None
            runtime.last_used_at = time.time()
        return runtime

    def _load(self, runtime: RuntimeModel) -> RuntimeModel:
        spec = runtime.spec
        model_id = spec.model_id
        with self._meta_lock:
            runtime.state = "LOADING"
        try:
            if spec.draft and spec.mtp:
                raise ValueError("draft 与 mtp 不能同时启用")
            log_info("开始加载模型", model_id, "context_length=", spec.load.context_length)
            self._admit(spec)

            engine = spec.detect_engine()
            if not engine:
                error("无法确定推理框架: model=", spec.model_id)
                with self._meta_lock:
                    runtime.state, runtime.error = "UNLOADED", "无法确定推理框架"
                return runtime

            cls = require(f"{__package__}.llmFramework.{engine}")
            runtime.loader = cls() if cls else None
            if runtime.loader is None:
                with self._meta_lock:
                    runtime.state, runtime.error = "UNLOADED", f"未能实例化引擎 {engine}"
                return runtime

            used_before = query_gpu_used_mb()
            engine_name = type(runtime.loader).__name__.lower()
            load_kwargs = spec.resolve_loader_kwargs(engine_name)

            runtime.loader.load(spec.path,config=load_kwargs, draft=spec.draft, lora=spec.lora)

            if spec.update_estimated_vram(used_before, query_gpu_used_mb(), force=runtime.needs_remeasure):
                runtime.needs_remeasure = False
                log_info("模型显存实测:", spec.model_id, f"{spec.estimated_vram_mb} MB")

            spec.apply_effective_load(runtime.loader.effective_load, runtime.loader.model_info)
            self._persist_spec(spec)

            with self._meta_lock:
                runtime.state, runtime.error = "RUNNING", None
                runtime.last_used_at = time.time()
            log_info("模型加载完成", model_id, "vram_mb=", spec.estimated_vram_mb)
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

    def _persist_spec(self, spec: ModelSpec) -> None:
        models = self._config.setdefault("models", {})
        spec.generation = self.generation_params(spec.model_id)
        spec.generation_present = True
        models[spec.model_id] = spec.to_config_node()

        tmp_path = self.config_path.with_name(f".{self.config_path.stem}.tmp.json")
        writeFile(self._config, str(tmp_path))
        tmp_path.replace(self.config_path)
        log_info("模型配置已补全并写入磁盘", spec.model_id)

    def generate(self, model_id: str, prompt: str, *, sampling: dict[str, Any] | None = None, system_prompt: str = "", messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> GenerationResult:
        runtime = self.ensure_loaded(model_id)
        if runtime.loader is None:
            raise RuntimeError(f"模型 {model_id} 未就绪")

        runtime.active = True
        try:
            result = runtime.loader.generate(
                prompt,
                sampling=sampling,
                system_prompt=system_prompt,
                messages=messages,
                **kwargs,
            )
            runtime.last_used_at = time.time()
            return result
        finally:
            runtime.active = False

    def reconfigure(self, model_id: str, changes: dict[str, Any]) -> RuntimeModel:
        runtime = self._get_runtime(model_id)
        spec = self.specs[model_id]
        wanted = {key: value for key, value in changes.items()
                  if key in LOAD_KEYS and getattr(spec.load, key) != value}
        if not wanted:
            return runtime
        if runtime.active:
            raise RuntimeError("模型当前正在生成，不能修改加载参数")
        log_info("重新配置模型:", model_id, wanted)
        self._unload_runtime(model_id)
        changed_spec = replace(spec, load=replace(spec.load, **wanted))
        changed_spec.load_fields = set(spec.load_fields) | set(wanted)
        changed_spec.load_present = True
        self.specs[model_id] = changed_spec
        self._get_runtime(model_id).needs_remeasure = True
        try:
            return self.ensure_loaded(model_id)
        except Exception:
            self.specs[model_id] = spec
            raise

    def update_generation(self, model_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        spec = self.specs.get(model_id)
        if spec is None:
            raise KeyError(f"models.json 中不存在模型: {model_id}")
        merged = self.generation_params(model_id)
        merged.update(changes)
        spec.generation = merged
        spec.generation_present = True
        models = self._config.setdefault("models", {})
        node = models.setdefault(spec.model_id, {})
        node["generation"] = dict(merged)
        tmp_path = self.config_path.with_name(f".{self.config_path.stem}.tmp.json")
        writeFile(self._config, str(tmp_path))
        tmp_path.replace(self.config_path)
        log_info("生成参数已持久化:", model_id, changes)
        return merged

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