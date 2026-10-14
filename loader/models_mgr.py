from __future__ import annotations

import atexit,gc
import copy
import time
from pathlib import Path
from typing import Any

from ..utils.hardware import check_gpu_memory, detect_gpu
from ..utils.common import info as log_info, readFile, writeFile, info, error
from .model_spec import load_model_specs, modelSpec


class modelsMgr:
    """模型注册表：读配置、按 id 取模型、跨模型的显存准入与 LRU 回收。单个模型的生命周期由 modelSpec 自己负责。"""

    def __init__(self, config_path: str = "assets/models.json", settings_path: str = "assets/modelSetting.json",) -> None:
        self.config_path = Path(config_path)
        self.settings_path = Path(settings_path)

        self._config: dict[str, Any] = readFile(str(self.config_path)) or {}
        self._model_settings: dict[str, Any] = readFile(str(self.settings_path)) or {}

        defaults = self._config.get("defaults", {})
        self.settings: dict[str, Any] = copy.deepcopy(defaults) if isinstance(defaults, dict) else {}
        self.specs: dict[str, modelSpec] = load_model_specs(self._config)
        for spec in self.specs.values():
            spec.settings, spec.admit, spec.persist = self._model_settings, self._admit, self._reconfigure

        self.ram_reserve_mb = 8192
        self.gpu_reserve_mb = 1024
        self.sleep_time = int(self.settings.get("sleepTime", 60))

        # 注册退出清理钩子：主进程退出时强制清理所有后端 Server 子进程
        atexit.register(self.unload_all)

        # 首次启动即补齐缺失的 launch 等字段并写回
        self._reconfigure()

    def getModel(self, model_id: str) -> modelSpec | None:
        """按 id 取模型（任意状态，无副作用）；不在 models.json 中返回 None。"""
        return self.specs.get(model_id)

    def autolaunch(self) -> None:
        print("~~~~~~~autolaunch~~~~~~~~~",self.specs)
        for spec in (s for s in self.specs.values() if s.launch):
            print("启动加载模型", spec.model_id)
            try:
                spec.ensure()
            except (RuntimeError, MemoryError) as exc:
                error("启动加载失败", spec.model_id, exc)

    def _reconfigure(self) -> None:
        template_load = self._model_settings.get("template", {}).get("load", {})
        models = self._config.setdefault("models", {})
        modified = False

        for model_id, model_cfg in models.items():
            if not isinstance(model_cfg, dict):
                continue

            if "launch" not in model_cfg:
                model_cfg["launch"] = False
                modified = True

            if "data_compression" not in model_cfg:
                model_cfg["data_compression"] = "default"
                if model_id in self.specs:
                    self.specs[model_id].data_compression = "default"
                modified = True

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
            log_info("已自动补齐 models.json 中的 launch/load/data_compression 并完成持久化")

    def list_models(self) -> list[dict[str, Any]]:
        return [spec.public_dict() for spec in self.specs.values()]

    def status(self) -> dict[str, Any]:
        return {"models": self.list_models(), "queue_length": 0}

    # ------------------------------------------------------------------ 跨模型的显存准入
    def _free_for(self, spec: modelSpec) -> bool:
        required = spec.required_vram_mb()
        if required is None or not detect_gpu():
            return True
        return check_gpu_memory(required, self.gpu_reserve_mb).allowed

    def _admit(self, spec: modelSpec) -> None:
        """spec 启动/唤醒前调用：显存不够就按 LRU 把别的空闲模型休眠，仍不够抛 MemoryError。"""
        if self._free_for(spec):
            return
        self._reclaim(spec)
        # clean_gpu_cache()
        gc.collect()
        if self._free_for(spec):
            return
        raise MemoryError(f"模型 {spec.model_id} 显存不足")

    def _reclaim(self, target: modelSpec) -> None:
        # 只挑空闲且在运行的模型；拿不到锁（正在加载/切换）的直接跳过，避免互相等待
        candidates = [s for s in self.specs.values() if s is not target and s.isRunning and not s.active]
        for spec in sorted(candidates, key=lambda s: s.last_used_at):
            spec.sleep(blocking=False)
            if self._free_for(target):
                return

    # ------------------------------------------------------------------ 回收/清理
    def unload_all(self) -> None:
        """强制关停所有模型的 Server 子进程。不拿 spec 锁：正在加载的线程持锁，等锁会让退出死等、子进程成孤儿占着显存。"""
        for spec in self.specs.values():
            loader = spec.loader
            if loader is None:
                continue
            try:
                loader.unload()
            except Exception as exc:
                error(f"卸载模型 {spec.model_id} 异常: {exc}")

    def reap_idle(self) -> list[str]:
        timeout = self.sleep_time
        if timeout <= 0:
            return []
        now = time.time()
        idle = [s for s in self.specs.values() if s.isRunning and not s.active and now - s.last_used_at >= timeout]
        return [s.model_id for s in idle if s.sleep(blocking=False)]