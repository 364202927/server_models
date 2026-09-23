"""多模型生命周期管理器：显存准入、RAM 休眠、卸载和空闲回收。"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..hardware import check_gpu_memory, check_ram, detect_gpu, query_gpu_used_mb
from ..utils.common import info as log_info, readFile, writeFile
from .llmFramework.baseInference import GenerationResult, baseInference
from .expand.lora import load_lora
from .model_spec import CACHE_DEFAULTS, LOAD_KEYS, ModelSpec, load_model_specs
from .tool_format import create_loader, load_snapshot, save_snapshot


@dataclass
class RuntimeModel:
    spec: ModelSpec
    loader: baseInference | None = None
    state: str = "UNLOADED"
    last_used_at: float = field(default_factory=time.time)
    active: bool = False
    error: str | None = None
    sleep_location: str | None = None
    # reconfigure 改过 load 参数后置位：原显存估算已失效，下次加载重测并覆盖。
    needs_remeasure: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.model_id,
            "state": self.state,
            "path": self.spec.path,
            "estimated_vram_mb": self.spec.estimated_vram_mb,
            "last_used_at": datetime.fromtimestamp(self.last_used_at, tz=timezone.utc).isoformat(),
            # 用实测估算值而不是现场查询：状态接口不该为了一个数字去 fork nvidia-smi。
            "gpu_memory_mb": self.spec.estimated_vram_mb if self.state == "RUNNING" else 0,
            "sleep_location": self.sleep_location,
            "error": self.error,
        }


class ModelsMgr:
    """串行场景下的模型注册表和运行时实例管理器。"""

    def __init__(self, config_path: str = "assets/models.json") -> None:
        self.config_path = Path(config_path)
        config = readFile(str(self.config_path)) or {}
        if not isinstance(config, dict):
            raise ValueError("models.json 根节点必须是对象")
        self._config: dict[str, Any] = config
        defaults = config.get("defaults", {})
        # 深拷贝：settings 与 _config["defaults"] 必须解耦，否则运行期对 settings 的
        # 任何调整都会跟着持久化写回用户的 models.json。
        self.settings: dict[str, Any] = copy.deepcopy(defaults) if isinstance(defaults, dict) else {}
        self.specs: dict[str, ModelSpec] = load_model_specs(config)
        self.runtime: dict[str, RuntimeModel] = {}
        self.asset_root = str(self.config_path.parent)
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- 配置读取

    @property
    def sleep_settings(self) -> dict[str, Any]:
        value = self.settings.get("sleep", {})
        return value if isinstance(value, dict) else {}

    @property
    def cache_settings(self) -> dict[str, Any]:
        value = self.settings.get("cache", {})
        return {**CACHE_DEFAULTS, **value} if isinstance(value, dict) else dict(CACHE_DEFAULTS)

    def generation_params(self, model_id: str) -> dict[str, Any]:
        """解析 ``defaults.generation`` ← 模型 ``generation``。

        调用方（MsgHandler）再叠加请求里的 ``deploy``，构成三层覆盖。
        """
        params = dict(self.settings.get("generation", {}))
        spec = self.specs.get(model_id)
        if spec:
            params.update(spec.generation)
        return params

    # ---------------------------------------------------------------- 状态查询

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            return [(self.runtime.get(model_id) or RuntimeModel(spec)).public_dict()
                    for model_id, spec in self.specs.items()]

    def status(self) -> dict[str, Any]:
        return {"models": self.list_models(), "queue_length": 0}

    def _get_runtime(self, model_id: str) -> RuntimeModel:
        if model_id not in self.specs:
            raise KeyError(f"models.json 中不存在模型: {model_id}")
        return self.runtime.setdefault(model_id, RuntimeModel(self.specs[model_id]))

    # ---------------------------------------------------------------- 显存准入

    def _required_mb(self, spec: ModelSpec) -> int | None:
        """模型预计显存；实测值缺失时退回权重文件大小。"""
        if spec.estimated_vram_mb is not None:
            return int(spec.estimated_vram_mb)
        return spec.size_on_disk_mb()

    def _free_for(self, spec: ModelSpec) -> bool:
        required = self._required_mb(spec)
        if required is None:
            return True
        # 无 CUDA/NVIDIA 环境时允许后端自行决定 CPU 加载，不能因显存估算阻断调试运行。
        if not detect_gpu():
            return True
        reserve = int(self.sleep_settings.get("gpu_reserve_mb", 512))
        return check_gpu_memory(required, reserve).allowed

    def _admit(self, spec: ModelSpec) -> None:
        """显存准入：不够先按 LRU 释放空闲模型，仍不够则明确报错而不是等 OOM。"""
        if self._free_for(spec):
            return
        self._reclaim(spec.model_id)
        if self._free_for(spec):
            return
        raise MemoryError(
            f"模型 {spec.model_id} 需要约 {self._required_mb(spec)} MB 显存，"
            f"释放全部空闲模型后仍不足"
        )

    def _reclaim(self, exclude: str) -> None:
        """按最久未使用顺序释放空闲模型，直到目标装得下或候选耗尽。

        只按 LRU 排序：主模型通常显存最大也最常用，按显存降序会反复把它挤掉，
        导致每次切到专家模型都要重载主模型。
        """
        candidates = [item for model_id, item in self.runtime.items()
                      if model_id != exclude and item.loader
                      and not item.active and item.state == "RUNNING"]
        candidates.sort(key=lambda item: item.last_used_at)
        log_info("显存不足，开始回收", "exclude=", exclude,
                 "candidates=", [item.spec.model_id for item in candidates])
        for item in candidates:
            self._release(item)
            if self._free_for(self.specs[exclude]):
                return

    def _release(self, runtime: RuntimeModel) -> None:
        """单个模型的释放：优先休眠到 RAM，RAM 不足或引擎不支持时卸载。

        GGUF 的 ``sleep_to_ram`` 恒为 False（llama.cpp 的 CUDA 上下文无法迁移后恢复），
        因此走卸载分支；其权重文件仍在 OS 文件缓存中，重载接近 RAM 休眠的效果。
        """
        sleep_cfg = self.sleep_settings
        model_id = runtime.spec.model_id
        ram_ok = bool(sleep_cfg.get("ram_enabled", True)) and check_ram(
            int(self._required_mb(runtime.spec) or 0),
            int(sleep_cfg.get("ram_reserve_mb", 8192)),
        ).allowed
        if ram_ok and runtime.loader is not None and runtime.loader.sleep_to_ram():
            runtime.state, runtime.sleep_location = "SLEEPING_RAM", "ram"
            self._save_snapshot(runtime)
            log_info("模型休眠到RAM", model_id)
            return
        log_info("模型卸载", model_id, "ram_ok=", ram_ok)
        self._unload_runtime(model_id)

    # ---------------------------------------------------------------- 加载与唤醒

    def ensure_loaded(self, model_id: str) -> RuntimeModel:
        with self._lock:
            runtime = self._get_runtime(model_id)
            if runtime.loader and runtime.state in {"RUNNING", "SLEEPING_RAM"}:
                return self._resume(runtime)
            return self._load(runtime)

    def _resume(self, runtime: RuntimeModel) -> RuntimeModel:
        """命中已加载模型。

        SLEEPING_RAM 的唤醒会把权重搬回显存，必须和冷加载走同一套准入检查，
        否则其它模型占满显存时唤醒会直接 OOM。
        """
        model_id = runtime.spec.model_id
        if runtime.state == "RUNNING":
            runtime.last_used_at = time.time()
            return runtime
        try:
            self._admit(runtime.spec)
            runtime.loader.wake()
        except Exception as exc:
            # 唤醒失败保持 SLEEPING_RAM，权重仍在 RAM，下次请求可以重试。
            log_info("模型唤醒失败", model_id, type(exc).__name__, exc)
            raise RuntimeError(f"模型 {model_id} 唤醒失败: {exc}") from exc
        runtime.state, runtime.sleep_location = "RUNNING", None
        runtime.last_used_at = time.time()
        self._save_snapshot(runtime)
        log_info("模型已从RAM唤醒", model_id)
        return runtime

    def _load(self, runtime: RuntimeModel) -> RuntimeModel:
        spec = runtime.spec
        model_id = spec.model_id
        runtime.state = "LOADING"
        try:
            log_info("模型加载配置", model_id,
                     "path=", spec.path, "path_exists=", spec.path_obj.exists(),
                     "estimated_vram_mb=", spec.estimated_vram_mb,
                     "load=", spec.load.to_dict())
            self._admit(spec)
            # 加载前后的整卡差值才是本模型的占用；torch 的计数器是进程级累计。
            used_before = query_gpu_used_mb()
            runtime.loader = create_loader(spec)
            if runtime.loader is None:
                # create_loader 已经打印了具体原因(未配置 engine 且无法按后缀推断)；
                # 这里不再包成异常抛出，直接把模型标记为未加载并返回。保留 runtime
                # 记录(不 pop)以便 status()/list_models() 能看到 error 原因。
                runtime.state, runtime.error = "UNLOADED", "无法确定推理框架"
                return runtime
            log_info("选择模型Loader", model_id, type(runtime.loader).__name__)
            runtime.loader.load(spec.path, **spec.load.loader_kwargs())
            load_lora(runtime.loader, spec.lora)
            self._measure_vram(runtime, used_before)
            self._fill_missing_load(runtime)
            self._persist_spec(spec)
            runtime.state, runtime.error = "RUNNING", None
            runtime.last_used_at = time.time()
            self._save_snapshot(runtime)
            log_info("模型加载完成", model_id,
                     "estimated_vram_mb=", spec.estimated_vram_mb,
                     "model_info=", runtime.loader.model_info)
            return runtime
        except Exception as exc:
            if runtime.loader:
                try:
                    runtime.loader.unload()
                except Exception as cleanup_exc:
                    log_info("加载失败后卸载异常", model_id, type(cleanup_exc).__name__, cleanup_exc)
            self.runtime.pop(model_id, None)
            raise RuntimeError(f"模型 {model_id} 加载失败: {exc}") from exc

    def _measure_vram(self, runtime: RuntimeModel, used_before: int) -> None:
        """把加载前后的整卡显存差值归属到本模型。

        已有估算值且 load 未变更时不覆盖，避免多模型驻留时互相污染、估算逐次膨胀。
        """
        spec = runtime.spec
        if spec.estimated_vram_mb is not None and not runtime.needs_remeasure:
            return
        delta = max(0, query_gpu_used_mb() - used_before)
        if delta <= 0:
            log_info("模型显存占用未检测到", spec.model_id)
            return
        spec.estimated_vram_mb = int(delta)
        runtime.needs_remeasure = False
        log_info("模型显存占用实测", spec.model_id, f"{spec.estimated_vram_mb} MB")

    def _fill_missing_load(self, runtime: RuntimeModel) -> None:
        """模型没有 load 节点时，用本次实际生效的参数补全；已配置的不覆盖。"""
        if runtime.spec.load_present or runtime.loader is None:
            return
        effective = runtime.loader.effective_load
        info = runtime.loader.model_info
        if info:
            effective.setdefault("context_length", info.context_length)
            effective.setdefault("dtype", info.dtype)
            # 量化方式不写入配置，只输出诊断信息。
            log_info("模型量化信息", runtime.spec.model_id,
                     info.quantization or info.extra.get("quantization", "未检测到"))
        for name, value in effective.items():
            if name in LOAD_KEYS:
                setattr(runtime.spec.load, name, value)

    # ---------------------------------------------------------------- 持久化

    def _persist_spec(self, spec: ModelSpec) -> None:
        """只补写缺失字段，其余键原样保留。

        绝不能重建整个模型节点：那会把用户手写的 ``engine``/``quantization`` 等
        未被本模块识别的键一并删掉。写入走临时文件 + rename，避免半写入。
        """
        models = self._config.setdefault("models", {})
        node = models.get(spec.model_id)
        if not isinstance(node, dict):
            node = {}
            models[spec.model_id] = node

        changed = False
        if spec.estimated_vram_mb is not None and node.get("estimated_vram_mb") != spec.estimated_vram_mb:
            node["estimated_vram_mb"] = spec.estimated_vram_mb
            changed = True
        if not spec.load_present:
            node["load"] = spec.load.to_dict()
            spec.load_present = True
            changed = True
        if not spec.generation_present:
            spec.generation = copy.deepcopy(self.settings.get("generation", {}))
            node["generation"] = dict(spec.generation)
            spec.generation_present = True
            changed = True
        if not changed:
            return

        # 临时文件保留 .json 后缀，复用 writeFile 的 JSON 序列化分支。
        tmp_path = self.config_path.with_name(f".{self.config_path.stem}.tmp.json")
        writeFile(self._config, str(tmp_path))
        tmp_path.replace(self.config_path)
        log_info("模型配置已补全", spec.model_id)

    def _save_snapshot(self, runtime: RuntimeModel) -> None:
        if not self.cache_settings.get("state_snapshot_enabled", False):
            return
        try:
            save_snapshot(self.asset_root, runtime.spec, state=runtime.state,
                          last_used_at=runtime.last_used_at)
        except OSError as exc:
            log_info("状态快照写入失败", runtime.spec.model_id, exc)

    def restore_from_snapshots(self) -> list[str]:
        """启动时把上次处于 RUNNING 的模型恢复回显存，最近使用的优先。"""
        if not self.cache_settings.get("state_snapshot_enabled", False):
            return []
        pending: list[tuple[float, str]] = []
        for model_id in self.specs:
            runtime_info = (load_snapshot(self.asset_root, model_id) or {}).get("runtime", {})
            if not isinstance(runtime_info, dict) or runtime_info.get("state") != "RUNNING":
                continue
            try:
                pending.append((float(runtime_info.get("last_used_at", 0.0)), model_id))
            except (TypeError, ValueError):
                pending.append((0.0, model_id))
        restored: list[str] = []
        for _, model_id in sorted(pending, reverse=True):
            try:
                self.ensure_loaded(model_id)
                restored.append(model_id)
            except Exception as exc:
                # 单个模型恢复失败不能阻塞服务启动。
                log_info("快照恢复失败", model_id, type(exc).__name__, exc)
        log_info("快照恢复完成", restored or "无")
        return restored

    # ---------------------------------------------------------------- 推理

    def generate(self, model_id: str, prompt: str, **kwargs: Any) -> GenerationResult:
        with self._lock:
            runtime = self.ensure_loaded(model_id)
            if runtime.loader is None:
                raise RuntimeError(f"模型 {model_id} 未加载")
            runtime.active = True
            log_info("开始生成", model_id, "prompt_chars=", len(prompt), "params=", kwargs)
            try:
                result = runtime.loader.generate(prompt, **kwargs)
                runtime.last_used_at = time.time()
                log_info("生成完成", model_id, "tokens=", result.tokens_generated,
                         "seconds=", round(result.time_seconds, 2))
                return result
            except Exception as exc:
                log_info("生成失败", model_id, type(exc).__name__, exc)
                raise
            finally:
                runtime.active = False

    def reconfigure(self, model_id: str, changes: dict[str, Any]) -> RuntimeModel:
        """应用会影响模型创建的参数；只有值确实变化时才卸载重载。"""
        with self._lock:
            runtime = self._get_runtime(model_id)
            spec = self.specs[model_id]
            wanted = {key: value for key, value in changes.items()
                      if key in LOAD_KEYS and getattr(spec.load, key) != value}
            if not wanted:
                return runtime
            if runtime.active:
                raise RuntimeError("模型当前正在生成，不能修改加载参数")
            log_info("重新配置模型", model_id, wanted)
            self._unload_runtime(model_id)
            changed_spec = replace(spec, load=replace(spec.load, **wanted))
            changed_spec.load_fields = set(spec.load_fields) | set(wanted)
            changed_spec.load_present = True
            self.specs[model_id] = changed_spec
            # load 变了，旧的显存估算不再成立，加载后必须重测。
            self._get_runtime(model_id).needs_remeasure = True
            try:
                return self.ensure_loaded(model_id)
            except Exception:
                self.specs[model_id] = spec
                raise

    def update_generation(self, model_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """把生成参数写入模型节点并持久化。

        请求里的 ``deploy`` 只影响当次调用；要改常驻默认值走这里，避免一次试验性的
        采样参数被永久写死。
        """
        with self._lock:
            spec = self.specs[model_id] if model_id in self.specs else None
            if spec is None:
                raise KeyError(f"models.json 中不存在模型: {model_id}")
            merged = self.generation_params(model_id)
            merged.update(changes)
            spec.generation = merged
            spec.generation_present = True
            models = self._config.setdefault("models", {})
            node = models.get(spec.model_id)
            if not isinstance(node, dict):
                node = {}
                models[spec.model_id] = node
            node["generation"] = dict(merged)
            tmp_path = self.config_path.with_name(f".{self.config_path.stem}.tmp.json")
            writeFile(self._config, str(tmp_path))
            tmp_path.replace(self.config_path)
            log_info("生成参数已更新", model_id, changes)
            return merged

    # ---------------------------------------------------------------- 释放

    def sleep(self, model_id: str) -> bool:
        with self._lock:
            runtime = self._get_runtime(model_id)
            if runtime.active or not runtime.loader or runtime.state != "RUNNING":
                return False
            self._release(runtime)
            return True

    def _unload_runtime(self, model_id: str) -> None:
        runtime = self.runtime.get(model_id)
        if not runtime or runtime.active:
            return
        if runtime.loader:
            runtime.state, runtime.sleep_location = "UNLOADED", None
            self._save_snapshot(runtime)
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
        sleep_cfg = self.sleep_settings
        if not sleep_cfg.get("enabled", True):
            return []
        timeout = int(sleep_cfg.get("idle_timeout_sec", 0))
        if timeout <= 0:
            return []
        now = time.time()
        changed: list[str] = []
        with self._lock:
            for model_id, runtime in list(self.runtime.items()):
                # 只回收仍占着显存的 RUNNING 模型；已休眠的再处理一次没有收益。
                if runtime.state != "RUNNING" or runtime.active:
                    continue
                if now - runtime.last_used_at >= timeout and self.sleep(model_id):
                    changed.append(model_id)
        return changed
