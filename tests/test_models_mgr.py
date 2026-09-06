"""ModelsMgr 的无 GPU 单元测试；真实模型测试需在 Win11 GPU 环境执行。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.loader.models_mgr import ModelsMgr


def _config(models: dict, **defaults) -> dict:
    base = {
        "sleep": {"enabled": True, "idle_timeout_sec": 600, "ram_enabled": True,
                  "ram_reserve_mb": 8192, "gpu_reserve_mb": 512},
        "cache": {"state_snapshot_enabled": False, "cache_dir": "assets/cache"},
        "generation": {"temperature": 0.3, "max_tokens": 512, "system_prompt": ""},
    }
    base.update(defaults)
    return {"defaults": base, "models": models}


def test_empty_registry(tmp_path: Path) -> None:
    config = tmp_path / "models.json"
    config.write_text('{"defaults": {}, "models": {}}', encoding="utf-8")
    assert ModelsMgr(str(config)).status()["models"] == []


# ------------------------------------------------------------------ 配置读写


def test_persist_preserves_unknown_keys(make_mgr) -> None:
    """补写配置不得删除用户手写的未识别字段。"""
    mgr = make_mgr(
        _config({"m1": {"path": "a.gguf", "engine": "vllm", "quantization": "awq",
                        "model_type": "qwen", "estimated_vram_mb": None}}),
        vram={"m1": 4096},
    )
    mgr.ensure_loaded("m1")

    saved = json.loads(mgr.config_path.read_text(encoding="utf-8"))
    node = saved["models"]["m1"]
    assert node["engine"] == "vllm"
    assert node["quantization"] == "awq"
    assert node["model_type"] == "qwen"
    # defaults 也不能被运行期的 settings 调整污染。
    assert saved["defaults"]["cache"]["state_snapshot_enabled"] is False
    assert saved["defaults"]["sleep"]["enabled"] is True


def test_missing_blocks_filled_once(make_mgr) -> None:
    """load / generation 缺失时补写一次，之后不再改动。"""
    mgr = make_mgr(_config({"m1": {"path": "a.gguf"}}), vram={"m1": 4096})
    mgr.ensure_loaded("m1")

    saved = json.loads(mgr.config_path.read_text(encoding="utf-8"))["models"]["m1"]
    assert saved["load"]["dtype"] == "float16"
    assert saved["generation"]["temperature"] == 0.3
    assert saved["estimated_vram_mb"] == 4096

    before = mgr.config_path.read_text(encoding="utf-8")
    mgr.unload("m1")
    mgr.ensure_loaded("m1")
    assert mgr.config_path.read_text(encoding="utf-8") == before


def test_existing_load_block_not_overwritten(make_mgr) -> None:
    mgr = make_mgr(
        _config({"m1": {"path": "a.gguf", "load": {"dtype": "bfloat16", "context_length": 8192}}}),
        vram={"m1": 4096},
    )
    mgr.ensure_loaded("m1")

    saved = json.loads(mgr.config_path.read_text(encoding="utf-8"))["models"]["m1"]
    assert saved["load"] == {"dtype": "bfloat16", "context_length": 8192}


def test_vram_estimate_not_inflated_by_other_models(make_mgr) -> None:
    """A 驻留时加载 B，B 的估算只能是 B 自己的占用。"""
    mgr = make_mgr(
        _config({"a": {"path": "a.gguf"}, "b": {"path": "b.gguf"}}),
        vram={"a": 8000, "b": 6000},
    )
    mgr.ensure_loaded("a")
    mgr.ensure_loaded("b")

    assert mgr.specs["a"].estimated_vram_mb == 8000
    assert mgr.specs["b"].estimated_vram_mb == 6000

    # 反复推理不得让估算值继续爬升，也不得重写配置。
    before = mgr.config_path.read_text(encoding="utf-8")
    for _ in range(3):
        mgr.generate("a", "hi")
        mgr.generate("b", "hi")
    assert mgr.specs["a"].estimated_vram_mb == 8000
    assert mgr.specs["b"].estimated_vram_mb == 6000
    assert mgr.config_path.read_text(encoding="utf-8") == before


# ------------------------------------------------------------------ 准入与释放


def test_reclaims_in_lru_order(make_mgr) -> None:
    """先释放最久未使用的，而不是显存最大的。"""
    mgr = make_mgr(
        _config({"main": {"path": "main.gguf", "estimated_vram_mb": 20000},
                 "expert": {"path": "expert.gguf", "estimated_vram_mb": 10000},
                 "new": {"path": "new.gguf", "estimated_vram_mb": 6000}}),
        vram={"main": 20000, "expert": 10000, "new": 6000},
        can_sleep=False,
    )
    mgr.ensure_loaded("expert")
    mgr.ensure_loaded("main")      # main 更近使用

    mgr.ensure_loaded("new")

    # 腾 6000 只需踢掉 expert；显存最大的 main 必须保住。
    assert mgr.runtime["main"].state == "RUNNING"
    assert "expert" not in mgr.runtime
    assert mgr.runtime["new"].state == "RUNNING"


def test_reclaims_repeatedly_until_enough(make_mgr) -> None:
    """一个不够就继续释放，直到装得下。"""
    mgr = make_mgr(
        _config({"main": {"path": "main.gguf", "estimated_vram_mb": 20000},
                 "expert": {"path": "expert.gguf", "estimated_vram_mb": 10000},
                 "big": {"path": "big.gguf", "estimated_vram_mb": 28000}}),
        vram={"main": 20000, "expert": 10000, "big": 28000},
        can_sleep=False,
    )
    mgr.ensure_loaded("expert")
    mgr.ensure_loaded("main")

    mgr.ensure_loaded("big")

    assert "expert" not in mgr.runtime and "main" not in mgr.runtime
    assert mgr.runtime["big"].state == "RUNNING"


def test_raises_when_reclaim_insufficient(make_mgr) -> None:
    """候选耗尽仍不够时明确报错，而不是放行等 OOM。"""
    mgr = make_mgr(
        _config({"resident": {"path": "r.gguf", "estimated_vram_mb": 4000},
                 "huge": {"path": "h.gguf", "estimated_vram_mb": 40000}}),
        vram={"resident": 4000, "huge": 40000},
        can_sleep=False,
    )
    mgr.ensure_loaded("resident")

    with pytest.raises(RuntimeError, match="显存不足|加载失败"):
        mgr.ensure_loaded("huge")


def test_sleep_prefers_ram_then_unload(make_mgr) -> None:
    mgr = make_mgr(_config({"m1": {"path": "a.gguf", "estimated_vram_mb": 4000}}),
                   vram={"m1": 4000})
    mgr.ensure_loaded("m1")
    assert mgr.sleep("m1") is True
    assert mgr.runtime["m1"].state == "SLEEPING_RAM"
    assert make_mgr.gpu.used == 0

    # RAM 不够时退化为卸载。
    mgr2 = make_mgr(_config({"m2": {"path": "b.gguf", "estimated_vram_mb": 4000}}),
                    vram={"m2": 4000}, ram_available_mb=0)
    mgr2.ensure_loaded("m2")
    assert mgr2.sleep("m2") is True
    assert "m2" not in mgr2.runtime


# ------------------------------------------------------------------ 唤醒


def test_wake_goes_through_admission(make_mgr) -> None:
    """唤醒也要过显存准入，必要时先回收其它模型。"""
    mgr = make_mgr(
        _config({"a": {"path": "a.gguf", "estimated_vram_mb": 20000},
                 "b": {"path": "b.gguf", "estimated_vram_mb": 20000}}),
        vram={"a": 20000, "b": 20000},
    )
    mgr.ensure_loaded("a")
    mgr.sleep("a")
    assert mgr.runtime["a"].state == "SLEEPING_RAM"

    mgr.ensure_loaded("b")       # b 占满显存
    mgr.ensure_loaded("a")       # 唤醒 a 必须先把 b 腾走

    assert mgr.runtime["a"].state == "RUNNING"
    assert mgr.runtime["b"].state == "SLEEPING_RAM"
    assert make_mgr.loaders["a"].wake_calls == 1


def test_wake_failure_keeps_sleeping_state(make_mgr) -> None:
    """唤醒失败时状态不能停在半途，权重仍在 RAM 可重试。"""
    mgr = make_mgr(_config({"a": {"path": "a.gguf", "estimated_vram_mb": 4000}}),
                   vram={"a": 4000})
    mgr.ensure_loaded("a")
    mgr.sleep("a")

    def boom() -> None:
        raise RuntimeError("cuda oom")

    make_mgr.loaders["a"].wake = boom
    with pytest.raises(RuntimeError, match="唤醒失败"):
        mgr.ensure_loaded("a")
    assert mgr.runtime["a"].state == "SLEEPING_RAM"


# ------------------------------------------------------------------ reconfigure


def test_reconfigure_skips_when_unchanged(make_mgr) -> None:
    """deploy 里带着相同的 load 参数不应触发重载。"""
    mgr = make_mgr(
        _config({"m1": {"path": "a.gguf", "estimated_vram_mb": 4000,
                        "load": {"dtype": "float16", "context_length": 4096}}}),
        vram={"m1": 4000},
    )
    mgr.ensure_loaded("m1")
    first = mgr.runtime["m1"].loader

    mgr.reconfigure("m1", {"context_length": 4096, "dtype": "float16"})
    assert mgr.runtime["m1"].loader is first

    mgr.reconfigure("m1", {"context_length": 8192})
    assert mgr.runtime["m1"].loader is not first
    assert mgr.specs["m1"].load.context_length == 8192


# ------------------------------------------------------------------ 空闲回收


def test_reap_idle_skips_already_sleeping(make_mgr) -> None:
    mgr = make_mgr(
        _config({"m1": {"path": "a.gguf", "estimated_vram_mb": 4000}},
                sleep={"enabled": True, "idle_timeout_sec": 1, "ram_enabled": True,
                       "ram_reserve_mb": 8192, "gpu_reserve_mb": 512}),
        vram={"m1": 4000},
    )
    mgr.ensure_loaded("m1")
    mgr.runtime["m1"].last_used_at -= 10

    assert mgr.reap_idle() == ["m1"]
    # 已休眠的模型不应被重复处理。
    assert mgr.reap_idle() == []


def test_reap_idle_respects_disabled(make_mgr) -> None:
    mgr = make_mgr(
        _config({"m1": {"path": "a.gguf", "estimated_vram_mb": 4000}},
                sleep={"enabled": False, "idle_timeout_sec": 1}),
        vram={"m1": 4000},
    )
    mgr.ensure_loaded("m1")
    mgr.runtime["m1"].last_used_at -= 10
    assert mgr.reap_idle() == []


# ------------------------------------------------------------------ 生成参数


def test_generation_layering(make_mgr) -> None:
    """模型级 generation 覆盖 defaults。"""
    mgr = make_mgr(
        _config({"m1": {"path": "a.gguf", "estimated_vram_mb": 4000,
                        "generation": {"temperature": 0.9}}}),
        vram={"m1": 4000},
    )
    params = mgr.generation_params("m1")
    assert params["temperature"] == 0.9
    assert params["max_tokens"] == 512


def test_update_generation_persists(make_mgr) -> None:
    mgr = make_mgr(_config({"m1": {"path": "a.gguf", "estimated_vram_mb": 4000}}),
                   vram={"m1": 4000})
    merged = mgr.update_generation("m1", {"temperature": 1.1})
    assert merged["temperature"] == 1.1

    saved = json.loads(mgr.config_path.read_text(encoding="utf-8"))
    assert saved["models"]["m1"]["generation"]["temperature"] == 1.1


# ------------------------------------------------------------------ 重启恢复


def test_restore_from_snapshots(make_mgr) -> None:
    mgr = make_mgr(
        _config({"a": {"path": "a.gguf", "estimated_vram_mb": 4000},
                 "b": {"path": "b.gguf", "estimated_vram_mb": 4000}},
                cache={"state_snapshot_enabled": True, "cache_dir": "assets/cache"}),
        vram={"a": 4000, "b": 4000},
    )
    mgr.ensure_loaded("a")       # a 保持 RUNNING，写下快照
    mgr.ensure_loaded("b")
    mgr.unload("b")              # b 快照记为 UNLOADED

    fresh = ModelsMgr(str(mgr.config_path))
    fresh.settings = mgr.settings
    restored = fresh.restore_from_snapshots()
    assert restored == ["a"]
