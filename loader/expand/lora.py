"""LoRA 适配器能力检测和生命周期辅助。"""

from __future__ import annotations

from typing import Any


def lora_enabled(lora_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in lora_config if item.get("enabled", True) and item.get("path")]


def load_lora(loader: Any, lora_config: list[dict[str, Any]]) -> list[str]:
    """调用引擎可选的 LoRA 接口；不支持时返回空列表而不阻塞基础模型。"""
    paths = [str(item["path"]) for item in lora_enabled(lora_config)]
    callback = getattr(loader, "load_lora", None)
    if paths and callable(callback):
        callback(paths)
        return paths
    return []


def unload_lora(loader: Any) -> None:
    callback = getattr(loader, "unload_lora", None)
    if callable(callback):
        callback()
