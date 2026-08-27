"""RAM 和显存释放策略。"""

from __future__ import annotations

from dataclasses import dataclass

from .detector import detect_memory


@dataclass(frozen=True)
class RamDecision:
    allowed: bool
    available_mb: int
    required_mb: int
    reason: str = ""


def check_ram(required_mb: int, reserve_mb: int = 8192) -> RamDecision:
    available = detect_memory().available_mb
    required = max(0, required_mb) + max(0, reserve_mb)
    return RamDecision(available >= required, available, required_mb,
                       "RAM 不足" if available < required else "")
