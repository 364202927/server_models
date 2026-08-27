"""当前服务进程资源信息。"""

from __future__ import annotations

import os
from typing import Any


def process_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0


def process_summary() -> dict[str, Any]:
    return {"pid": os.getpid(), "rss_mb": round(process_memory_mb(), 1)}
