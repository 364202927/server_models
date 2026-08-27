"""配置驱动的 benchmark 检查器。"""

from __future__ import annotations

import json
import re
from typing import Any


def run_check(response: str, check: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    kind = str(check.get("type", "keywords"))
    if kind == "keywords":
        values = [str(value) for value in check.get("values", [])]
        found = [value for value in values if value.lower() in response.lower()]
        return (len(found) / len(values) if values else 1.0), {"found": found, "missing": [v for v in values if v not in found]}
    if kind == "forbidden_keywords":
        values = [str(value) for value in check.get("values", [])]
        found = [value for value in values if value.lower() in response.lower()]
        return (0.0 if found else 1.0), {"forbidden_found": found}
    if kind == "regex":
        matched = bool(re.search(str(check.get("pattern", "")), response, re.IGNORECASE))
        return float(matched), {"matched": matched}
    if kind == "json":
        try:
            json.loads(response)
            return 1.0, {"parseable": True}
        except json.JSONDecodeError:
            return 0.0, {"parseable": False}
    if kind == "min_length":
        minimum = int(check.get("value", 0))
        return float(len(response) >= minimum), {"length": len(response), "minimum": minimum}
    raise ValueError(f"不支持的 benchmark 检查类型: {kind}")


def evaluate_case(response: str, case: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    checks = case.get("checks", [])
    if not checks:
        return 100.0, {}
    scores: list[float] = []
    details: dict[str, Any] = {}
    for index, check in enumerate(checks):
        score, detail = run_check(response, check)
        scores.append(score)
        details[f"check_{index}"] = detail
    return round(sum(scores) / len(scores) * 100, 2), details
