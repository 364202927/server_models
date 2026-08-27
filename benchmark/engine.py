"""通用 benchmark 执行器，不绑定具体问题或模型类型。"""

from __future__ import annotations

from typing import Any

from ..loader.base import ModelLoader
from ..utils.common import readFile
from .checks import evaluate_case


def run_cases(model: ModelLoader, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        prompt = str(case.get("prompt", ""))
        generated = model.generate(prompt)
        score, details = evaluate_case(generated.text, case)
        results.append({"id": case.get("id", ""), "score": score, "passed": score >= float(case.get("pass_score", 60)),
                        "details": details, "response": generated.text})
    return results


def load_cases(path: str) -> list[dict[str, Any]]:
    value = readFile(path)
    return value if isinstance(value, list) else value.get("cases", []) if isinstance(value, dict) else []
