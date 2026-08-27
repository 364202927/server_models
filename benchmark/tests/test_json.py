"""
JSON读写能力测试

测试内容:
- 解析输入JSON数据
- 按要求输出结构化JSON
- 字段完整性和值有效性检查
"""

import json

from ...config import TestDimension
from ...loader.base import ModelLoader
from .base import BaseTest, DimensionResult, TestResult


class JsonCapabilityTest(BaseTest):
    """JSON读写能力测试"""

    dimension = TestDimension.JSON_CAPABILITY
    name = "JSON读写能力"

    def run(self, model: ModelLoader) -> DimensionResult:
        cases = self.test_cases.get("json_capability", {}).get("cases", [])
        results: list[TestResult] = []

        for case in cases:
            test_id = case["id"]
            input_json = case.get("input_json", {})
            prompt_template = case.get("prompt", "")
            required_fields = case.get("required_fields", [])
            valid_values = case.get("valid_signal_values", [])

            # 构建提示
            prompt = prompt_template.replace("{input}", json.dumps(input_json, ensure_ascii=False, indent=2))

            try:
                gen_result = self._generate(model, prompt, max_new_tokens=512)
                response_text = gen_result.text

                # 评估JSON输出能力
                score, details = self._evaluate_json_response(
                    response_text,
                    required_fields,
                    valid_values,
                    case.get("validation")
                )

                results.append(TestResult(
                    test_id=test_id,
                    passed=score >= 60,
                    score=score,
                    details=details,
                    raw_response=response_text
                ))

            except Exception as e:
                results.append(TestResult(
                    test_id=test_id,
                    passed=False,
                    score=0,
                    error=str(e)
                ))

        total_score = sum(r.score for r in results) / len(results) if results else 0

        return DimensionResult(
            dimension=self.dimension,
            total_score=total_score,
            test_results=results,
            summary=f"测试了 {len(results)} 个JSON读写场景"
        )

    def _evaluate_json_response(
        self,
        response: str,
        required_fields: list[str],
        valid_values: list[str],
        validation_type: str | None
    ) -> tuple[float, dict]:
        """
        评估JSON响应质量

        评分规则:
        - 能解析JSON: 40分基础分
        - 字段完整性: 30分
        - 内容正确性: 30分
        """
        details = {
            "json_parseable": False, "fields_found": [], "fields_missing": [],
            "value_valid": True, "structure_score": 0, "content_score": 0
        }

        # 1. 尝试提取JSON
        parsed = self._extract_json(response)
        if parsed is None:
            details["error"] = "Cannot parse JSON from response"
            return 20, details

        details["json_parseable"] = True
        details["structure_score"] = 40

        # 2. 检查必需字段
        obj = parsed[0] if isinstance(parsed, list) and parsed else parsed
        if isinstance(obj, dict):
            for field in required_fields:
                (details["fields_found"] if field in obj else details["fields_missing"]).append(field)

            if required_fields:
                details["structure_score"] += len(details["fields_found"]) / len(required_fields) * 30

        # 3. 检查值的有效性
        if valid_values and isinstance(parsed, dict):
            signal = parsed.get("signal", "").lower()
            if signal in [v.lower() for v in valid_values]:
                details["content_score"] = 30
            else:
                details["value_valid"] = False
                details["content_score"] = 15
        elif validation_type == "math_check":
            details["content_score"] = 20
        else:
            details["content_score"] = 30

        return round(min(100, details["structure_score"] + details["content_score"]), 2), details
