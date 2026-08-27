"""
数学精确性测试

测试内容: 金融计算能力 (盈亏计算、收益率、杠杆等)
评分: 基于计算误差百分比，<=0.1%优秀，>5%不及格
"""

import re

from ...config import TestDimension, MathThresholds
from ...loader.base import ModelLoader
from .base import BaseTest, DimensionResult, TestResult

from ...utils.common import aContainB


class MathPrecisionTest(BaseTest):
    """数学精确性测试"""

    dimension = TestDimension.MATH_PRECISION
    name = "数学精确性"

    def __init__(self, test_cases: dict, thresholds: MathThresholds | None = None):
        super().__init__(test_cases)
        self.thresholds = thresholds or MathThresholds()

    def run(self, model: ModelLoader) -> DimensionResult:
        cases = self.test_cases.get("math_precision", {}).get("cases", [])
        results: list[TestResult] = []

        for case in cases:
            test_id = case["id"]
            prompt = case.get("prompt", "")
            expected = case.get("expected")
            tolerance = case.get("tolerance", 0.01)
            math_type = case.get("type", "")

            try:
                gen_result = self._generate(model, prompt, max_new_tokens=256, temperature=0.1)
                response_text = gen_result.text

                # 评估数学计算
                score, details = self._evaluate_math(
                    response_text,
                    expected,
                    tolerance,
                    math_type,
                    case
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
            summary=f"测试了 {len(results)} 道金融数学题"
        )

    def _evaluate_math(
        self,
        response: str,
        expected: float | None,
        tolerance: float,
        math_type: str,
        case: dict
    ) -> tuple[float, dict]:
        """
        评估数学计算结果

        评分规则:
        - 误差 <= 0.1%: 100分
        - 误差 <= 1%: 80分
        - 误差 <= 5%: 60分
        - 误差 <= 10%: 40分
        - 否则: 20分
        - 显示计算过程: +10分
        """
        details = {
            "expected": expected, "extracted": None, "error": None,
            "error_percent": None, "calculation_shown": False
        }

        # 检查是否有计算过程 - 复用common.aContainB
        calc_indicators = ["=", "×", "+", "-", "*", "/", "计算", "得到", "结果"]
        details["calculation_shown"] = aContainB(response, calc_indicators)

        # 提取数字
        extracted = self._extract_number(response)
        if extracted is None:
            details["error"] = "无法从回答中提取数字"
            return 30 if details["calculation_shown"] else 10, details

        details["extracted"] = extracted

        # 特殊处理盈亏计算
        if math_type == "pnl_calculation":
            return self._evaluate_pnl(response, case, details)

        if expected is None:
            return 50, details

        # 计算误差百分比
        error_percent = abs(extracted - expected) / abs(expected) if expected != 0 else abs(extracted)
        details["error_percent"] = round(error_percent * 100, 4)

        # 根据误差评分
        th = self.thresholds
        if error_percent <= th.excellent_error:
            score = 100
        elif error_percent <= th.good_error:
            score = 80
        elif error_percent <= th.acceptable_error:
            score = 60
        elif error_percent <= 0.1:
            score = 40
        else:
            score = 20

        # 显示计算过程加分
        if details["calculation_shown"] and score < 100:
            score = min(100, score + 10)

        return score, details

    def _evaluate_pnl(self, response: str, case: dict, details: dict) -> tuple[float, dict]:
        """评估盈亏计算 - 需要同时验证PnL值和收益率"""
        expected_pnl = case.get("expected_pnl")
        expected_return = case.get("expected_return")
        tolerance_percent = case.get("tolerance_percent", 1)

        # 提取响应中的所有数字
        numbers = [float(n) for n in re.findall(r'[-+]?\d+\.?\d*', response) if n]

        # 检查是否找到正确的值
        pnl_found = any(
            abs(num - expected_pnl) / abs(expected_pnl) <= tolerance_percent / 100
            for num in numbers
        ) if expected_pnl else False

        return_found = any(
            abs(num - expected_return) / abs(expected_return) <= tolerance_percent / 100
            for num in numbers
        ) if expected_return else False

        details["pnl_correct"] = pnl_found
        details["return_correct"] = return_found

        if pnl_found and return_found:
            return 100, details
        if pnl_found or return_found:
            return 70, details
        return 30, details

    def _extract_number(self, text: str) -> float | None:
        """从文本中提取数字，优先匹配带单位或等号后的数字"""
        patterns = [
            r'([-+]?\d{1,3}(?:,\d{3})*\.?\d*)\s*(?:美元|元|USD|\$|%)',  # 带单位
            r'(?:等于|=|是|为)\s*([-+]?\d+\.?\d*)',                      # 等于xxx
            r'([-+]?\d+\.?\d+)',                                        # 小数
            r'([-+]?\d+)',                                              # 整数
        ]
        for pattern in patterns:
            if matches := re.findall(pattern, text):
                try:
                    return float(matches[-1].replace(',', ''))  # 取最后一个匹配
                except ValueError:
                    continue
        return None
