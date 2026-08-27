"""
评分计算器 - 分数分解、等级计算、建议生成
"""

from dataclasses import dataclass

from ..utils.common import listFind

from ..config import BenchmarkConfig, DEFAULT_WEIGHTS, DIMENSION_NAMES
from .runner import BenchmarkResult


@dataclass
class ScoreBreakdown:
    """分数分解详情"""
    dimension: str
    dimension_name: str
    raw_score: float
    weight: float
    weighted_score: float
    passed: bool


class Scorer:
    """评分计算器"""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config
        self.weights = config.weights if config else DEFAULT_WEIGHTS

    def get_breakdown(self, result: BenchmarkResult) -> list[ScoreBreakdown]:
        """获取分数分解"""
        breakdowns = []

        for dimension, dim_result in result.dimension_results.items():
            weight = self.weights.get(dimension, 0)
            weighted = dim_result.total_score * weight

            breakdowns.append(ScoreBreakdown(
                dimension=dimension,
                dimension_name=DIMENSION_NAMES.get(dimension, dimension),
                raw_score=round(dim_result.total_score, 2),
                weight=weight,
                weighted_score=round(weighted, 2),
                passed=dim_result.total_score >= 60
            ))

        return sorted(breakdowns, key=lambda x: self.weights.get(x.dimension, 0), reverse=True)

    def calculate_grade(self, total_score: float) -> str:
        """计算等级，复用common.listFind查找匹配的阈值"""
        grade_thresholds = [
            (95, "S"), (90, "A+"), (85, "A"), (80, "B+"),
            (75, "B"), (70, "C+"), (65, "C"), (60, "D"),
        ]
        result = listFind(grade_thresholds, lambda item: total_score >= item[0])
        return result[1] if result else "F"

    def get_strengths_weaknesses(
        self,
        result: BenchmarkResult
    ) -> tuple[list[str], list[str]]:
        """获取优势和劣势"""
        breakdowns = self.get_breakdown(result)

        strengths = []
        weaknesses = []

        for bd in breakdowns:
            if bd.raw_score >= 85:
                strengths.append(f"{bd.dimension_name}: {bd.raw_score:.0f}分")
            elif bd.raw_score < 60:
                weaknesses.append(f"{bd.dimension_name}: {bd.raw_score:.0f}分")

        return strengths, weaknesses

    def get_recommendation(self, result: BenchmarkResult) -> str:
        """根据评估结果生成使用建议"""
        if not result.passed:
            _, weaknesses = self.get_strengths_weaknesses(result)
            if weaknesses:
                weak_areas = "、".join([w.split(":")[0] for w in weaknesses[:2]])
                return f"该模型在 {weak_areas} 等方面存在不足，不建议直接用于量化交易决策。"
            return "该模型未达到量化交易的最低要求，建议选择其他模型。"

        if result.total_score >= 90:
            return "该模型各项能力表现优秀，强烈推荐用于量化交易系统。"
        if result.total_score >= 80:
            return "该模型整体能力良好，适合用于量化交易的信号生成和分析任务。"
        return "该模型基本达标，建议在非关键环节使用，重要决策需人工复核。"
