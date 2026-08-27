"""
评估执行器 - 协调所有测试的运行

核心流程:
1. 初始化所有维度测试
2. 依次执行各维度测试
3. 计算加权总分
4. 判断是否通过评估
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..utils.common import switch

from ..config import BenchmarkConfig, DEFAULT_CONFIG, TestDimension
from ..loader.base import ModelLoader, ModelInfo
from ..hardware import HardwareInfo, detect_hardware
from .tests.base import load_test_cases, DimensionResult
from .tests import (LogicConsistencyTest, JsonCapabilityTest, UnstructuredDataTest,CausalReasoningTest, InferenceSpeedTest, MathPrecisionTest, RiskAwarenessTest)


@dataclass
class BenchmarkResult:
    """完整评估结果"""
    model_info: ModelInfo
    hardware_info: HardwareInfo
    dimension_results: dict[str, DimensionResult]
    total_score: float
    weighted_scores: dict[str, float]
    passed: bool
    verdict: str
    config: BenchmarkConfig = field(default_factory=lambda: DEFAULT_CONFIG)


class BenchmarkRunner:
    """评估执行器"""

    def __init__(self,config: BenchmarkConfig | None = None,test_cases_path: Path | None = None):
        self.config = config or DEFAULT_CONFIG
        self.test_cases = load_test_cases(test_cases_path)
        self.hardware_info: HardwareInfo | None = None
        self._progress_callback: Callable[[str, int, int], None] | None = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """设置进度回调 callback(dimension_name, current, total)"""
        self._progress_callback = callback

    def run(self, model: ModelLoader) -> BenchmarkResult:
        """执行完整评估，返回评估结果"""
        if not model.is_loaded:
            raise RuntimeError("Model not loaded. Call model.load() first.")

        self.hardware_info = detect_hardware()

        # 初始化7个维度测试
        tests = [
            LogicConsistencyTest(self.test_cases),
            JsonCapabilityTest(self.test_cases),
            UnstructuredDataTest(self.test_cases),
            CausalReasoningTest(self.test_cases),
            InferenceSpeedTest(self.test_cases, self.config.speed_thresholds),
            MathPrecisionTest(self.test_cases, self.config.math_thresholds),
            RiskAwarenessTest(self.test_cases),
        ]

        # 运行所有测试并收集结果
        dimension_results: dict[str, DimensionResult] = {}
        for i, test in enumerate(tests):
            if self._progress_callback:
                self._progress_callback(test.dimension, i + 1, len(tests))
            dimension_results[test.dimension] = test.run(model)

        # 计算加权分数
        weighted_scores = {dim: result.total_score * self.config.weights.get(dim, 0)
                            for dim, result in dimension_results.items()}
        total_score = sum(weighted_scores.values())

        # 判断是否通过
        passed, verdict = self._evaluate_pass(dimension_results, total_score)

        return BenchmarkResult(
                        model_info=model.model_info,
                        hardware_info=self.hardware_info,
                        dimension_results=dimension_results,
                        total_score=round(total_score, 2),
                        weighted_scores={k: round(v, 2) for k, v in weighted_scores.items()},
                        passed=passed,
                        verdict=verdict,
                        config=self.config)

    def _evaluate_pass(self,results: dict[str, DimensionResult],total_score: float) -> tuple[bool, str]:
        """
        评估是否通过

        通过条件:
        1. 总分 >= 70
        2. 关键维度(数学/风险) >= 60
        3. 所有维度 >= 50
        """
        thresholds = self.config.evaluation_thresholds

        # 检查总分
        if total_score < thresholds.total_pass:
            return False, f"总分 {total_score:.1f} 低于通过线 {thresholds.total_pass}"

        # 检查关键维度
        for critical in thresholds.critical_dimensions:
            if critical in results and results[critical].total_score < thresholds.critical_minimum:
                dim_name = critical.replace("_", " ").title()
                return False, f"关键维度 [{dim_name}] 得分 {results[critical].total_score:.1f} 低于最低要求 {thresholds.critical_minimum}"

        # 检查各维度最低分
        for dimension, result in results.items():
            if result.total_score < thresholds.dimension_minimum:
                dim_name = dimension.replace("_", " ").title()
                return False, f"维度 [{dim_name}] 得分 {result.total_score:.1f} 低于最低要求 {thresholds.dimension_minimum}"

        # 根据总分给出评价
        if total_score >= 90:
            return True, "优秀 - 强烈推荐用于量化交易"
        if total_score >= 80:
            return True, "良好 - 适合量化交易使用"
        return True, "合格 - 可以谨慎用于量化交易"

    def run_single_dimension(self, model: ModelLoader, dimension: str) -> DimensionResult | None:
        """运行单个维度测试，复用common.switch进行字典分支查询"""
        test_map = {
            TestDimension.LOGIC_CONSISTENCY: LogicConsistencyTest,
            TestDimension.JSON_CAPABILITY: JsonCapabilityTest,
            TestDimension.UNSTRUCTURED_DATA: UnstructuredDataTest,
            TestDimension.CAUSAL_REASONING: CausalReasoningTest,
            TestDimension.INFERENCE_SPEED: lambda tc: InferenceSpeedTest(tc, self.config.speed_thresholds),
            TestDimension.MATH_PRECISION: lambda tc: MathPrecisionTest(tc, self.config.math_thresholds),
            TestDimension.RISK_AWARENESS: RiskAwarenessTest}
        test_class = switch(test_map, dimension)
        if not test_class:
            return None

        test = test_class(self.test_cases)
        return test.run(model)
