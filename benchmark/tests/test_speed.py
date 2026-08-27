"""
推理速度测试

测试内容: 测量模型生成速度 (tokens/second)
评分: 基于阈值线性插值，>30优秀，<5不及格
"""

from ...config import TestDimension, SpeedThresholds
from ...loader.base import ModelLoader
from .base import BaseTest, DimensionResult, TestResult


class InferenceSpeedTest(BaseTest):
    """推理速度测试"""

    dimension = TestDimension.INFERENCE_SPEED
    name = "推理速度"

    def __init__(self, test_cases: dict, thresholds: SpeedThresholds | None = None):
        super().__init__(test_cases)
        self.thresholds = thresholds or SpeedThresholds()

    def run(self, model: ModelLoader) -> DimensionResult:
        cases = self.test_cases.get("inference_speed", {}).get("cases", [])
        results: list[TestResult] = []

        all_speeds: list[float] = []

        for case in cases:
            test_id = case["id"]
            prompt = case.get("prompt", "")
            min_tokens = case.get("min_tokens", 50)

            try:
                # 预热一次（可选）
                _ = self._generate(model, "Hello", max_new_tokens=10)

                # 正式测试
                gen_result = self._generate(
                    model,
                    prompt,
                    max_new_tokens=min_tokens + 100,  # 给足够空间
                    temperature=0.3
                )

                tokens_per_second = gen_result.tokens_per_second
                all_speeds.append(tokens_per_second)

                # 评分
                score = self._speed_to_score(tokens_per_second)

                results.append(TestResult(
                    test_id=test_id,
                    passed=tokens_per_second >= self.thresholds.minimum,
                    score=score,
                    details={
                        "tokens_per_second": round(tokens_per_second, 2),
                        "tokens_generated": gen_result.tokens_generated,
                        "time_seconds": round(gen_result.time_seconds, 3),
                        "prompt_tokens": gen_result.prompt_tokens
                    },
                    raw_response=gen_result.text[:200] + "..."  # 只保留前200字符
                ))

            except Exception as e:
                results.append(TestResult(
                    test_id=test_id,
                    passed=False,
                    score=0,
                    error=str(e)
                ))

        # 计算平均速度和总分
        if all_speeds:
            avg_speed = sum(all_speeds) / len(all_speeds)
            total_score = self._speed_to_score(avg_speed)
            speed_summary = f"平均速度: {avg_speed:.1f} tokens/s"
        else:
            total_score = 0
            speed_summary = "无有效测试结果"

        return DimensionResult(
            dimension=self.dimension,
            total_score=total_score,
            test_results=results,
            summary=speed_summary
        )

    def _speed_to_score(self, tps: float) -> float:
        """
        将 tokens/second 转换为分数 (0-100)

        使用分段线性插值:
        - >= 30 tps: 100分 (优秀)
        - 20-30 tps: 80-100分 (良好)
        - 10-20 tps: 60-80分 (可接受)
        - 5-10 tps: 40-60分 (最低要求)
        - < 5 tps: 0-40分 (不及格)
        """
        th = self.thresholds
        if tps >= th.excellent:
            return 100
        if tps >= th.good:
            return 80 + (tps - th.good) / (th.excellent - th.good) * 20
        if tps >= th.acceptable:
            return 60 + (tps - th.acceptable) / (th.good - th.acceptable) * 20
        if tps >= th.minimum:
            return 40 + (tps - th.minimum) / (th.acceptable - th.minimum) * 20
        return max(0, 40 * (tps / th.minimum)) if tps > 0 else 0
