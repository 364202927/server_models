"""
非结构化数据理解测试

测试内容: 解析财经新闻/公告，提取多空信号和关键因素
评分: 情绪判断50% + 关键因素30% + 推理质量20%
"""

from ...config import TestDimension
from ...loader.base import ModelLoader
from .base import BaseTest, DimensionResult, TestResult

from ...utils.common import aContainB


class UnstructuredDataTest(BaseTest):
    """非结构化数据理解测试"""

    dimension = TestDimension.UNSTRUCTURED_DATA
    name = "非结构化数据理解"

    # 情绪关键词映射表
    SENTIMENT_KEYWORDS = {
        "bullish": ["看涨", "利好", "上涨", "买入", "多头", "增长", "突破", "bullish", "buy", "long"],
        "bearish": ["看跌", "利空", "下跌", "卖出", "空头", "下降", "跌破", "bearish", "sell", "short"],
        "neutral": ["中性", "观望", "震荡", "持平", "不确定", "neutral", "hold", "sideways"],
    }

    def run(self, model: ModelLoader) -> DimensionResult:
        cases = self.test_cases.get("unstructured_data", {}).get("cases", [])
        results: list[TestResult] = []

        for case in cases:
            test_id = case["id"]
            text = case.get("text", "")
            prompt_template = case.get("prompt", "")
            expected_sentiment = case.get("expected_sentiment", "")
            key_factors = case.get("key_factors", [])

            # 构建提示
            prompt = prompt_template.replace("{input}", text)

            try:
                gen_result = self._generate(model, prompt, max_new_tokens=384)
                response_text = gen_result.text

                # 评估理解能力
                score, details = self._evaluate_understanding(
                    response_text,
                    expected_sentiment,
                    key_factors
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
            summary=f"测试了 {len(results)} 个非结构化数据理解场景"
        )

    def _evaluate_understanding(
        self,
        response: str,
        expected_sentiment: str,
        key_factors: list[str]
    ) -> tuple[float, dict]:
        """评估非结构化数据理解能力"""
        details = {
            "detected_sentiment": None, "sentiment_correct": False,
            "factors_found": [], "factors_missing": [],
            "sentiment_score": 0, "factor_score": 0, "reasoning_quality": 0
        }

        response_lower = response.lower()

        # 1. 检测情绪 - 计算各情绪关键词出现次数
        sentiment_scores = {
            sentiment: sum(1 for kw in keywords if kw.lower() in response_lower)
            for sentiment, keywords in self.SENTIMENT_KEYWORDS.items()
        }
        detected = max(sentiment_scores, key=sentiment_scores.get)
        if sentiment_scores[detected] > 0:
            details["detected_sentiment"] = detected

        # 评估情绪判断正确性 (50分)
        if expected_sentiment:
            if details["detected_sentiment"] == expected_sentiment:
                details["sentiment_correct"] = True
                details["sentiment_score"] = 50
            elif details["detected_sentiment"] is None:
                details["sentiment_score"] = 20
            else:
                details["sentiment_score"] = 10

        # 2. 检测关键因素 (30分)
        for factor in key_factors:
            target = details["factors_found"] if factor.lower() in response_lower else details["factors_missing"]
            target.append(factor)
        if key_factors:
            details["factor_score"] = len(details["factors_found"]) / len(key_factors) * 30

        # 3. 评估推理质量 (20分) - 检查是否有因果推理词
        causal_words = ["因为", "所以", "导致", "由于", "因此", "表明", "说明", "意味着"]
        has_reasoning = aContainB(response, causal_words)
        if has_reasoning and len(response) > 100:
            details["reasoning_quality"] = 20
        elif len(response) > 50:
            details["reasoning_quality"] = 10

        total = details["sentiment_score"] + details["factor_score"] + details["reasoning_quality"]
        return round(min(100, total), 2), details
