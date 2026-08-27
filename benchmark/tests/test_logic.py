"""
推理与逻辑一致性测试

测试方法: 同一问题用不同表述提问多次，检查答案一致性
评分: 正确性60% + 一致性40%
"""

from ...config import TestDimension
from ...loader.base import ModelLoader
from .base import BaseTest, DimensionResult, TestResult


class LogicConsistencyTest(BaseTest):
    """推理与逻辑一致性测试"""

    dimension = TestDimension.LOGIC_CONSISTENCY
    name = "推理与逻辑一致性"

    def run(self, model: ModelLoader) -> DimensionResult:
        cases = self.test_cases.get("logic_consistency", {}).get("cases", [])
        results: list[TestResult] = []

        for case in cases:
            test_id = case["id"]
            variants = case.get("variants", [])
            expected_pattern = case.get("expected_pattern")
            expected_keywords = case.get("expected_keywords", [])

            if len(variants) < 2:
                results.append(TestResult(
                    test_id=test_id,
                    passed=False,
                    score=0,
                    error="Not enough variants to test consistency"
                ))
                continue

            # 对每个变体生成回答
            responses = []
            for variant in variants:
                try:
                    result = self._generate(model, variant, max_new_tokens=256)
                    responses.append(result.text)
                except Exception as e:
                    responses.append(f"ERROR: {e}")

            # 评估一致性
            score, details = self._evaluate_consistency(
                responses,
                expected_pattern,
                expected_keywords
            )

            results.append(TestResult(
                test_id=test_id,
                passed=score >= 60,
                score=score,
                details=details,
                raw_response="\n---\n".join(responses)
            ))

        # 计算维度总分
        total_score = sum(r.score for r in results) / len(results) if results else 0

        return DimensionResult(
            dimension=self.dimension,
            total_score=total_score,
            test_results=results,
            summary=f"测试了 {len(results)} 组问题变体的一致性"
        )

    def _evaluate_consistency(
        self,
        responses: list[str],
        expected_pattern: str | None,
        expected_keywords: list[str]
    ) -> tuple[float, dict]:
        """
        评估回答的一致性

        评分逻辑:
        - 正确性(60%): 是否匹配预期模式/关键词
        - 一致性(40%): 多个回答之间的相似度
        """
        details = {
            "response_count": len(responses),
            "consistency_score": 0,
            "correctness_score": 0,
            "keywords_found": [],
            "pattern_matched": []
        }

        if not responses:
            return 0, details

        # 1. 评估正确性
        correctness_scores = []
        for i, response in enumerate(responses):
            if expected_pattern and self._matches_pattern(response, expected_pattern):
                correctness_scores.append(100)
                details["pattern_matched"].append(i)
            elif expected_keywords:
                has_kw, found_kw = self._contains_keywords(response, expected_keywords)
                if has_kw:
                    kw_score = len(found_kw) / len(expected_keywords) * 100
                    correctness_scores.append(kw_score)
                    details["keywords_found"].extend(found_kw)
                else:
                    correctness_scores.append(0)
            else:
                correctness_scores.append(50)  # 无预期时给中等分

        avg_correctness = sum(correctness_scores) / len(correctness_scores)
        details["correctness_score"] = round(avg_correctness, 2)

        # 2. 评估一致性 (两两相似度)
        if len(responses) >= 2:
            similarities = [
                self._calculate_similarity(responses[i], responses[j])
                for i in range(len(responses))
                for j in range(i + 1, len(responses))
            ]
            consistency_score = (sum(similarities) / len(similarities)) * 100 if similarities else 100
        else:
            consistency_score = 100

        details["consistency_score"] = round(consistency_score, 2)

        # 综合评分: 正确性60% + 一致性40%
        return round(avg_correctness * 0.6 + consistency_score * 0.4, 2), details
