"""
因果推断能力测试

测试内容: 理解市场因果关系链条
评分: 因果链50% + 关键词30% + 推理结构20%
"""

from ...config import TestDimension
from ...loader.base import ModelLoader
from .base import BaseTest, DimensionResult, TestResult

from ...utils.common import aContainB


class CausalReasoningTest(BaseTest):
    """因果推断能力测试"""

    dimension = TestDimension.CAUSAL_REASONING
    name = "因果推断能力"

    def run(self, model: ModelLoader) -> DimensionResult:
        cases = self.test_cases.get("causal_reasoning", {}).get("cases", [])
        results: list[TestResult] = []

        for case in cases:
            test_id = case["id"]
            prompt = case.get("prompt", "")
            expected_chains = case.get("expected_chains", [])
            expected_keywords = case.get("expected_keywords", [])

            try:
                gen_result = self._generate(model, prompt, max_new_tokens=512)
                response_text = gen_result.text

                # 评估因果推断能力
                score, details = self._evaluate_causal_reasoning(
                    response_text,
                    expected_chains,
                    expected_keywords
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
            summary=f"测试了 {len(results)} 个因果推断场景"
        )

    def _evaluate_causal_reasoning(
        self,
        response: str,
        expected_chains: list[dict],
        expected_keywords: list[str]
    ) -> tuple[float, dict]:
        """评估因果推断能力"""
        details = {
            "chains_found": 0, "chains_expected": len(expected_chains),
            "keywords_found": [], "chain_score": 0, "keyword_score": 0, "structure_score": 0
        }

        response_lower = response.lower()

        # 1. 检查因果链 (50分)
        if expected_chains:
            chains_matched = sum(
                1 for chain in expected_chains
                if chain.get("cause", "").lower() in response_lower
                and chain.get("effect", "").lower() in response_lower
            )
            details["chains_found"] = chains_matched
            details["chain_score"] = chains_matched / len(expected_chains) * 50

        # 2. 检查关键词 (30分)
        if expected_keywords:
            details["keywords_found"] = [kw for kw in expected_keywords if kw.lower() in response_lower]
            details["keyword_score"] = len(details["keywords_found"]) / len(expected_keywords) * 30

        # 3. 评估推理结构 (20分) - 检查因果连接词和结构标记
        causal_connectors = [
            "因为", "所以", "导致", "引发", "进而", "从而",
            "因此", "于是", "结果", "使得", "造成",
            "if", "then", "because", "therefore", "thus", "hence"
        ]
        connector_count = sum(1 for c in causal_connectors if c in response_lower)

        # 结构标记检测 - 复用common.aContainB
        structure_markers = ["1.", "1、", "①", "->", "→"]
        sequence_markers = ["首先"]
        has_structure = aContainB(response, structure_markers) or (
            aContainB(response, sequence_markers) and aContainB(response, ["其次", "然后"])
        )

        if connector_count >= 3 and has_structure:
            details["structure_score"] = 20
        elif connector_count >= 2 or has_structure:
            details["structure_score"] = 15
        elif connector_count >= 1:
            details["structure_score"] = 10

        total = details["chain_score"] + details["keyword_score"] + details["structure_score"]
        return round(min(100, total), 2), details
