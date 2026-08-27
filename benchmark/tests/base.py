"""
测试基类 - 定义测试接口和通用工具方法

提供:
- TestResult/DimensionResult: 测试结果数据结构
- BaseTest: 测试基类，包含JSON提取、关键词匹配等工具方法
- load_test_cases: 测试用例加载函数
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...utils.common import readFile, aContainB

from ...loader.base import ModelLoader, GenerationResult


@dataclass
class TestResult:
    """单个测试结果"""
    test_id: str
    passed: bool
    score: float                      # 0-100
    details: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    error: str | None = None


@dataclass
class DimensionResult:
    """维度测试结果"""
    dimension: str
    total_score: float                # 0-100
    test_results: list[TestResult]
    summary: str = ""


class BaseTest(ABC):
    """测试基类，所有维度测试继承此类"""

    dimension: str = ""    # 子类必须定义
    name: str = ""         # 测试显示名称

    def __init__(self, test_cases: dict):
        self.test_cases = test_cases

    @abstractmethod
    def run(self, model: ModelLoader) -> DimensionResult:
        """运行测试，返回维度结果"""
        pass

    def _generate(self, model: ModelLoader, prompt: str, **kwargs) -> GenerationResult:
        """调用模型生成"""
        return model.generate(prompt, **kwargs)

    @staticmethod
    def _extract_json(text: str) -> dict | list | None:
        """从文本中提取JSON，支持markdown代码块和原始JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 提取 ```json ... ``` 块
        for match in re.findall(r'```(?:json)?\s*([\s\S]*?)```', text):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # 提取 { } 或 [ ] 块
        for match in re.findall(r'(\{[\s\S]*\}|\[[\s\S]*\])', text):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _contains_keywords(text: str, keywords: list[str]) -> tuple[bool, list[str]]:
        """检查文本是否包含关键词，复用common.aContainB"""
        text_lower = text.lower()
        found = [k for k in keywords if k.lower() in text_lower]
        return aContainB(text_lower, [k.lower() for k in keywords]), found

    @staticmethod
    def _matches_pattern(text: str, pattern: str) -> bool:
        """检查文本是否匹配正则表达式"""
        return bool(re.search(pattern, text, re.IGNORECASE))

    @staticmethod
    def _extract_number(text: str) -> float | None:
        """从文本中提取数字，支持百分比和逗号分隔"""
        patterns = [
            r'[-+]?\d+\.?\d*%?',                   # 基本数字和百分比
            r'[-+]?\d{1,3}(?:,\d{3})*\.?\d*',     # 带逗号的数字
        ]
        for pattern in patterns:
            if match := re.search(pattern, text):
                num_str = match.group().replace(',', '').replace('%', '')
                try:
                    return float(num_str)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """计算两个文本的Jaccard相似度 (基于词集合)"""
        words1, words2 = set(text1.lower().split()), set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)


def load_test_cases(test_file: Path | None = None) -> dict:
    """加载测试用例，复用common.readFile自动识别json格式"""
    if test_file is None:
        test_file = Path(__file__).parent.parent / "test_cases" / "default.json"
    return readFile(str(test_file))
