from .base import BaseTest, TestResult, DimensionResult, load_test_cases
from .test_logic import LogicConsistencyTest
from .test_json import JsonCapabilityTest
from .test_unstructured import UnstructuredDataTest
from .test_causal import CausalReasoningTest
from .test_speed import InferenceSpeedTest
from .test_math import MathPrecisionTest
from .test_risk import RiskAwarenessTest

ALL_TESTS = [
    LogicConsistencyTest,
    JsonCapabilityTest,
    UnstructuredDataTest,
    CausalReasoningTest,
    InferenceSpeedTest,
    MathPrecisionTest,
    RiskAwarenessTest,
]

__all__ = [
    "BaseTest",
    "TestResult",
    "DimensionResult",
    "load_test_cases",
    "LogicConsistencyTest",
    "JsonCapabilityTest",
    "UnstructuredDataTest",
    "CausalReasoningTest",
    "InferenceSpeedTest",
    "MathPrecisionTest",
    "RiskAwarenessTest",
    "ALL_TESTS",
]
