"""
配置管理模块 - 评估权重、阈值和参数

核心配置说明:
- TestDimension: 定义7个量化交易能力评估维度
- DEFAULT_WEIGHTS: 各维度权重，数学能力权重最高(0.20)因为量化交易核心是数值计算
- BenchmarkConfig: 完整评估配置，包含阈值、生成参数等
"""

from dataclasses import dataclass, field
from pathlib import Path

from .common import dictFind


class TestDimension:
    """评估维度常量 - 7个核心能力维度"""
    LOGIC_CONSISTENCY = "logic_consistency"       # 推理与逻辑一致性
    JSON_CAPABILITY = "json_capability"           # JSON读写能力
    UNSTRUCTURED_DATA = "unstructured_data"       # 非结构化数据理解
    CAUSAL_REASONING = "causal_reasoning"         # 因果推断能力
    INFERENCE_SPEED = "inference_speed"           # 实时性与推理速度
    MATH_PRECISION = "math_precision"             # 数学能力
    RISK_AWARENESS = "risk_awareness"             # 风险控制与不确定性


# 默认权重配置 (总和=1.0)
# 权重设计原则: 数学能力最高因为量化交易本质是数值计算，风险意识其次因为资金安全第一
DEFAULT_WEIGHTS: dict[str, float] = {
    TestDimension.LOGIC_CONSISTENCY: 0.15,
    TestDimension.JSON_CAPABILITY: 0.15,
    TestDimension.UNSTRUCTURED_DATA: 0.10,
    TestDimension.CAUSAL_REASONING: 0.15,
    TestDimension.INFERENCE_SPEED: 0.10,
    TestDimension.MATH_PRECISION: 0.20,       # 量化交易核心能力
    TestDimension.RISK_AWARENESS: 0.15,
}

# 维度中文名称
DIMENSION_NAMES: dict[str, str] = {
    TestDimension.LOGIC_CONSISTENCY: "推理与逻辑一致性",
    TestDimension.JSON_CAPABILITY: "JSON读写能力",
    TestDimension.UNSTRUCTURED_DATA: "非结构化数据理解",
    TestDimension.CAUSAL_REASONING: "因果推断能力",
    TestDimension.INFERENCE_SPEED: "推理速度",
    TestDimension.MATH_PRECISION: "数学精确性",
    TestDimension.RISK_AWARENESS: "风险控制与不确定性",
}


@dataclass
class SpeedThresholds:
    """推理速度阈值 (单位: tokens/s)，用于速度测试评分"""
    excellent: float = 30.0
    good: float = 20.0
    acceptable: float = 10.0
    minimum: float = 5.0       # 低于此值不适合实时交易


@dataclass
class MathThresholds:
    """数学精度阈值 (误差百分比)，金融计算对精度要求高"""
    excellent_error: float = 0.001    # 0.1%
    good_error: float = 0.01          # 1%
    acceptable_error: float = 0.05    # 5%


@dataclass
class EvaluationThresholds:
    """评估通过阈值 - 关键维度(数学/风险)有更高要求"""
    total_pass: float = 70.0
    dimension_minimum: float = 50.0
    critical_dimensions: list[str] = field(default_factory=lambda: [
        TestDimension.MATH_PRECISION,
        TestDimension.RISK_AWARENESS,
    ])
    critical_minimum: float = 60.0        # 关键维度比普通维度高10分


@dataclass
class GenerationParams:
    """模型生成参数 - 低温度确保输出稳定性"""
    max_new_tokens: int = 512
    temperature: float = 0.3          # 量化交易需要确定性输出
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05


@dataclass
class BenchmarkConfig:
    """完整评估配置"""
    # 权重
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    # 阈值
    speed_thresholds: SpeedThresholds = field(default_factory=SpeedThresholds)
    math_thresholds: MathThresholds = field(default_factory=MathThresholds)
    evaluation_thresholds: EvaluationThresholds = field(default_factory=EvaluationThresholds)

    # 生成参数
    generation_params: GenerationParams = field(default_factory=GenerationParams)

    # 测试设置
    consistency_test_rounds: int = 3          # 一致性测试轮数
    speed_test_samples: int = 5               # 速度测试样本数
    custom_test_dir: Path | None = None       # 自定义测试集目录

    # 输出设置
    verbose: bool = False                     # 详细输出
    save_raw_responses: bool = False          # 保存原始响应

    def validate_weights(self) -> bool:
        """验证权重总和是否为1"""
        total = sum(self.weights.values())
        return abs(total - 1.0) < 0.001


# 默认配置实例
DEFAULT_CONFIG = BenchmarkConfig()


class ModelType:
    """支持的模型类型"""
    DEEPSEEK_R1 = "deepseek-r1"
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    GENERIC = "generic"


# 模型类型检测规则: 模型类型 -> 必须包含的关键词列表
MODEL_TYPE_PATTERNS: dict[str, list[str]] = {
    ModelType.DEEPSEEK_R1: ["deepseek", "r1"],
    ModelType.QWEN: ["qwen"],
    ModelType.LLAMA: ["llama"],
    ModelType.MISTRAL: ["mistral"],
    ModelType.PHI: ["phi"],
}


def detect_model_type(model_name: str) -> str:
    """根据模型名称检测类型，复用common.dictFind匹配关键词"""
    name_lower = model_name.lower()
    result = dictFind(MODEL_TYPE_PATTERNS,
                      lambda mtype, patterns: all(p in name_lower for p in patterns))
    return result[0] if result else ModelType.GENERIC
