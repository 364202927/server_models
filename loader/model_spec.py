"""模型注册配置与请求级生成参数。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """来自 ``assets/models.json`` 的模型描述。

    模型创建参数由模型配置文件和引擎自动探测决定；这里只保存注册信息与可选覆盖。
    """

    model_id: str
    path: str
    engine: str = "hf"
    model_type: str = "auto"
    quantization: str | None = None
    dtype: str = "float16"
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    estimated_vram_mb: int | None = None
    lora: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def path_obj(self) -> Path:
        return Path(self.path)


def load_model_specs(config: dict[str, Any]) -> dict[str, ModelSpec]:
    """将 JSON 配置转换为强类型模型定义。"""
    defaults = config.get("defaults", {}).get("model", {})
    result: dict[str, ModelSpec] = {}
    for model_id, raw in config.get("models", {}).items():
        values = {**defaults, **raw}
        known = {key: values.pop(key) for key in list(values) if key in {
            "path", "engine", "model_type", "quantization", "dtype", "max_model_len",
            "tensor_parallel_size", "trust_remote_code", "estimated_vram_mb", "lora",
        }}
        if "path" not in known:
            raise ValueError(f"模型 {model_id} 缺少 path 配置")
        result[model_id] = ModelSpec(model_id=model_id, extra=values, **known)
    return result
