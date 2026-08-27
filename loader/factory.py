"""模型加载器工厂。"""

from .base import ModelLoader
from .hf_loader import HFLoader
from .model_spec import ModelSpec
from .vllm_loader import VLLMLoader


def create_loader(spec: ModelSpec) -> ModelLoader:
    """根据模型引擎创建统一 Loader。"""
    if spec.engine.lower() in {"hf", "transformers", "huggingface"}:
        return HFLoader()
    if spec.engine.lower() == "vllm":
        return VLLMLoader()
    raise ValueError(f"不支持的模型引擎: {spec.engine}")
