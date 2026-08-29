"""模型加载器工厂。"""

from .base import ModelLoader
from .hf_loader import HFLoader
from .gguf_loader import GGUFLoader
from .model_spec import ModelSpec


def create_loader(spec: ModelSpec) -> ModelLoader:
    """根据模型文件后缀创建 Loader。"""
    path = spec.path_obj
    has_gguf = path.suffix.lower() == ".gguf"
    if path.is_dir():
        has_gguf = any(path.glob("*.gguf"))
    if has_gguf:
        return GGUFLoader()
    # safetensors 及标准 Transformers 目录统一由 HF Loader 处理。
    return HFLoader()
