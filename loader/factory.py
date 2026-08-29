"""模型加载器工厂。"""

from .base import ModelLoader
from .hf_loader import HFLoader
from .gguf_loader import GGUFLoader
from .model_spec import ModelSpec
from .vllm_loader import VLLMLoader


def create_loader(spec: ModelSpec) -> ModelLoader:
    """根据显式引擎和模型文件格式创建 Loader。"""
    path = spec.path_obj
    has_gguf = path.suffix.lower() == ".gguf"
    if path.is_dir():
        has_gguf = any(path.glob("*.gguf"))
    has_safetensors = (path.suffix.lower() == ".safetensors") or (
        path.is_dir() and any(path.glob("*.safetensors"))
    )
    engine = spec.load.engine.lower()
    if "engine" not in spec.load_fields:
        if has_gguf:
            engine = "gguf"
        elif has_safetensors:
            engine = "hf"
    if engine == "gguf":
        if not has_gguf:
            raise ValueError("配置 engine=gguf，但模型路径中未找到 .gguf 文件")
        return GGUFLoader()
    if engine in {"hf", "transformers", "huggingface"}:
        if has_gguf:
            raise ValueError("GGUF 模型不能使用 HF Loader，请将 engine 设置为 gguf")
        return HFLoader()
    if engine == "vllm":
        return VLLMLoader()
    raise ValueError(f"不支持的模型引擎: {engine}")
