"""推理引擎扩展：不改变引擎选型逻辑，只在已创建的引擎上叠加能力（如 LoRA）。"""

from .lora import load_lora, lora_enabled, unload_lora

__all__ = ["load_lora", "lora_enabled", "unload_lora"]
