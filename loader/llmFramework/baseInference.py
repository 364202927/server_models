"""
baseInference.py
推理框架抽象基类。

统一"模型逻辑"(messages 归一化、工具提示词渲染、采样参数组装、显存/内存
统计、休眠唤醒默认行为),子类(vllm/sglang/llama...)只负责:
- 创建/持有具体推理引擎实例(``_create_engine`` / ``load``)
- 单次引擎调用(``_run_engine``,或整体覆盖 ``generate`` —— 引擎形状差异
  太大时,如 llama 的消息级 chat-completions 接口)
- 提供 tokenizer(``_get_tokenizer``)

新增推理框架时,优先复用本类已有的模板方法,只在真正与已有引擎形状不同的
地方覆盖。
"""

from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# torch/psutil 都是可选依赖:在模块级尝试一次,失败则置为 None,
# 后续方法用 `is not None` 判断即可,不需要在每个方法里各自 try/except。
try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None


def detect_model_type(name: str) -> str:
    """从模型名/路径粗略猜测模型系列,仅用于诊断信息展示。"""
    lowered = name.lower()
    for marker in ("qwen", "llama", "mistral", "deepseek", "yi", "glm", "baichuan"):
        if marker in lowered:
            return marker
    return "unknown"


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    model_type: str
    quantization: str | None = None
    dtype: str = "float16"
    parameters: str = "unknown"         # 参数量 e.g. "7B"
    context_length: int = 4096
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    prompt_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


class ToolCapabilityError(ValueError):
    """The configured model backend cannot honor a tool request."""


class ToolOutputError(RuntimeError):
    """The model produced an invalid tool call."""


@dataclass
class MemoryUsage:
    """模型内存/显存占用信息"""
    gpu_allocated_mb: float = 0       # GPU已分配显存 (模型+推理临时)
    gpu_reserved_mb: float = 0        # GPU预留显存 (含缓存池)
    gpu_total_mb: float = 0           # GPU总显存
    gpu_free_mb: float = 0            # GPU空闲显存
    process_rss_mb: float = 0         # 进程物理内存占用
    system_available_mb: float = 0    # 系统可用内存
    details: dict[str, Any] | None = None  # verbose模式下的详细信息


class baseInference(ABC):
    """
    推理框架抽象基类

    使用方式:
        with vllm() as engine:
            engine.load("model_path")
            result = engine.generate("prompt")

    子类命名与文件名保持一致且全小写(``vllm``/``sglang``/``llama``),
    这是 ``utils.common.require()`` 动态创建子类的硬性要求——它取模块路径
    最后一段作为类名去查找。
    """

    # 采样参数改名映射:通用键 -> 引擎实际接受的关键字。未出现的键原样传递。
    _SAMPLING_KEY_MAP: dict[str, str] = {}
    # 除通用五个采样键外,该引擎还认识的可选采样字段(从 kwargs 里按需摘取)。
    _EXTRA_SAMPLING_KEYS: tuple[str, ...] = ()
    # HF/vLLM 类模型解码后残留的特殊 token,工具输出解析后需要清理。
    _STRIP_TOKENS: tuple[str, ...] = ()

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_info: ModelInfo | None = None
        self._effective_load: dict[str, Any] = {}
        self._tool_parser: str | None = None

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def effective_load(self) -> dict[str, Any]:
        """返回本次加载实际采用的参数,供 ModelsMgr 同步到 models.json。"""
        return dict(self._effective_load)

    @abstractmethod
    def load(self, model_path: str, *, quantization: str | None = None, dtype: str = "float16",
             max_model_len: int | None = None, tensor_parallel_size: int = 1,
             trust_remote_code: bool = True, **kwargs: Any) -> "baseInference":
        """加载模型,返回self支持链式调用"""

    @abstractmethod
    def unload(self) -> None:
        """卸载模型,释放显存"""

    def sleep_to_ram(self) -> bool:
        """将模型权重移出 GPU 保留在 RAM;引擎不支持时返回 False。"""
        return False

    def wake(self) -> None:
        """唤醒 RAM 中的模型;不支持休眠的引擎无需实现。"""
        return None

    def supports_state_snapshot(self) -> bool:
        return True

    def supports_prompt_cache(self) -> bool:
        return False

    def supports_kv_cache_persistence(self) -> bool:
        return False

    def load_lora(self, paths: list[str]) -> None:
        """加载适配器;具体引擎不支持时显式报告,避免静默误用。"""
        raise NotImplementedError("当前推理框架不支持 LoRA")

    def unload_lora(self) -> None:
        """卸载已加载的适配器。"""
        return None

    def memory_usage(self, verbose: bool = False) -> MemoryUsage:
        """
        检测当前模型内存/显存占用

        Args:
            verbose: True时返回详细信息(各GPU设备、模型参数量等)
        """
        usage = MemoryUsage()
        if torch is not None and torch.cuda.is_available():
            self._fill_gpu_usage(usage, verbose)

        usage.process_rss_mb = round(self._get_process_rss_mb(), 1)
        usage.system_available_mb = round(self._get_system_available_mb(), 1)

        if verbose:
            usage.details = usage.details or {}
            usage.details["model_loaded"] = self.is_loaded
            if self._model_info:
                usage.details["model_name"] = self._model_info.name
                usage.details["parameters"] = self._model_info.parameters

        return usage

    def _fill_gpu_usage(self, usage: MemoryUsage, verbose: bool) -> None:
        """填充当前设备的 GPU 显存占用;verbose 且多卡时附加各设备明细。"""
        device_index = self._get_gpu_device_index()
        usage.gpu_allocated_mb = round(torch.cuda.memory_allocated(device_index) / (1024 ** 2), 1)
        usage.gpu_reserved_mb = round(torch.cuda.memory_reserved(device_index) / (1024 ** 2), 1)
        usage.gpu_total_mb = round(torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 2), 1)
        usage.gpu_free_mb = round(usage.gpu_total_mb - usage.gpu_reserved_mb, 1)
        if verbose and torch.cuda.device_count() > 1:
            usage.details = usage.details or {}
            usage.details["gpu_devices"] = [self._gpu_device_detail(i) for i in range(torch.cuda.device_count())]

    @staticmethod
    def _gpu_device_detail(index: int) -> dict[str, Any]:
        return {"device": index, "name": torch.cuda.get_device_properties(index).name,
                "allocated_mb": round(torch.cuda.memory_allocated(index) / (1024 ** 2), 1),
                "reserved_mb": round(torch.cuda.memory_reserved(index) / (1024 ** 2), 1)}

    def release_cache(self) -> MemoryUsage:
        """
        释放推理过程中产生的临时显存/内存占用 (KV cache, 临时张量等)
        保持模型本身已加载状态不变

        Returns:
            清理后的MemoryUsage
        """
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return self.memory_usage()

    def _get_gpu_device_index(self) -> int:
        """获取模型所在的GPU设备索引,子类可覆盖"""
        return 0

    @staticmethod
    def _get_process_rss_mb() -> float:
        """获取当前进程物理内存占用(MB)"""
        if psutil is not None:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        return baseInference._read_proc_field("/proc/self/status", "VmRSS:")

    @staticmethod
    def _get_system_available_mb() -> float:
        """获取系统可用内存(MB)"""
        if psutil is not None:
            return psutil.virtual_memory().available / (1024 ** 2)
        return baseInference._read_proc_field("/proc/meminfo", "MemAvailable:")

    @staticmethod
    def _read_proc_field(path: str, prefix: str) -> float:
        """从 /proc 下的键值文件读取一个以 kB 为单位的字段并换算为 MB;
        文件不存在、无权限或格式异常时返回 0(psutil 不可用时的兜底路径,仅 Linux 有效)。"""
        try:
            with open(path) as handle:
                for line in handle:
                    if line.startswith(prefix):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except (OSError, ValueError, IndexError):
            pass
        return 0

    def _extract_model_info(self, model_path: str, **kwargs) -> ModelInfo:
        """从路径提取模型信息"""
        path = Path(model_path)
        name = path.name if path.exists() else model_path.split("/")[-1]
        return ModelInfo(
            name=name, path=model_path, model_type=detect_model_type(name),
            quantization=kwargs.get("quantization"), dtype=kwargs.get("dtype", "float16"),
        )

    def __enter__(self) -> "baseInference":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    # ---------------------------------------------------------------- 生成:模板方法
    #
    # 默认实现覆盖"渲染 prompt 字符串 -> 单次调用引擎 -> 解析文本"这一形状,
    # 适用于 vllm/sglang 这类整段 prompt 进、整段文本出的引擎:子类只需实现
    # `_run_engine`。llama(GGUF)的后端是消息级 chat-completions 接口,
    # 形状不同,完整覆盖本方法。

    def generate(self, prompt: str, *, max_new_tokens: int = 512, temperature: float = 0.3,
                top_p: float = 0.95, top_k: int = 50, repetition_penalty: float = 1.05,
                stop_sequences: list[str] | None = None, system_prompt: str = "",
                **kwargs: Any) -> GenerationResult:
        """生成文本,返回GenerationResult;system_prompt 非空时作为系统角色注入"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = self._normalize_messages(prompt, system_prompt, kwargs)
        tools = kwargs.get("tools") or []
        tool_choice = kwargs.get("tool_choice")
        tool_mode = bool(tools and tool_choice != "none")

        # 分支键是 tools,不是 messages:没有工具时走普通聊天模板,
        # 和只有一条 user 消息时的行为完全一致。
        rendered = (self._build_tool_prompt(messages, tools, tool_choice)
                   if tool_mode else self._build_chat_prompt(messages))
        sampling = self._build_sampling(max_new_tokens, temperature, top_p, top_k,
                                        repetition_penalty, stop_sequences, kwargs)

        start_time = time.perf_counter()
        text, tokens_generated, prompt_tokens = self._run_engine(rendered, sampling)
        elapsed = time.perf_counter() - start_time

        content, tool_calls = self._parse_tool_output(text) if tool_mode else (text, [])
        return GenerationResult(
            text=content, tokens_generated=tokens_generated, time_seconds=elapsed,
            tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0,
            prompt_tokens=prompt_tokens, tool_calls=tool_calls,
            finish_reason=self._finish_reason(tool_calls, tokens_generated, max_new_tokens),
        )

    def _run_engine(self, rendered_prompt: str, sampling: dict[str, Any]) -> tuple[str, int, int]:
        """执行一次引擎调用,返回 (生成文本, 生成token数, prompt token数)。

        使用默认 ``generate()`` 模板方法的子类(vllm/sglang)必须实现本方法;
        完整覆盖 ``generate()`` 的子类(llama)不需要。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 _run_engine")

    @staticmethod
    def _normalize_messages(prompt: str, system_prompt: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """兼容绕过 MsgHandler 直接调用引擎的场景(脚本/测试);
        正常链路里 MsgHandler 已经把 prompt 统一成 messages。"""
        messages = kwargs.get("messages")
        if messages is not None:
            return messages
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return messages + [{"role": "user", "content": prompt}]

    def _get_tokenizer(self) -> Any:
        """返回用于渲染 chat template 的 tokenizer 对象,子类覆盖。"""
        return self._tokenizer

    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        """无工具场景:优先用 tokenizer 的 chat template 渲染完整多轮对话,不支持时退化为逐条拼接。"""
        try:
            tokenizer = self._get_tokenizer()
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError):
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)

    def _build_tool_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                           tool_choice: Any) -> str:
        """有工具场景:渲染带工具 schema 的 chat template。当前要求 tool_parser=hermes_json。"""
        import copy
        import json

        if self._tool_parser != "hermes_json":
            raise ToolCapabilityError(f"{type(self).__name__} 工具调用需要配置 load.tool_parser=hermes_json")
        prepared = copy.deepcopy(messages)
        for message in prepared:
            for call in message.get("tool_calls", []):
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        if tool_choice == "none":
            tools = []
        elif tool_choice == "required":
            prepared.insert(0, {"role": "system", "content": "你必须调用至少一个可用工具。"})
        elif isinstance(tool_choice, dict):
            name = tool_choice["function"]["name"]
            tools = [tool for tool in tools if tool["function"]["name"] == name]
            prepared.insert(0, {"role": "system", "content": f"你必须调用工具 {name}。"})
        try:
            tokenizer = self._get_tokenizer()
            return tokenizer.apply_chat_template(prepared, tools=tools, tokenize=False,
                                                 add_generation_prompt=True)
        except (AttributeError, ValueError, TypeError) as exc:
            raise ToolCapabilityError(f"当前 {type(self).__name__} tokenizer 缺少可用的工具聊天模板") from exc

    def _parse_tool_output(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        # 延迟导入避免循环:tool_format 需要从本模块导入 ToolOutputError。
        from ..tool_format import parse_hermes_tool_calls

        if self._tool_parser != "hermes_json":
            raise ToolCapabilityError(f"{type(self).__name__} 工具调用需要配置 load.tool_parser=hermes_json")
        content, calls = parse_hermes_tool_calls(text)
        for token in self._STRIP_TOKENS:
            content = content.replace(token, "")
        return content.strip(), calls

    def _build_sampling(self, max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                        repetition_penalty: float, stop_sequences: list[str] | None,
                        extra: dict[str, Any]) -> dict[str, Any]:
        """组装采样参数字典,通用键按 ``_SAMPLING_KEY_MAP`` 改名,并附加
        ``_EXTRA_SAMPLING_KEYS`` 里已知的可选采样字段(从 extra 按需摘取)。"""
        sampling = {"max_tokens": max_new_tokens, "temperature": max(temperature, 0.01),
                   "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty,
                   "stop": stop_sequences}
        sampling.update({key: extra[key] for key in self._EXTRA_SAMPLING_KEYS if key in extra})
        for old_key, new_key in self._SAMPLING_KEY_MAP.items():
            if old_key in sampling:
                sampling[new_key] = sampling.pop(old_key)
        return sampling

    @staticmethod
    def _finish_reason(tool_calls: list[dict[str, Any]], tokens_generated: int,
                       max_new_tokens: int) -> str:
        if tool_calls:
            return "tool_calls"
        return "length" if tokens_generated >= max_new_tokens else "stop"
