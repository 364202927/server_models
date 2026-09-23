"""
tool_format.py
工具调用相关格式化、模型快照缓存、推理框架选择——三块合并到一个模块。

## 工具数量较多时的按需检索机制

思路（对应 Anthropic Tool Search Tool / 生产级 agent 常见的分层路由）：只把常驻的
__search_tools__ 元工具的完整定义交给模型，其余工具只在 system 段给一份
"名称 + 一句话描述"的目录；模型需要用到目录里的工具时，自己调用
__search_tools__ 按关键词换取完整参数定义，再正式调用。

__search_tools__ 完全由 MsgHandler 在内部拦截处理，从不出现在返回给客户端的
tool_calls 里，对客户端和各个 Loader 都透明——Loader 侧不需要做任何改动。

检索用的是不依赖第三方库的简化版 BM25：工具目录通常只有几十条短文本，
这个规模下朴素实现和引入 numpy/rank_bm25 之类的库相比没有性能差异，
但少一个依赖。

## 模型状态快照 / 推理框架选择
"""


import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

from ..utils.common import aContainB, error, require
from .llmFramework.baseInference import ToolOutputError, baseInference
from .model_spec import ModelSpec

SEARCH_TOOL_NAME = "__search_tools__"

SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": ("按关键词在完整工具目录中查找工具，返回匹配工具的完整参数定义。"
                        "本轮如果需要使用目录里提到、但未直接给出参数的工具，先调用这个函数。"),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "描述需要的能力，可以是工具名或功能关键词"},
            },
            "required": ["query"],
        },
    },
}

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
 
TOOL_SYSTEM_TEMPLATE = (
    "你可以调用以下工具。需要调用时，在回复中输出一个或多个 XML 标签，"
    "每个标签内是一个 JSON 对象，形如 "
    '<tool_call>{{"name": "工具名", "arguments": {{...}}}}</tool_call>。'
    "不需要调用工具时，正常回答即可。\n\n可用工具：\n{schemas}"
)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")


def prepare_tool_view(tools: list[dict[str, Any]], tool_choice: Any,
                      threshold: int) -> tuple[list[dict[str, Any]], str | None]:
    """决定这一轮该把哪些工具的完整定义交给模型。

    三种情况直接原样透传、不引入检索（返回 (tools, None)）：
    - 工具数没超过阈值——本来就不贵，绕一轮检索反而多一次往返；
    - tool_choice 指定了具体某个函数——已经知道要调用谁，没有"搜索"的必要；
    - tool_choice="none"——本轮不允许调用任何工具，展示目录没有意义。
    否则只把 __search_tools__ 交给模型，其余工具收进一份精简目录文本返回，
    调用方负责把这段文本注入 system 消息。
    """
    if len(tools) <= threshold or tool_choice == "none" or isinstance(tool_choice, dict):
        return tools, None
    catalog = "\n".join(f"- {tool['function']['name']}: {_first_sentence(tool['function'].get('description', ''))}"
                        for tool in tools)
    notice = (
        "除下面直接给出完整定义的工具外，还有以下工具可用（仅列出用途，未给出参数）：\n"
        f"{catalog}\n\n"
        f"需要使用其中某个工具时，先调用 {SEARCH_TOOL_NAME} 按关键词查找，"
        "拿到完整参数定义后再正式调用该工具；不要凭空编造参数直接调用未给出定义的工具。"
    )
    return [SEARCH_TOOL_SPEC], notice


def search_tools(query: str, tools: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """在完整工具目录里用 BM25 给 query 打分，返回最相关的最多 top_k 个工具定义（不含零分工具）。"""
    corpus = [_tool_tokens(tool) for tool in tools]
    scores = _bm25_scores(_tokenize(query), corpus)
    ranked = sorted(zip(scores, tools), key=lambda pair: pair[0], reverse=True)
    return [tool for score, tool in ranked[:top_k] if score > 0]


def merge_tools(existing: list[dict[str, Any]], found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把新找到的工具追加进当前已对模型可见的工具列表，按函数名去重。"""
    names = {tool["function"]["name"] for tool in existing}
    return existing + [tool for tool in found if tool["function"]["name"] not in names]


def _first_sentence(text: str, limit: int = 60) -> str:
    """取描述的第一行，超长再截断，控制目录文本的体积。"""
    line = text.strip().split("\n", 1)[0]
    return line if len(line) <= limit else line[:limit].rstrip() + "…"


def _tokenize(text: str) -> list[str]:
    """粗粒度分词：按字母数字/中文切分，并把下划线拆开，让 snake_case 工具名和自然语言 query 命中同样的词。"""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text.replace("_", " "))]


def _tool_tokens(tool: dict[str, Any]) -> list[str]:
    function = tool.get("function", {})
    return _tokenize(function.get("name", "")) + _tokenize(function.get("description", ""))


def _bm25_scores(query_tokens: list[str], corpus: list[list[str]],
                 k1: float = 1.5, b: float = 0.75) -> list[float]:
    """极简 BM25 实现，不依赖第三方库；工具目录通常只有几十条短文本，性能足够。"""
    doc_count = len(corpus)
    if not doc_count:
        return []
    avg_len = sum(len(doc) for doc in corpus) / doc_count
    doc_freq: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    def idf(term: str) -> float:
        hits = doc_freq.get(term, 0)
        return math.log((doc_count - hits + 0.5) / (hits + 0.5) + 1)

    scores = []
    for doc in corpus:
        length = len(doc) or 1
        score = sum(idf(term) * (doc.count(term) * (k1 + 1))
                   / (doc.count(term) + k1 * (1 - b + b * length / avg_len))
                   for term in set(query_tokens) if term in doc)
        scores.append(score)
    return scores

def render_tool_system_prompt(tools: list[dict[str, Any]],
                              tool_choice: Any) -> tuple[str, list[dict[str, Any]]]:
    """按 tool_choice 过滤/强制工具，渲染成注入 system 段的指令文本。

    返回 (指令文本, 过滤后的 tools)；调用方应在 tool_choice == "none" 时跳过整个工具流程，
    不要调用这个函数（此函数不处理 "none"，因为那种情况下不该渲染任何工具信息）。
    """
    directive = ""
    if isinstance(tool_choice, dict):
        name = tool_choice["function"]["name"]
        tools = [tool for tool in tools if tool["function"]["name"] == name]
        directive = f"你必须调用工具 {name}。"
    elif tool_choice == "required":
        directive = "你必须调用至少一个可用工具。"
    schemas = "\n".join(json.dumps(tool["function"], ensure_ascii=False) for tool in tools)
    instruction = TOOL_SYSTEM_TEMPLATE.format(schemas=schemas)
    return (f"{instruction}\n\n{directive}" if directive else instruction), tools


def parse_hermes_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """解析形如 ``<tool_call>{...}</tool_call>`` 的模型输出。

    返回 (去除标签后的正文, 规范化为 OpenAI tool_calls 结构的调用列表)。
    """
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if "<tool_call>" in text and not matches:
        raise ToolOutputError("模型返回了未闭合的 <tool_call>")
    calls = []
    for match in matches:
        try:
            value = json.loads(match.group(1))
            function = value.get("function", value)
            name = function["name"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ToolOutputError("模型返回了无效的 Hermes 工具调用") from exc
        calls.append({"id": str(value.get("id") or f"call_{uuid.uuid4().hex}"),
                      "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(arguments, ensure_ascii=False,
                                                            separators=(",", ":"))}})
    return TOOL_CALL_PATTERN.sub("", text).strip(), calls


# ---------------------------------------------------------------------------
# 模型状态快照（原 cache.py）：快照只记录模型自身的身份和运行时状态，
# 缓存策略在 defaults 里配置。
# ---------------------------------------------------------------------------


def snapshot_path(root: str | Path, model_id: str) -> Path:
    return Path(root) / "runtime" / "snapshots" / f"{model_id}.json"


def save_snapshot(root: str | Path, spec: ModelSpec, **runtime: Any) -> Path:
    path = snapshot_path(root, spec.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": spec.model_id, "path": spec.path,
               "dtype": spec.load.dtype, "runtime": runtime}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_snapshot(root: str | Path, model_id: str) -> dict[str, Any] | None:
    path = snapshot_path(root, model_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# 推理框架选择与创建（原 engine_select.py）：按 spec.load.engine 显式选择；
# 未配置且路径后缀也没有默认映射时，打印错误并返回 None（不抛异常）——
# 调用方（models_mgr._load）据此把模型标记为未加载，而不是让整条加载链路
# 带着一个不存在的推理框架往下走。
#
# 不经过独立的 factory 模块——直接用 utils.common.require() 按模块路径
# 最后一段类名动态创建子类（vllm/sglang/llama），新增推理框架只需在
# llmFramework/ 下加一个同名模块，不需要改这里的分支。
# ---------------------------------------------------------------------------

# __package__ 就是本模块所在的 "loader" 包的完整点分路径(例如 "ai.loader" 或
# 直接执行时的 "server_models.loader")；用它拼出 llmFramework 子模块路径，
# 而不是硬编码顶层包名——repo 顶层目录名在不同部署方式下并不总是 "ai"
# (main.py 也是这样按 __package__ 二选一处理导入的)。
_ENGINE_MODULE = f"{__package__}.llmFramework.{{engine}}"
# 路径名里出现这些标记时默认用 vllm(量化格式对 llama.cpp 没有意义)。
_VLLM_PATH_HINTS = ("fp8", "awq", "gptq")


def detect_engine(path) -> str | None:
    """按文件/目录后缀推断默认推理框架:

    - ``.gguf`` 文件或含 ``*.gguf`` 的目录 -> llama
    - 含 ``config.json`` + ``*.safetensors`` 的标准 HF 目录 -> vllm
    - 路径名包含 fp8/awq/gptq -> vllm
    - 其余情况没有默认值,返回 None(调用方按此打印错误,不再继续加载)。
    """
    if path.suffix.lower() == ".gguf":
        return "llama"
    if path.is_dir():
        if any(path.glob("*.gguf")):
            return "llama"
        if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
            return "vllm"
    if aContainB(path.name.lower(), _VLLM_PATH_HINTS):
        return "vllm"
    return None


def create_loader(spec: ModelSpec) -> baseInference | None:
    """按 ``spec.load.engine`` 或路径后缀选择并创建推理框架实例。

    没有配置 engine 且路径也推断不出默认值时,打印错误并返回 None——
    不抛异常。"""
    engine = spec.engine or detect_engine(spec.path_obj)
    if not engine:
        error("无法确定推理框架，请在 models.json 配置 load.engine：model=", spec.model_id, " path=", spec.path)
        return None
    cls = require(_ENGINE_MODULE.format(engine=engine))
    return cls() if cls else None