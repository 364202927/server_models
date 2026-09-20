"""
tool_search.py
工具数量较多时的按需检索机制。

思路（对应 Anthropic Tool Search Tool / 生产级 agent 常见的分层路由）：只把常驻的
__search_tools__ 元工具的完整定义交给模型，其余工具只在 system 段给一份
"名称 + 一句话描述"的目录；模型需要用到目录里的工具时，自己调用
__search_tools__ 按关键词换取完整参数定义，再正式调用。

__search_tools__ 完全由 MsgHandler 在内部拦截处理，从不出现在返回给客户端的
tool_calls 里，对客户端和各个 Loader 都透明——Loader 侧不需要做任何改动。

检索用的是不依赖第三方库的简化版 BM25：工具目录通常只有几十条短文本，
这个规模下朴素实现和引入 numpy/rank_bm25 之类的库相比没有性能差异，
但少一个依赖。
"""

import math
import re
from typing import Any

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