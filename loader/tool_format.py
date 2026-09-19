"""
tool_format.py
Hermes 风格 ``<tool_call>`` 标签的渲染与解析。

供原生使用该格式的后端共用：
- HFLoader：模型自带 chat template 支持 tools= 参数时，仍用这里的正则解析输出。
- GGUFLoader：未配置 ``chat_format=chatml-function-calling`` 时的兜底路径，
  既用这里的模板把工具 schema 写进 system 段，也用这里的正则解析输出。
"""

import json
import re
import uuid
from typing import Any

from .base import ToolOutputError

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

TOOL_SYSTEM_TEMPLATE = (
    "你可以调用以下工具。需要调用时，在回复中输出一个或多个 XML 标签，"
    "每个标签内是一个 JSON 对象，形如 "
    '<tool_call>{{"name": "工具名", "arguments": {{...}}}}</tool_call>。'
    "不需要调用工具时，正常回答即可。\n\n可用工具：\n{schemas}"
)


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
            if not isinstance(value, dict):
                raise ValueError("tool call must be an object")
            function = value.get("function", value)
            name = function["name"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ToolOutputError("模型返回了无效的 Hermes 工具调用") from exc
        calls.append({"id": str(value.get("id") or f"call_{uuid.uuid4().hex}"),
                      "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(arguments, ensure_ascii=False,
                                                            separators=(",", ":"))}})
    return TOOL_CALL_PATTERN.sub("", text).strip(), calls
