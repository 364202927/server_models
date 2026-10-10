from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Iterator


# ----------------------------------------------------------------------------
# 通用小工具（无状态，多处复用）
# ----------------------------------------------------------------------------
def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _as_json_str(value: Any) -> str:
    """工具参数统一为 JSON 字符串：已是字符串则原样返回，否则序列化。"""
    return value if isinstance(value, str) else _dumps(value)


def _loads_args(raw: Any) -> Any:
    """工具参数反序列化：非字符串原样返回，解析失败时包装为 {"raw": ...}。"""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _sse(data: Any, event: str | None = None) -> str:
    """序列化为一帧 SSE；event 为空时只输出 data 行（OpenAI 风格）。"""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {_dumps(data)}\n\n"


def _split(text: str, size: int | None) -> Iterator[str]:
    """按 size 切片；size 为空则整体一片，空串不产出任何片段。"""
    step = size or len(text) or 1
    return (text[i : i + step] for i in range(0, len(text), step))


def _parse_response(resp: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    """从 OpenAI 响应中取出 (文本, 工具调用列表, finish_reason)，兼容 response/tool_calls 顶层字段。"""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or resp.get("response") or ""
    tool_calls = message.get("tool_calls") or resp.get("tool_calls") or []
    return content, tool_calls, choice.get("finish_reason") or "stop"


# ----------------------------------------------------------------------------
# Claude 协议转换（单条构造器）
# ----------------------------------------------------------------------------
def _claude_stop_reason(tool_calls: list[dict[str, Any]], finish_reason: str) -> str:
    if tool_calls:
        return "tool_use"
    return "max_tokens" if finish_reason == "length" else "end_turn"


def _claude_message(
    resp: dict[str, Any], content: list[dict[str, Any]], stop_reason: str | None, output_tokens: int
) -> dict[str, Any]:
    """构造 Anthropic message 对象；非流式返回体与流式 message_start 共用同一结构。"""
    return {
        "id": resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": resp.get("model", ""),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": output_tokens,
        },
    }


# Claude Code 每次请求都会在 system 首块带一个变化的计费头，会使前缀 KV 缓存全部失效
_BILLING_HEADER = re.compile(r"^x-anthropic-billing-header:[^\n]*\n?", re.MULTILINE)


def _blocks_to_text(content: Any) -> str:
    """content（字符串 / text 块列表）→ 纯字符串，非 text 块忽略。"""
    if not isinstance(content, list):
        return str(content or "")
    return "\n".join(
        p if isinstance(p, str) else p.get("text", "")
        for p in content
        if isinstance(p, str) or p.get("type") == "text"
    )


def _system_text(system: Any) -> str:
    """system（字符串 / 块列表）→ 字符串，同时剥掉 billing header。"""
    parts = [system] if isinstance(system, str) else (system or [])
    texts = (_BILLING_HEADER.sub("", _blocks_to_text([p])).strip() for p in parts)
    return "\n\n".join(t for t in texts if t)


def _claude_message_to_openai(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """单条 Claude 消息 → 若干 OpenAI 消息：tool_result 各成一条 tool 消息；
    assistant 的 text 与 tool_use 合成一条（按原顺序）；user 文本合并成一条。"""
    role, content = msg.get("role"), msg.get("content")
    if not isinstance(content, list):
        return [{"role": role, "content": str(content or "")}]

    tool_msgs: list[dict[str, Any]] = []
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            texts.append(part.get("text", ""))
        elif kind == "tool_result":
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": part.get("tool_use_id", ""),
                "content": _blocks_to_text(part.get("content")),
            })
        elif kind == "tool_use":
            calls.append({
                "id": part.get("id", ""),
                "type": "function",
                "function": {"name": part.get("name", ""), "arguments": _as_json_str(part.get("input", {}))},
            })

    text = "\n".join(texts)
    if role == "assistant" and calls:
        return tool_msgs + [{"role": role, "content": text or None, "tool_calls": calls}]
    return tool_msgs + ([{"role": role, "content": text}] if texts else [])


class clientAdapter:
    """网关双向转换器：在客户端特有格式与内部标准 OpenAI 协议之间进行适配。"""

    # ------------------------------------------------------------------ 入站
    @staticmethod
    def inbound_to_openai(payload: dict[str, Any], client_type: str = "openai") -> dict[str, Any]:
        if client_type != "claude":
            return payload

        system = _system_text(payload.get("system"))
        messages = [{"role": "system", "content": system}] if system else []
        for msg in payload.get("messages", []):
            messages.extend(_claude_message_to_openai(msg))

        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in payload.get("tools", [])
        ]
        return {
            "model": payload.get("model", ""),
            "messages": messages,
            "tools": tools or None,
            "temperature": payload.get("temperature", 0.3),
            "max_tokens": payload.get("max_tokens", 4096),
            "stream": payload.get("stream", False),
        }

    # ------------------------------------------------------------------ 出站
    @classmethod
    def outbound(
        cls,
        openai_resp: dict[str, Any],
        client_type: str = "openai",
        stream: bool = False,
        chunk_size: int | None = None,
    ) -> Any:
        """统一出站格式转换。非流式/流式均以 OpenAI 协议为基础中间层。"""
        is_claude = client_type == "claude"
        if not stream:
            return cls._claude_response(openai_resp) if is_claude else openai_resp

        chunks = cls._iter_openai_chunks(openai_resp, chunk_size)
        if is_claude:
            return cls._claude_stream(openai_resp, chunks)

        async def openai_sse() -> AsyncGenerator[str, None]:
            async for chunk in chunks:
                yield _sse(chunk)
            yield "data: [DONE]\n\n"

        return openai_sse()

    # ------------------------------------------------- 中间层：OpenAI chunk 流
    @staticmethod
    async def _iter_openai_chunks(
        openai_resp: dict[str, Any],
        chunk_size: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """将完整响应标准化切分为 OpenAI chat.completion.chunk 字典流。"""
        content, tool_calls, finish_reason = _parse_response(openai_resp)
        base = {
            "id": openai_resp.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": openai_resp.get("created", int(time.time())),
            "model": openai_resp.get("model", ""),
        }

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        # 1. 文本增量
        for piece in _split(content, chunk_size):
            yield chunk({"role": "assistant", "content": piece})

        # 2. 工具调用增量：先发头信息（ID 与函数名），再发参数片段
        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            yield chunk({
                "role": "assistant",
                "tool_calls": [{
                    "index": idx,
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": ""},
                }],
            })
            for piece in _split(_as_json_str(fn.get("arguments", "{}")), chunk_size):
                yield chunk({"tool_calls": [{"index": idx, "function": {"arguments": piece}}]})

        # 3. 终结块
        yield chunk({}, finish_reason)

    # ----------------------------------------------------- Claude 协议（出站）
    @staticmethod
    def _claude_response(openai_resp: dict[str, Any]) -> dict[str, Any]:
        """非流式：OpenAI 响应 → Anthropic message。"""
        content, tool_calls, finish_reason = _parse_response(openai_resp)

        blocks: list[dict[str, Any]] = [{"type": "text", "text": content}] if content else []
        for tc in tool_calls:
            fn = tc.get("function", {})
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:8]}",
                "name": fn.get("name"),
                "input": _loads_args(fn.get("arguments", "{}")),
            })

        return _claude_message(
            openai_resp,
            blocks,
            _claude_stop_reason(tool_calls, finish_reason),
            openai_resp.get("usage", {}).get("completion_tokens", 0),
        )

    @staticmethod
    async def _claude_stream(
        openai_resp: dict[str, Any],
        chunks: AsyncGenerator[dict[str, Any], None],
    ) -> AsyncGenerator[str, None]:
        """流式：消费 OpenAI chunk 流，映射为 Anthropic SSE 事件。"""
        _, tool_calls, finish_reason = _parse_response(openai_resp)
        output_tokens = openai_resp.get("usage", {}).get("completion_tokens", 0)

        def emit(name: str, /, **data: Any) -> str:
            return _sse({"type": name, **data}, event=name)

        yield emit("message_start", message=_claude_message(openai_resp, [], None, 1))

        block = 0                          # 当前（或下一个）content block 序号
        text_open = False
        tool_blocks: dict[int, int] = {}   # OpenAI tool index -> Claude block 序号

        async for chunk in chunks:
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})

            if delta.get("content"):
                if not text_open:
                    text_open = True
                    yield emit("content_block_start", index=block, content_block={"type": "text", "text": ""})
                yield emit("content_block_delta", index=block, delta={"type": "text_delta", "text": delta["content"]})

            for tc in delta.get("tool_calls") or []:
                if text_open:  # 切到工具前先闭合文本块
                    yield emit("content_block_stop", index=block)
                    text_open = False
                    block += 1

                fn = tc.get("function", {})
                t_idx = tc.get("index", 0)
                if fn.get("name"):
                    tool_blocks[t_idx] = block
                    yield emit(
                        "content_block_start",
                        index=block,
                        content_block={"type": "tool_use", "id": tc.get("id"), "name": fn["name"], "input": {}},
                    )
                    block += 1
                if fn.get("arguments"):
                    yield emit(
                        "content_block_delta",
                        index=tool_blocks[t_idx],
                        delta={"type": "input_json_delta", "partial_json": fn["arguments"]},
                    )

        # 终结：闭合仍打开的 block，输出 stop_reason
        if text_open:
            yield emit("content_block_stop", index=block)
        for idx in tool_blocks.values():
            yield emit("content_block_stop", index=idx)
        yield emit(
            "message_delta",
            delta={"stop_reason": _claude_stop_reason(tool_calls, finish_reason), "stop_sequence": None},
            usage={"output_tokens": output_tokens},
        )
        yield emit("message_stop")