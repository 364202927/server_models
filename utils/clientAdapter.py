from __future__ import annotations

import json
import uuid
from typing import Any


class clientAdapter:
    """网关双向转换器：在客户端特有格式与内部标准 OpenAI 协议之间进行适配。"""

    @classmethod
    def inbound_to_openai(cls, payload: dict[str, Any], client_type: str = "openai") -> dict[str, Any]:
        if client_type == "openai":
            return payload

        if client_type == "claude":
            openai_messages = []
            if payload.get("system"):
                openai_messages.append({"role": "system", "content": payload["system"]})

            for msg in payload.get("messages", []):
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "tool_result":
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": part.get("tool_use_id", ""),
                                "content": part.get("content", "")
                            })
                    if text_parts:
                        openai_messages.append({"role": role, "content": "\n".join(text_parts)})
                else:
                    openai_messages.append({"role": role, "content": str(content or "")})

            openai_tools = []
            for t in payload.get("tools", []):
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {})
                    }
                })

            return {
                "model": payload.get("model", ""),
                "messages": openai_messages,
                "tools": openai_tools or None,
                "temperature": payload.get("temperature", 0.3),
                "max_tokens": payload.get("max_tokens", 4096),
                "stream": payload.get("stream", False),
            }

        return payload

    @classmethod
    def outbound_from_openai(cls, openai_resp: dict[str, Any], client_type: str = "openai") -> dict[str, Any]:
        if client_type == "openai":
            return openai_resp

        if client_type == "claude":
            tool_calls = openai_resp.get("tool_calls") or []
            content_blocks = []
            raw_text = openai_resp.get("response", "")

            if raw_text:
                content_blocks.append({"type": "text", "text": raw_text})

            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                    "name": fn.get("name"),
                    "input": json.loads(args) if isinstance(args, str) else args
                })

            stop_reason = "tool_use" if tool_calls else ("max_tokens" if openai_resp.get("finish_reason") == "length" else "end_turn")

            return {
                "id": openai_resp.get("request_id"),
                "type": "message",
                "role": "assistant",
                "model": openai_resp.get("model"),
                "content": content_blocks,
                "stop_reason": stop_reason,
                "usage": {
                    "input_tokens": openai_resp.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": openai_resp.get("usage", {}).get("completion_tokens", 0),
                }
            }

        return openai_resp