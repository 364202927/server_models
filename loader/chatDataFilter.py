from __future__ import annotations

import json
import re
import uuid
from typing import Any


def flatten_messages(messages: list[dict[str, Any]], system_prompt: str = "") -> str:
    def _text(turn: dict[str, Any]) -> str:
        role = turn.get("role", "user")
        content = turn.get("content")
        if role == "tool":
            return f"[Tool Result]: {content or ''}"
        parts = []
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            parts.append("".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"))
        if turn.get("tool_calls"):
            for tc in turn["tool_calls"]:
                fn = tc.get("function", {})
                parts.append(f"[Call Tool: {fn.get('name')}({fn.get('arguments', '')})]")
        return "\n".join(parts)

    turns = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
    return "\n".join(f"{turn.get('role', 'user')}: {_text(turn)}" for turn in turns)


def resolve_think_level(*, think: Any = None, reasoning_effort: Any = None) -> int:
    levels = {"none": 0, "low": 1, "medium": 3, "high": 5}
    if think is not None:
        return max(0, min(5, int(think)))
    if isinstance(reasoning_effort, int):
        return max(0, min(5, reasoning_effort))
    return levels.get(str(reasoning_effort).lower(), 0)


class chatDataFilter:
    """聊天数据过滤器：清洗与状态校正。"""

    @staticmethod
    def repair_think_tags(text: str) -> str:
        if "</think>" in text and not text.lstrip().startswith("<think>"):
            return "<think>\n" + text
        return text

    @staticmethod
    def clean_think_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        for item in messages:
            msg = dict(item)
            content = msg.get("content")
            if msg.get("role") == "assistant" and isinstance(content, str):
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                content = re.sub(r"<tool_call>\s*</tool_call>", "", content, flags=re.DOTALL)
                msg["content"] = content.strip()
            cleaned.append(msg)
        return cleaned

    @classmethod
    def sanitize_multimodal(cls, messages: list[dict[str, Any]], supported_modalities: set[str]) -> list[dict[str, Any]]:
        if "image" in supported_modalities and "video" in supported_modalities:
            return messages
        cleaned = []
        for msg in messages:
            item = dict(msg)
            content = item.get("content")
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    pt = part.get("type", "text")
                    if pt == "text":
                        new_content.append(part)
                    elif pt == "image_url" and "image" not in supported_modalities:
                        new_content.append({"type": "text", "text": "[图片数据已省略]"})
                    elif pt == "video_url" and "video" not in supported_modalities:
                        new_content.append({"type": "text", "text": "[视频数据已省略]"})
                    elif pt in supported_modalities:
                        new_content.append(part)
                item["content"] = new_content
            cleaned.append(item)
        return cleaned

    @classmethod
    def normalize_tool_call_mappings(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for msg in messages:
            item = dict(msg)
            if item.get("role") == "assistant" and item.get("tool_calls"):
                clean_tcs = []
                for tc in item["tool_calls"]:
                    tc_dict = dict(tc)
                    fn = dict(tc_dict.get("function", {}))
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args) if args.strip() else {}
                        except Exception:
                            fn["arguments"] = {}
                    elif not isinstance(args, dict):
                        fn["arguments"] = {}
                    tc_dict["function"] = fn
                    clean_tcs.append(tc_dict)
                item["tool_calls"] = clean_tcs
            normalized.append(item)
        return normalized

    @classmethod
    def preprocess_messages(
        cls,
        messages: list[dict[str, Any]],
        supported_modalities: set[str],
        system_instruction: str = "",
    ) -> list[dict[str, Any]]:
        """收敛为 3 个入参。"""
        current_messages = [dict(m) for m in messages]
        cleaned = cls.clean_think_history(current_messages)
        cleaned = cls.sanitize_multimodal(cleaned, supported_modalities)
        cleaned = cls.normalize_tool_call_mappings(cleaned)

        if system_instruction:
            if cleaned and cleaned[0].get("role") == "system":
                cleaned[0]["content"] = f"{system_instruction}\n\n{cleaned[0].get('content', '')}".strip()
            else:
                cleaned.insert(0, {"role": "system", "content": system_instruction})
        return cleaned

    @classmethod
    def postprocess_result(
        cls,
        raw_text: str,
        detected_calls: list[dict[str, Any]],
        finish_reason: str,
    ) -> tuple[str, list[dict[str, Any]], str]:
        tool_calls = list(detected_calls or [])
        clean_text = raw_text

        if "</think>" in clean_text and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text

        if not tool_calls:
            # Hermes 格式匹配
            for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", clean_text, re.DOTALL):
                try:
                    data = json.loads(match.group(1))
                    if "name" in data:
                        args = data.get("arguments", {})
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": data["name"],
                                "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                            },
                        })
                except Exception:
                    pass
            clean_text = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", clean_text, flags=re.DOTALL)

            # Qwen XML 格式匹配
            for match in re.finditer(r"<function=([^>]+)>(.*?)(?:</function>|(?=<function=)|$)", clean_text, re.DOTALL):
                fn_name = match.group(1).strip()
                body = match.group(2)
                args = {pm.group(1).strip(): pm.group(2).strip() for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL)}
                if fn_name:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })
            clean_text = re.sub(r"<function=[^>]+>.*?(?:</function>|$)", "", clean_text, flags=re.DOTALL)

        clean_text = re.sub(r"<tool_call>\s*</tool_call>", "", clean_text, flags=re.DOTALL)
        clean_text = re.sub(r"</?tool_call>", "", clean_text)

        if tool_calls:
            finish_reason = "tool_calls"
            if "</think>" in clean_text:
                clean_text = clean_text.split("</think>", 1)[1].strip()
            elif clean_text.startswith("<think>"):
                clean_text = ""

        return clean_text.strip(), tool_calls, finish_reason