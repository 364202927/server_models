from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any


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
    def _canon_args(args: Any) -> str:
        """工具参数规范化（字符串/字典等价），保证两侧指纹一致。"""
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return args
        return json.dumps(args, ensure_ascii=False, sort_keys=True)

    @classmethod
    def memo_key(cls, prev: dict[str, Any] | None, msg: dict[str, Any]) -> str:
        """assistant 消息指纹 = 触发它的上一条消息 + 客户端能原样回传的部分。
        有 tool_calls 时只看调用（伴随文本不会回传），否则看去 think 后的正文。"""
        calls = msg.get("tool_calls") or []
        if calls:
            body = json.dumps(
                [[tc.get("function", {}).get("name"), cls._canon_args(tc.get("function", {}).get("arguments"))] for tc in calls],
                ensure_ascii=False,
            )
        else:
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            text = content or ""
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            body = text.strip()
        prev_part = json.dumps(prev, ensure_ascii=False, sort_keys=True, default=str) if prev else ""
        return hashlib.md5(f"{prev_part}\x00{body}".encode("utf-8")).hexdigest()

    @staticmethod
    def split_raw(raw_text: str) -> dict[str, str]:
        """引擎原文拆成模板所需的两段：think 正文（reasoning_content）与 </think> 之后的文字（去掉工具调用块）。"""
        head, sep, tail = raw_text.partition("</think>")
        if not sep:
            return {"reasoning": "", "content": ""}
        reasoning = head.strip().removeprefix("<think>").strip()
        tail = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", tail, flags=re.DOTALL)
        tail = re.sub(r"<function=.*?(?:</function>|$)", "", tail, flags=re.DOTALL)
        return {"reasoning": reasoning, "content": tail.strip()}

    @classmethod
    def restore_raw_assistant(
        cls, messages: list[dict[str, Any]], memo: dict[str, dict[str, str]]
    ) -> tuple[list[dict[str, Any]], int, int]:
        """客户端丢掉 think 的 assistant 消息，按指纹从记忆库补回 reasoning_content，
        使模板渲染出的历史与 KV 里的原文逐 token 一致。返回 (消息, 还原条数, 未命中条数)。"""
        restored = missed = 0
        out = list(messages)
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant" or msg.get("reasoning_content"):
                continue
            hit = memo.get(cls.memo_key(messages[i - 1] if i else None, msg))
            if hit is None:
                missed += 1
                continue
            item = dict(msg, reasoning_content=hit["reasoning"])
            if msg.get("tool_calls") and not item.get("content"):
                item["content"] = hit["content"]  # 客户端不会回传工具调用前的伴随文字
            out[i] = item
            restored += 1
        return out, restored, missed

    @staticmethod
    def clean_think_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """content 内联的 <think>…</think> 拆到 reasoning_content（Qwen 模板只认该字段）；
        不丢弃 think，否则渲染出的历史与 KV 分叉，混合架构整段重算。"""
        cleaned = []
        for item in messages:
            msg = dict(item)
            content = msg.get("content")
            if msg.get("role") == "assistant" and isinstance(content, str):
                if "</think>" in content:
                    head, _, content = content.partition("</think>")
                    msg.setdefault("reasoning_content", head.strip().removeprefix("<think>").strip())
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