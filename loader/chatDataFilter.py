from __future__ import annotations

import json
import re
import uuid
from typing import Any

GENERATION_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
    "max_tokens", "max_new_tokens", "stop_sequences", "system_prompt", "seed",
    "logit_bias", "frequency_penalty", "presence_penalty", "repeat_last_n",
    "tfs_z", "mirostat", "mirostat_eta", "mirostat_tau", "regex", "json_schema",
})


def flatten_messages(messages: list[dict[str, Any]], system_prompt: str = "") -> str:
    def _text(turn: dict[str, Any]) -> str:
        role = turn.get("role", "user")
        content = turn.get("content")
        
        # 兼容工具执行结果回传
        if role == "tool":
            return f"[Tool Result]: {content or ''}"

        parts = []
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            parts.append("".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"))

        # 如果包含历史工具调用，平铺为自然语言文本防止纯文本补全变空
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


def normalize_generation_params(*sources: dict[str, Any], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    alias_map = {"stop": "stop_sequences", "repeat_penalty": "repetition_penalty", **(aliases or {})}
    values: dict[str, Any] = {}
    for source in sources:
        for k, v in sorted(source.items(), key=lambda kv: kv[0] in alias_map):
            if v is not None:
                values.setdefault(alias_map.get(k, k), v)
    return {k: v for k, v in values.items() if k in GENERATION_KEYS}


class chatDataFilter:
    """聊天上下文数据过滤器：负责生命周期清洗、多模态降级、工具参数结构化与生成后状态校正。"""

    @staticmethod
    def repair_think_tags(text: str) -> str:
        """补全被截断或缺失的 <think> 开始标签。"""
        if "</think>" in text and not text.lstrip().startswith("<think>"):
            return "<think>\n" + text
        return text

    @staticmethod
    def clean_think_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从历史消息中剥离旧的思考过程与残留空标签。"""
        cleaned = []
        for item in messages:
            msg = dict(item)
            content = msg.get("content")
            if msg.get("role") == "assistant" and isinstance(content, str):
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                # 清除历史 assistant 消息中意外遗留的空 tool_call 占位符
                content = re.sub(r"<tool_call>\s*</tool_call>", "", content, flags=re.DOTALL)
                msg["content"] = content.strip()
            cleaned.append(msg)
        return cleaned

    @classmethod
    def sanitize_multimodal(cls, messages: list[dict[str, Any]], supported_modalities: set[str]) -> list[dict[str, Any]]:
        """当模型不支持图片/视频等模态时，将复杂的 base64/二进制段替换为紧凑占位符。"""
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
                    part_type = part.get("type", "text")
                    if part_type == "text":
                        new_content.append(part)
                    elif part_type == "image_url" and "image" not in supported_modalities:
                        new_content.append({"type": "text", "text": "[图片数据已省略]"})
                    elif part_type == "video_url" and "video" not in supported_modalities:
                        new_content.append({"type": "text", "text": "[视频数据已省略]"})
                    elif part_type in supported_modalities:
                        new_content.append(part)
                item["content"] = new_content
            cleaned.append(item)
        return cleaned

    @classmethod
    def normalize_tool_call_mappings(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """治理 Jinja2 模板异常（TypeError: Can only get item pairs from a mapping）。
        确保传入模板的历史 tool_calls.function.arguments 严格为 mapping/dict 对象。
        """
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

    @staticmethod
    def extract_invoked_tool_names(messages: list[dict[str, Any]], assistant_text: str = "") -> set[str]:
        """提取实际被调用的工具名称（兼容 OpenAI、Qwen 与 Hermes 格式）。"""
        invoked_tools = set()

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    name = call.get("function", {}).get("name")
                    if name:
                        invoked_tools.add(name)

        if assistant_text:
            xml_matches = re.findall(r"<function=([^>]+)>", assistant_text)
            invoked_tools.update(m.strip() for m in xml_matches)

            for json_str in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", assistant_text, re.DOTALL):
                try:
                    data = json.loads(json_str)
                    if "name" in data:
                        invoked_tools.add(data["name"])
                except Exception:
                    pass

        return invoked_tools

    @staticmethod
    def is_dialog_turn_complete(finish_reason: str, pending_tool_calls: list[Any]) -> bool:
        """判定本轮对话链路是否彻底完结。"""
        return not (finish_reason == "tool_calls" or bool(pending_tool_calls))

    @classmethod
    def prune_tools_for_storage(cls, all_tools: list[dict[str, Any]], invoked_tool_names: set[str]) -> list[dict[str, Any]]:
        """Pruning Filter：仅保留本轮实际命中的工具，未调用的全部从持久化历史中剔除。"""
        if not invoked_tool_names or not all_tools:
            return []
        return [
            tool for tool in all_tools
            if tool.get("function", {}).get("name") in invoked_tool_names
        ]

    @classmethod
    def preprocess_messages(
        cls,
        messages: list[dict[str, Any]] | None,
        supported_modalities: set[str],
        default_system: str = "",
        user_prompt: str = "",
        think_instruction: str = "",
    ) -> list[dict[str, Any]]:
        """全量前置清洗：剥离历史 think、清洗残留空标签、多模态降级、arguments 映射规整、注入 system。"""
        current_messages = [dict(m) for m in messages] if messages else [{"role": "user", "content": user_prompt}]
        cleaned = cls.clean_think_history(current_messages)
        cleaned = cls.sanitize_multimodal(cleaned, supported_modalities)
        cleaned = cls.normalize_tool_call_mappings(cleaned)

        sys_parts = [t for t in (default_system, think_instruction) if t]
        if sys_parts:
            prefix = "\n\n".join(sys_parts)
            if cleaned and cleaned[0].get("role") == "system":
                cleaned[0]["content"] = f"{prefix}\n\n{cleaned[0].get('content', '')}".strip()
            else:
                cleaned.insert(0, {"role": "system", "content": prefix})

        return cleaned

    @classmethod
    def postprocess_result(
        cls,
        raw_text: str,
        detected_calls: list[dict[str, Any]],
        finish_reason: str,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """后置清洗与状态纠偏：
        1. 兜底正则提取漏抓的工具调用。
        2. 清理残余空标签（<tool_call></tool_call>）。
        3. 只要存在工具调用，强制修正 finish_reason='tool_calls' 并净化 content。
        """
        tool_calls = list(detected_calls or [])
        clean_text = raw_text

        if "</think>" in clean_text and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text

        # 原生引擎未解析到时进行正则抓取
        if not tool_calls:
            # 1. Hermes 格式匹配: <tool_call> {JSON} </tool_call>
            hermes_matches = re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", clean_text, re.DOTALL)
            for match in hermes_matches:
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

            # 2. Qwen XML 格式匹配: <function=name><parameter=k>v</parameter></function>
            qwen_matches = re.finditer(r"<function=([^>]+)>(.*?)(?:</function>|(?=<function=)|$)", clean_text, re.DOTALL)
            for match in qwen_matches:
                fn_name = match.group(1).strip()
                body = match.group(2)
                args = {}
                for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL):
                    args[pm.group(1).strip()] = pm.group(2).strip()
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

        # 彻底移除残留的空壳标签与裸标签
        clean_text = re.sub(r"<tool_call>\s*</tool_call>", "", clean_text, flags=re.DOTALL)
        clean_text = re.sub(r"</?tool_call>", "", clean_text)

        # 状态纠正
        if tool_calls:
            finish_reason = "tool_calls"
            if "</think>" in clean_text:
                clean_text = clean_text.split("</think>", 1)[1].strip()
            elif clean_text.startswith("<think>"):
                clean_text = ""

        return clean_text.strip(), tool_calls, finish_reason