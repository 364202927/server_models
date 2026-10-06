from __future__ import annotations

import json,re
from typing import Any

GENERATION_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
    "max_tokens", "max_new_tokens", "stop_sequences", "system_prompt", "seed",
    "logit_bias", "frequency_penalty", "presence_penalty", "repeat_last_n",
    "tfs_z", "mirostat", "mirostat_eta", "mirostat_tau", "regex", "json_schema",
})


def flatten_messages(messages: list[dict[str, Any]], system_prompt: str = "") -> str:
    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content or "")

    turns = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
    return "\n".join(f"{turn.get('role', 'user')}: {_text(turn.get('content'))}" for turn in turns)


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
    """聊天上下文数据过滤器：负责前置清洗、多模态降级、工具修剪与生成后文本修复。"""

    @staticmethod
    def repair_think_tags(text: str) -> str:
        """补全被截断或缺失的 <think> 开始标签。"""
        if "</think>" in text and not text.lstrip().startswith("<think>"):
            return "<think>\n" + text
        return text

    @staticmethod
    def clean_think_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从历史消息中剥离旧的思考过程，避免污染多轮对话。"""
        cleaned = []
        for item in messages:
            msg = dict(item)
            content = msg.get("content")
            if msg.get("role") == "assistant" and isinstance(content, str) and "</think>" in content:
                msg["content"] = content.split("</think>", 1)[1].strip()
            cleaned.append(msg)
        return cleaned

    @classmethod
    def sanitize_multimodal(cls,messages: list[dict[str, Any]],supported_modalities: set[str]) -> list[dict[str, Any]]:
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
        if finish_reason == "tool_calls" or bool(pending_tool_calls):
            return False
        return True

    @classmethod
    def prune_tools_for_storage(cls,all_tools: list[dict[str, Any]],invoked_tool_names: set[str]) -> list[dict[str, Any]]:
        """仅保留本轮实际命中的工具，未调用的全部从持久化历史中剔除。"""
        if not invoked_tool_names or not all_tools:
            return []
        return [
            tool for tool in all_tools
            if tool.get("function", {}).get("name") in invoked_tool_names
        ]

    @classmethod
    def preprocess_messages(cls,messages: list[dict[str, Any]] | None,supported_modalities: set[str],default_system: str = "",user_prompt: str = "",think_instruction: str = "",) -> list[dict[str, Any]]:
        """进入推理引擎前的全流程清洗：剥离旧 think、多模态脱敏、注入 system。"""
        current_messages = [dict(m) for m in messages] if messages else [{"role": "user", "content": user_prompt}]
        cleaned = cls.clean_think_history(current_messages)
        cleaned = cls.sanitize_multimodal(cleaned, supported_modalities)

        sys_parts = [t for t in (default_system, think_instruction) if t]
        if sys_parts:
            prefix = "\n\n".join(sys_parts)
            if cleaned and cleaned[0].get("role") == "system":
                cleaned[0]["content"] = f"{prefix}\n\n{cleaned[0].get('content', '')}".strip()
            else:
                cleaned.insert(0, {"role": "system", "content": prefix})

        return cleaned