from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..utils.common import warn as log_warn


def resolve_think_level(*, think: Any = None, reasoning_effort: Any = None) -> int:
    levels = {"none": 0, "low": 1, "medium": 3, "high": 5}
    if think is not None:
        return max(0, min(5, int(think)))
    if isinstance(reasoning_effort, int):
        return max(0, min(5, reasoning_effort))
    return levels.get(str(reasoning_effort).lower(), 0)


# 可以确定"重新生成一次大概率能好"的失败类别；truncated（被 max_tokens 截断）/ unclosed_think 重试通常无益，只记录
RETRYABLE_PROBLEMS = frozenset({"json_invalid", "unclosed_call", "unknown_tool", "missing_required", "empty"})


@dataclass
class parsedResult:
    """模型原文解析结果。problems 非空表示工具调用/输出有问题：(类别, 说明)。"""

    text: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    problems: list[tuple[str, str]] = field(default_factory=list)

    @property
    def retryable(self) -> bool:
        return any(code in RETRYABLE_PROBLEMS for code, _ in self.problems)


class chatDataFilter:
    """聊天数据过滤器：清洗与状态校正。"""

    _HERMES_CALL = re.compile(r"<tool_call>(?!\s*<function=)(.*?)</tool_call>", re.DOTALL)  # 内容不是 JSON 也匹配，才能识别"标签闭合但 JSON 残缺"；XML 包裹形式留给下面的 XML 解析
    _FUNC_CALL = re.compile(r"<function=([^>]+)>(.*?)(?:(</function>)|(?=<function=)|$)", re.DOTALL)  # 第 3 组：是否有闭合标签
    _FUNC_PARAM = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

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
        if not sep:  # 没有闭合：以 <think> 开头说明被截断，整段都是思考
            stripped = raw_text.lstrip()
            return {"reasoning": stripped.removeprefix("<think>").strip() if stripped.startswith("<think>") else "", "content": ""}
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

    @classmethod
    def assistant_keys(cls, messages: list[dict[str, Any]]) -> set[str]:
        """历史中所有 assistant 消息的指纹；未命中缓存清理记忆库时，只保留仍在这些历史里的条目。"""
        return {cls.memo_key(messages[i - 1] if i else None, m) for i, m in enumerate(messages) if m.get("role") == "assistant"}

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

    @staticmethod
    def trim_history(messages: list[dict[str, Any]], budget_chars: int) -> list[dict[str, Any]]:
        """按锚点裁剪历史：保留 system + 首轮（子代理任务书通常在这里）+ 最近 2 轮，
        从中间最旧的整轮开始丢，直到塞进 budget_chars；没有中间轮可丢时原样返回。"""
        size = lambda msgs: sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in msgs)
        system = [m for m in messages if m.get("role") == "system"]
        rounds: list[list[dict[str, Any]]] = []
        for msg in (m for m in messages if m.get("role") != "system"):
            if not rounds or msg.get("role") == "user":  # 每轮从 user 开始，工具消息跟随其所属轮次一起丢
                rounds.append([msg])
            else:
                rounds[-1].append(msg)
        if len(rounds) <= 3:
            return messages
        head, middle, tail = rounds[:1], rounds[1:-2], rounds[-2:]
        flat = lambda groups: [m for group in groups for m in group]
        while middle and size(system + flat(head + middle + tail)) > budget_chars:
            middle.pop(0)
        return system + flat(head + middle + tail)

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
        bad = 0
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
                        except json.JSONDecodeError:
                            fn["arguments"] = {}
                            bad += 1
                    elif not isinstance(args, dict):
                        fn["arguments"] = {}
                    tc_dict["function"] = fn
                    clean_tcs.append(tc_dict)
                item["tool_calls"] = clean_tcs
            normalized.append(item)
        if bad:
            log_warn("[工具调用]", f"历史消息里有 {bad} 个 tool_call 的 arguments 不是合法 JSON，已替换为 {{}}（会改变该处前缀）")
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
    def parse_result(
        cls,
        raw_text: str,
        detected_calls: list[dict[str, Any]],
        finish_reason: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> parsedResult:
        """解析模型原文：取出正文与工具调用，按请求的 tools schema 修正参数类型并校验，问题记入 problems。
        解析失败的片段保留在正文里（不静默删除），由调用方决定重试还是原样透传。"""
        problems: list[tuple[str, str]] = []
        truncated = finish_reason == "length"
        text = raw_text

        # 约定：返回文本永远不含 think（思考由 split_raw 单独取出）；先剥再解析，避免误解析 think 里的工具标记
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif text.lstrip().startswith("<think>"):
            problems.append(("truncated" if truncated else "unclosed_think", "思考未闭合，其后的正文/工具调用不可见"))
            text = ""

        calls = [dict(c) for c in detected_calls or []]
        if not calls:
            # Hermes：<tool_call>{json}</tool_call>
            pieces, last = [], 0
            for m in cls._HERMES_CALL.finditer(text):
                pieces.append(text[last : m.start()])
                last = m.end()
                body = m.group(1).strip()
                if not body:
                    continue  # 空的 <tool_call></tool_call>：直接丢弃
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as exc:
                    problems.append(("json_invalid", f"<tool_call> 内不是合法 JSON: {exc.msg}"))
                    pieces.append(m.group(0))
                    continue
                if not isinstance(data, dict) or "name" not in data:
                    problems.append(("json_invalid", "<tool_call> 的 JSON 缺少 name 字段"))
                    pieces.append(m.group(0))
                    continue
                calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": data["name"], "arguments": data.get("arguments", {})}})
            pieces.append(text[last:])
            text = "".join(pieces)

            # XML：<function=name><parameter=k>v</parameter></function>（参数值此时都是字符串，类型在下面按 schema 修正）
            pieces, last = [], 0
            matches = list(cls._FUNC_CALL.finditer(text))
            for i, m in enumerate(matches):
                pieces.append(text[last : m.start()])
                last = m.end()
                name = m.group(1).strip()
                if truncated and i == len(matches) - 1 and m.group(3) is None:
                    problems.append(("truncated", f"{name} 调用被截断（缺少 </function>），未下发半截调用"))
                    pieces.append(m.group(0))
                    continue
                if not name:
                    continue
                args = {pm.group(1).strip(): pm.group(2).strip() for pm in cls._FUNC_PARAM.finditer(m.group(2))}
                calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": name, "arguments": args}})
            pieces.append(text[last:])
            text = "".join(pieces)

            if not calls and not problems and text.count("<tool_call>") > text.count("</tool_call>"):
                problems.append(("truncated" if truncated else "unclosed_call", "<tool_call> 没有闭合，调用不可解析"))

        # 参数规范化 + 按 tools schema 修正类型/校验（无 tools 则只做 JSON 规范化）
        schemas = {(t.get("function") or t).get("name"): (t.get("function") or t).get("parameters") or {} for t in tools or [] if isinstance(t, dict)}
        for call in calls:
            fn = call["function"] = dict(call.get("function", {}))
            name, raw_args = fn.get("name"), fn.get("arguments", {})
            args = raw_args
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError as exc:
                    problems.append(("json_invalid", f"{name} 的 arguments 不是合法 JSON: {exc.msg}"))
                    continue
            if not isinstance(args, dict):
                problems.append(("json_invalid", f"{name} 的 arguments 不是 JSON 对象"))
                fn["arguments"] = json.dumps(args, ensure_ascii=False)
                continue

            schema = schemas.get(name) or {}
            props = schema.get("properties") or {}
            coerced = False
            for key, val in args.items():
                want = (props.get(key) or {}).get("type")
                want = {want} if isinstance(want, str) else set(want or [])
                if not isinstance(val, str) or not want or "string" in want:
                    continue
                cand = val.strip()
                if "boolean" in want and cand.lower() in {"true", "false"}:
                    cand = cand.lower()
                try:
                    parsed = json.loads(cand)
                except json.JSONDecodeError:
                    continue
                kind = "boolean" if isinstance(parsed, bool) else "integer" if isinstance(parsed, int) else "number" if isinstance(parsed, float) else "array" if isinstance(parsed, list) else "object" if isinstance(parsed, dict) else "null" if parsed is None else "string"
                if kind in want or (kind == "integer" and "number" in want):
                    args[key] = parsed
                    coerced = True

            if schemas and name not in schemas:
                problems.append(("unknown_tool", f"工具 {name} 不在请求的 tools 里"))
            missing = [r for r in schema.get("required") or [] if r not in args]
            if missing:
                problems.append(("missing_required", f"{name} 缺少必填参数 {missing}"))
            if coerced or not isinstance(raw_args, str):
                fn["arguments"] = json.dumps(args, ensure_ascii=False)

        # 解析成功才清理残留标签；有解析问题时原样保留，不把模型写的东西悄悄吞掉
        if not {code for code, _ in problems} & {"json_invalid", "unclosed_call", "truncated", "unclosed_think"}:
            text = re.sub(r"<tool_call>\s*</tool_call>", "", text, flags=re.DOTALL)
            text = re.sub(r"</?tool_call>", "", text)
        if not calls and not problems and not text.strip():
            problems.append(("truncated" if truncated else "empty", "模型没有输出任何正文或工具调用"))

        return parsedResult(text.strip(), calls, "tool_calls" if calls else finish_reason, problems)

    @classmethod
    def postprocess_result(
        cls,
        raw_text: str,
        detected_calls: list[dict[str, Any]],
        finish_reason: str,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """兼容旧调用方：只返回 (正文, 工具调用, finish_reason)；需要 schema 校验/问题列表请用 parse_result。"""
        parsed = cls.parse_result(raw_text, detected_calls, finish_reason)
        return parsed.text, parsed.tool_calls, parsed.finish_reason