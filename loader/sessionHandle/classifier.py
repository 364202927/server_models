from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any


@dataclass
class sessionCategory:
    is_main: bool              # True: 主会话; False: 支路会话
    importance_lv: int         # 1~4 级（主会话与 4 级必须由主力模型处理）
    target_model_tier: str     # "heavy" (主力大模型) 或 "light" (轻量小模型)
    reason: str                # 判定原因摘要
    session_id: str = ""       # 会话短标识（首条消息指纹），用于日志与指纹清理
    extended: bool = False     # True: 前缀延长命中已知会话; False: 新会话/历史被改写
    model_id: str = ""         # 主会话绑定的模型（KV 在它身上）；空表示尚未绑定

    @property
    def priority(self) -> int:
        """排队优先级，数值越小越先：主会话 0，支路 Lv4=1 … Lv1=4。"""
        return 0 if self.is_main else 5 - self.importance_lv


@dataclass
class _track:
    category: sessionCategory
    seen_at: float


class classifier:
    """
    识别主/支会话，并对支会话做 1~4 级重要度评级。主会话与 Lv4 走主力模型，Lv1~3 走轻量模型。
    只有主会话保存指纹，靠“前缀延长”追踪：本次请求的历史前缀 == 上次请求的整段历史，即同一会话的下一轮，
    继承其属性与绑定的模型。支路会话不保存任何记录，每轮按工具集与首条任务描述重判（输入不变，结果就不变）。
    """

    # 写/执行类工具（按分词匹配，如 NotebookEdit -> notebook,edit）
    _WRITE_TOKENS = frozenset({"edit", "write", "patch", "bash", "shell", "exec", "execute",
                               "terminal", "run", "cmd", "delete", "remove", "move", "apply"})
    # 只读类工具
    _READ_TOKENS = frozenset({"read", "grep", "glob", "search", "list", "ls", "view", "find", "fetch", "get", "cat", "tree"})
    # 不影响判级的规划类工具（名字里带 write 但不改文件）
    _NEUTRAL_TOOLS = frozenset({"todowrite", "todoread", "exitplanmode", "enterplanmode", "askuserquestion"})
    # 只有编排者（主会话）才会挂载的派生子代理工具
    _SPAWN_TOOLS = frozenset({"task", "agent", "dispatch_agent", "spawn_agent", "subagent"})

    _HEAVY_INTENT = re.compile(r"(refactor|architect|implement|debug|root\s*cause|重构|编写|实现|排查原因)", re.IGNORECASE)
    _LIGHT_INTENT = re.compile(r"(summari[sz]e|extract|\btitle\b|find\s*symbol|总结|概括|标题|查找|提取)", re.IGNORECASE)

    _TRACK_TTL = 7200.0        # 会话指纹存活秒数
    _TRACK_MAX = 256           # 指纹条数上限，超出淘汰最久未见的
    _ACTIVE_SEC = 120.0        # forget() 保留此窗口内仍活跃的会话

    def __init__(self) -> None:
        # 键: 该会话上次请求的整段历史链式指纹（下一轮请求的前缀）
        self._tracks: dict[str, _track] = {}
        self._main_tools: set[str] = set()   # 已确认主会话的工具集指纹，用于 /clear 后重开识别

    # ------------------------------------------------------------------ 工具函数
    @staticmethod
    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        return ""

    @staticmethod
    def _tokens(name: str) -> set[str]:
        return set(re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower())) - {""}

    @staticmethod
    def _chain(messages: list[dict[str, Any]]) -> list[str]:
        """逐条累积的历史指纹：chain[i] 代表 messages[:i+1] 的整段前缀。"""
        chain: list[str] = []
        acc = ""
        for msg in messages:
            body = json.dumps([msg.get("role"), msg.get("content"), msg.get("tool_calls"), msg.get("tool_call_id")],
                              ensure_ascii=False, sort_keys=True, default=str)
            acc = hashlib.md5((acc + body).encode("utf-8")).hexdigest()
            chain.append(acc)
        return chain

    @staticmethod
    def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
        names = (
            (t.get("function") or {}).get("name") or t.get("name")
            for t in (tools or []) if isinstance(t, dict)
        )
        return {str(n) for n in names if n}

    # ------------------------------------------------------------------ 指纹追踪
    def _match(self, chain: list[str]) -> tuple[str, _track] | None:
        """从最长前缀向短找，命中即同一会话的下一轮（含完全相同的重试）。"""
        for fp in reversed(chain):
            track = self._tracks.get(fp)
            if track:
                return fp, track
        return None

    def _remember(self, category: sessionCategory, chain: list[str], old_fp: str | None = None) -> None:
        now = time.time()
        self._tracks.pop(old_fp or "", None)
        self._tracks[chain[-1]] = _track(category, now)
        expired = [fp for fp, t in self._tracks.items() if now - t.seen_at > self._TRACK_TTL]
        for fp in expired:
            del self._tracks[fp]
        overflow = len(self._tracks) - self._TRACK_MAX
        for fp in sorted(self._tracks, key=lambda k: self._tracks[k].seen_at)[:max(0, overflow)]:
            del self._tracks[fp]

    @staticmethod
    def bind(category: sessionCategory, model_id: str) -> None:
        """记录主会话落在哪个模型上。category 就是指纹表里存的那个对象，改它即改指纹；支路会话不绑定。"""
        if category.is_main:
            category.model_id = model_id

    def unbind(self, model_ids: set[str]) -> int:
        """模型被换出池后，清掉绑定在它们身上的会话归属（会话下次按空闲度重新挑）。"""
        bound = [t.category for t in self._tracks.values() if t.category.model_id in model_ids]
        for category in bound:
            category.model_id = ""
        return len(bound)

    def forget(self, model_id: str, keep_sid: str) -> int:
        """某模型上的主会话未命中缓存时，清理绑定在该模型上的陈旧指纹（不碰其它模型的会话），返回清理条数。"""
        cutoff = time.time() - self._ACTIVE_SEC
        stale = [fp for fp, t in self._tracks.items()
                 if t.category.model_id == model_id and t.category.session_id != keep_sid and t.seen_at < cutoff]
        for fp in stale:
            del self._tracks[fp]
        return len(stale)

    # ------------------------------------------------------------------ 判定入口
    def classify(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        model_hint: str = "",
    ) -> sessionCategory:
        if not messages:
            return sessionCategory(False, 3, "light", "空消息")
        chain = self._chain(messages)
        matched = self._match(chain)
        if matched:
            category = replace(matched[1].category, extended=True)
            self._remember(category, chain, matched[0])
            return category

        first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), 0)
        category = self._judge(messages, self._tool_names(tools), max_tokens, model_hint, chain[first_user][:6])
        if category.is_main:   # 只有主会话保存指纹；支路每轮按工具与任务意图重判，结果是确定的
            self._remember(category, chain)
        return category

    def _judge(
        self,
        messages: list[dict[str, Any]],
        names: set[str],
        max_tokens: int | None,
        model_hint: str,
        sid: str,
    ) -> sessionCategory:
        """新会话（无前缀血缘）的首次判定；后续轮次靠前缀延长继承，不再重判。"""
        msg_count = len(messages)
        lowered = {n.lower() for n in names}
        tool_fp = hashlib.md5(",".join(sorted(lowered)).encode()).hexdigest()[:8] if names else ""

        # 1. 主会话：能派生子代理 / 工具集与已知主会话一致
        main_reason = ""
        if lowered & self._SPAWN_TOOLS:
            main_reason = "挂载派生子代理工具，属于编排者"
        elif tool_fp and tool_fp in self._main_tools:
            main_reason = "工具集与已知主会话一致（新开/压缩后重开）"
        if main_reason:
            if tool_fp:
                self._main_tools.add(tool_fp)
            return sessionCategory(True, 4, "heavy", main_reason, sid)

        # 2. 支路分级：工具特征 + 任务意图（只看首条 user，即任务描述；system 与工具输出会污染关键词）
        real = {n for n in names if n.lower() not in self._NEUTRAL_TOOLS}
        toks = {n: self._tokens(n) for n in real}
        has_write = any(t & self._WRITE_TOKENS for t in toks.values())
        pure_read = bool(real) and all(t & self._READ_TOKENS and not t & self._WRITE_TOKENS for t in toks.values())

        first_user = next((m for m in messages if m.get("role") == "user"), {})
        intent = self._text(first_user.get("content"))[:2000]
        heavy_intent = bool(self._HEAVY_INTENT.search(intent))
        light_intent = bool(self._LIGHT_INTENT.search(intent))
        small_hint = "haiku" in model_hint.lower()   # 客户端自己把请求标成小模型（标题/话题判断等）

        if has_write or (heavy_intent and not pure_read):
            return sessionCategory(False, 4, "heavy", "支路任务带修改/执行权限或核心推理意图，分配给主力模型", sid)
        if (max_tokens and max_tokens <= 64) or (not real and msg_count <= 3 and (light_intent or small_hint)):
            return sessionCategory(False, 1, "light", "极短生成或无工具的标题/摘要类元任务", sid)
        if pure_read:
            return sessionCategory(False, 2, "light", "仅只读工具的扫描/检索任务", sid)
        return sessionCategory(False, 3, "light", "常规分析/格式转换类支路任务", sid)
