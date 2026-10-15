from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any

from ...utils.common import info


@dataclass
class sessionCategory:
    is_main: bool  # True: 主会话; False: 支路会话（统一走副模型或主模型 slot 1）
    reason: str  # 判定原因摘要
    session_id: str = ""  # 会话短标识（首条消息指纹），用于日志与指纹清理
    extended: bool = False  # True: 前缀延长命中已知会话; False: 新会话/历史被改写
    model_id: str = ""  # 主会话绑定的模型（KV 在它身上）；空表示尚未绑定


@dataclass
class _track:
    category: sessionCategory
    seen_at: float


class classifier:
    """
    只区分主会话与支路会话，支路不再分级。
    主会话 = 挂载了派生子代理工具的编排者（或工具集与已知主会话一致）。
    只有主会话保存指纹，靠“前缀延长”追踪：本次请求的历史前缀 == 上次请求的整段历史，即同一会话的下一轮，
    继承其属性与绑定的模型。支路会话不保存任何记录，每轮按工具集重判（输入不变，结果就不变）。
    """

    # 只有编排者（主会话）才会挂载的派生子代理工具（Hermes 的叶子子代理拿不到 delegate_task）
    _SPAWN_TOOLS = frozenset({"task", "agent", "dispatch_agent", "spawn_agent", "subagent", "delegate_task"})

    _TRACK_TTL = 7200.0  # 会话指纹存活秒数
    _TRACK_MAX = 256  # 指纹条数上限，超出淘汰最久未见的
    _ACTIVE_SEC = 120.0  # forget() 保留此窗口内仍活跃的会话

    def __init__(self) -> None:
        # 键: 该会话上次请求的整段历史链式指纹（下一轮请求的前缀）
        self._tracks: dict[str, _track] = {}
        self._main_tools: set[str] = set()  # 已确认主会话的工具集指纹，用于 /clear 后重开识别
        self._seen_tools: set[str] = set()  # 已打印过的工具集指纹（每个工具集只打印一次）

    # ------------------------------------------------------------------ 工具函数
    @staticmethod
    def _chain(messages: list[dict[str, Any]]) -> list[str]:
        """逐条累积的历史指纹：chain[i] 代表 messages[:i+1] 的整段前缀。"""
        chain: list[str] = []
        acc = ""
        for msg in messages:
            body = json.dumps([msg.get("role"), msg.get("content"), msg.get("tool_calls"), msg.get("tool_call_id")], ensure_ascii=False, sort_keys=True, default=str)
            acc = hashlib.md5((acc + body).encode("utf-8")).hexdigest()
            chain.append(acc)
        return chain

    @staticmethod
    def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
        names = ((t.get("function") or {}).get("name") or t.get("name") for t in (tools or []) if isinstance(t, dict))
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
        for fp in sorted(self._tracks, key=lambda k: self._tracks[k].seen_at)[: max(0, overflow)]:
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
        stale = [fp for fp, t in self._tracks.items() if t.category.model_id == model_id and t.category.session_id != keep_sid and t.seen_at < cutoff]
        for fp in stale:
            del self._tracks[fp]
        return len(stale)

    # ------------------------------------------------------------------ 判定入口
    def classify(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> sessionCategory:
        if not messages:
            return sessionCategory(False, "空消息")
        chain = self._chain(messages)
        names = self._tool_names(tools)
        matched = self._match(chain)
        if matched:
            category = replace(matched[1].category, extended=True)
            self._remember(category, chain, matched[0])
        else:
            first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), 0)
            category = self._judge(names, chain[first_user][:6])
            if category.is_main:  # 只有主会话保存指纹；支路每轮按工具集重判，结果是确定的
                self._remember(category, chain)

        info(
            "[分级] 判定",
            "主会话" if category.is_main else "支路",
            f"#{category.session_id or '-'}",
            "前缀延长命中" if matched else "无前缀血缘",
            f"消息={len(messages)}",
            f"工具={len(names)}#{self._tool_fp(names) or '-'}",
            f"在册主会话={len(self._tracks)}",
        )
        return category

    @staticmethod
    def _tool_fp(names: set[str]) -> str:
        return hashlib.md5(",".join(sorted(n.lower() for n in names)).encode()).hexdigest()[:8] if names else ""

    def _judge(self, names: set[str], sid: str) -> sessionCategory:
        """新会话（无前缀血缘）的首次判定；后续轮次靠前缀延长继承，不再重判。"""
        lowered = {n.lower() for n in names}
        tool_fp = self._tool_fp(names)
        if tool_fp and tool_fp not in self._seen_tools:  # 新工具集打印一次，用于核对派生子代理词表
            self._seen_tools.add(tool_fp)
            info("[分级] 新工具集", tool_fp, f"共{len(names)}个:", ",".join(sorted(names)))

        if lowered & self._SPAWN_TOOLS:
            reason = f"挂载派生子代理工具 {sorted(lowered & self._SPAWN_TOOLS)}，属于编排者"
        elif tool_fp and tool_fp in self._main_tools:
            reason = "工具集与已知主会话一致（新开/压缩后重开）"
        else:
            return sessionCategory(False, "无派生子代理工具，按支路处理", sid)

        if tool_fp:
            self._main_tools.add(tool_fp)
        return sessionCategory(True, reason, sid)
