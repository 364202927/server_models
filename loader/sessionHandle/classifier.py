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
    rewound: bool = False  # True: 客户端发生了回退/二次编辑重发（非严格追加）


@dataclass
class _track:
    category: sessionCategory
    seen_at: float
    depth: int = 0  # 记录该指纹对应的消息条数，用于判断新请求是正常延长还是 ESC 回退


class classifier:
    """
    只区分主会话与支路会话，支路不再分级。
    主会话 = 挂载了派生子代理工具的编排者（或工具集与已知主会话一致）。
    只有主会话保存指纹，靠“前缀延长”追踪：支持正常追加，也支持客户端按 ESC 撤回/重新编辑最后一条消息。
    """

    _SPAWN_TOOLS = frozenset({"task", "agent", "dispatch_agent", "spawn_agent", "subagent", "delegate_task"})

    _TRACK_TTL = 7200.0
    _TRACK_MAX = 256
    _ACTIVE_SEC = 120.0

    def __init__(self) -> None:
        self._tracks: dict[str, _track] = {}
        self._main_tools: set[str] = set()
        self._seen_tools: set[str] = set()

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
    def _match(self, chain: list[str]) -> tuple[str, _track, int] | None:
        """从最长前缀向短找，命中即同一会话（返回 命中的指纹, track记录, 命中位置索引）。"""
        for idx in range(len(chain) - 1, -1, -1):
            fp = chain[idx]
            track = self._tracks.get(fp)
            if track:
                return fp, track, idx + 1
        return None

    def _remember(self, category: sessionCategory, chain: list[str], matched_fp: str | None = None) -> None:
        """保留 matched_fp 不删除，作为客户端按 ESC 重新编辑时的回退锚点；同时登记最新末端与倒数第二条锚点。"""
        now = time.time()
        depth = len(chain)
        if matched_fp and matched_fp in self._tracks:
            self._tracks[matched_fp].seen_at = now
        if len(chain) >= 2:
            self._tracks[chain[-2]] = _track(category, now, depth - 1)
        self._tracks[chain[-1]] = _track(category, now, depth)

        expired = [fp for fp, t in self._tracks.items() if now - t.seen_at > self._TRACK_TTL]
        for fp in expired:
            del self._tracks[fp]
        overflow = len(self._tracks) - self._TRACK_MAX
        for fp in sorted(self._tracks, key=lambda k: self._tracks[k].seen_at)[: max(0, overflow)]:
            del self._tracks[fp]

    @staticmethod
    def bind(category: sessionCategory, model_id: str) -> None:
        if category.is_main:
            category.model_id = model_id

    def unbind(self, model_ids: set[str]) -> int:
        bound = [t.category for t in self._tracks.values() if t.category.model_id in model_ids]
        for category in bound:
            category.model_id = ""
        return len(bound)

    def forget(self, model_id: str, keep_sid: str) -> int:
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
            matched_fp, track, matched_depth = matched
            # 如果当前总长度 <= 上次记录的深度，说明用户按了 ESC 撤回重发或重试
            is_rewound = len(chain) <= track.depth and matched_fp != chain[-1]
            category = replace(track.category, extended=not is_rewound, rewound=is_rewound)
            self._remember(category, chain, matched_fp)
        else:
            first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), 0)
            category = self._judge(names, chain[first_user][:6])
            if category.is_main:
                self._remember(category, chain)

        state_str = "回退重发" if category.rewound else ("前缀延长命中" if matched else "无前缀血缘")
        info(
            "[分级] 判定",
            "主会话" if category.is_main else "支路",
            f"#{category.session_id or '-'}",
            state_str,
            f"消息={len(messages)}",
            f"工具={len(names)}#{self._tool_fp(names) or '-'}",
            f"在册锚点={len(self._tracks)}",
        )
        return category

    @staticmethod
    def _tool_fp(names: set[str]) -> str:
        return hashlib.md5(",".join(sorted(n.lower() for n in names)).encode()).hexdigest()[:8] if names else ""

    def _judge(self, names: set[str], sid: str) -> sessionCategory:
        lowered = {n.lower() for n in names}
        tool_fp = self._tool_fp(names)
        if tool_fp and tool_fp not in self._seen_tools:
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