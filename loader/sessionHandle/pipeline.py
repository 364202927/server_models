from __future__ import annotations

import asyncio
import itertools
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from ...utils.common import cache_verdict, info, warn
from ..llmFramework.baseInference import GenerationResult
from ..model_spec import modelSpec
from .classifier import classifier, sessionCategory


@dataclass(eq=False)
class _lane:
    """池内的一个模型：并发上限 = 该模型的并行槽位数（与 llama-server 的 -np 一致）。"""

    spec: modelSpec
    slots: int = 1
    running: int = 0
    used: set[int] = field(default_factory=set)
    last_sid: dict[int, str] = field(default_factory=dict)
    active_leases: dict[int, lease] = field(default_factory=dict)  # 槽位号 -> 当前正在执行的 lease，用于同会话重发时中止旧任务


@dataclass(eq=False)
class _ticket:
    category: sessionCategory
    want: int | None
    seq: int
    enq_at: float
    granted: asyncio.Future


@dataclass(eq=False)
class _pool:
    isMain: bool
    lanes: list[_lane] = field(default_factory=list)
    waiting: list[_ticket] = field(default_factory=list)

    @property
    def alive(self) -> list[_lane]:
        return [lane for lane in self.lanes if lane.spec.isAlive]


@dataclass
class lease:
    lane: _lane
    isMain: bool
    waited: float = 0.0
    ahead: int = 0
    prev_sid: str = ""
    prev_bound: str = ""
    slot: int | None = None
    warnings: list[str] = field(default_factory=list)
    job: asyncio.Future | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    sid: str = ""


class pipeline:
    def __init__(self) -> None:
        self._classifier = classifier()
        self._warned: set[str] = set()
        self._seq = itertools.count()
        self._main = _pool(True)
        self._sub = _pool(False)
        self._switch_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()

    # ------------------------------------------------------------------ 登记与查询
    def classify(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> sessionCategory:
        return self._classifier.classify(messages, tools)

    @property
    def running(self) -> int:
        return sum(lane.running for pool in (self._main, self._sub) for lane in pool.lanes)

    @property
    def pending(self) -> int:
        return len(self._main.waiting) + len(self._sub.waiting)

    def busy(self) -> bool:
        return bool(self.running or self.pending)

    def addModel(self, isMain: bool, models: list[modelSpec]) -> bool:
        if self.busy():
            return False
        pool, other = (self._main, self._sub) if isMain else (self._sub, self._main)
        unique = list({spec.model_id: spec for spec in models}.values())
        removed = {lane.spec.model_id for lane in pool.lanes} - {s.model_id for s in unique} - {l.spec.model_id for l in other.lanes}
        pool.lanes = [_lane(spec, self._lane_slots(spec)) for spec in unique]
        cleared = self._classifier.unbind(removed)
        info(
            "[调度] 登记",
            "主池" if isMain else "副池",
            [s.model_id for s in unique],
            f"解除会话绑定 {cleared} 个" if cleared else "",
        )
        return True

    async def settled(self) -> None:
        await self._idle.wait()

    async def ensure(self, specs: list[modelSpec], isMain: bool, args: dict[str, Any]) -> None:
        await self._switch_lock.acquire()
        job = None
        try:
            register = not all(self.has(spec) for spec in specs)
            if register and self.busy():
                raise RuntimeError(f"pipeline 仍有任务在执行，暂不能切换{'主' if isMain else '副'}模型到 {[s.model_id for s in specs]}")
            self._idle.clear()
            job = asyncio.ensure_future(self._load(specs, args))
            await asyncio.shield(job)
            if register and not self.addModel(isMain, specs):
                raise RuntimeError("加载期间 pipeline 被占用，请稍后重试")
        finally:
            self._after(job, self._end_switch)

    @staticmethod
    async def _load(specs: list[modelSpec], args: dict[str, Any]) -> None:
        for spec in specs:
            await asyncio.to_thread(spec.ensure, args)

    def _end_switch(self) -> None:
        self._idle.set()
        self._switch_lock.release()

    @staticmethod
    def _after(job: asyncio.Future | None, release: Callable[[], None]) -> None:
        if job is None or job.done():
            release()
            return

        def _on_done(done: asyncio.Future) -> None:
            if not done.cancelled():
                done.exception()
            release()

        job.add_done_callback(_on_done)

    def hasMain(self) -> bool:
        return bool(self._main.alive)

    @staticmethod
    def _lane_slots(spec: modelSpec) -> int:
        """槽位数以引擎实测为准（-np 可能不被老版本接受），回退到配置值。"""
        actual = int(spec.loader.effective_load.get("parallel_slots", 0)) if spec.loader and spec.isAlive else 0
        return max(1, actual or int(spec.full_load().get("parallel_slots", 1)))

    def has(self, spec: modelSpec) -> bool:
        return spec.isAlive and any(lane.spec is spec for pool in (self._main, self._sub) for lane in pool.lanes)

    def describe(self) -> str:
        fmt = lambda pool: "[" + ", ".join(f"{lane.spec.model_id}({lane.spec.state})" for lane in pool.lanes) + "]"
        return f"主池={fmt(self._main)} 副池={fmt(self._sub)}"

    def stats(self) -> dict[str, Any]:
        return {
            name: {
                "models": [
                    {
                        "model": l.spec.model_id,
                        "state": l.spec.state,
                        "slots": l.slots,
                        "running": l.running,
                    }
                    for l in pool.lanes
                ],
                "waiting": len(pool.waiting),
            }
            for name, pool in (("main", self._main), ("sub", self._sub))
        }

    # ------------------------------------------------------------------ 路由与排队
    def _route(self, category: sessionCategory) -> tuple[_pool, int | None, str]:
        if not self._main.alive:
            raise RuntimeError("没有可用的主模型")
        if category.is_main:
            return self._main, 0, ""
        if self._sub.alive:
            return self._sub, None, ""
        if any(lane.slots >= 2 for lane in self._main.alive):
            return self._main, 1, ""
        return self._main, 0, "没有副模型且主模型无 slot 1（parallel_slots<2），支路会话被迫占用主会话槽位 0，会打断主会话缓存"

    @staticmethod
    def _rank(ticket: _ticket) -> tuple[int, int]:
        return (0 if ticket.category.is_main else 1), ticket.seq

    @staticmethod
    def _pick(pool: _pool, ticket: _ticket) -> tuple[_lane, int] | None:
        category, want = ticket.category, ticket.want
        lanes = pool.alive
        bound = [lane for lane in lanes if category.is_main and lane.spec.model_id == category.model_id]
        free: list[tuple[_lane, int]] = []
        for lane in bound or lanes:
            slot = want if want is not None else next((i for i in range(lane.slots) if i not in lane.used), None)
            if slot is not None and slot < lane.slots and slot not in lane.used:
                free.append((lane, slot))
        return min(free, key=lambda item: item[0].running / item[0].slots, default=None)
    
    def _abort_stale_main_session(self, pool: _pool, category: sessionCategory) -> None:
        """主会话独占 Slot 0：用户 ESC 回退重编辑、/clear、/resume 开启新主会话时，
        清退队列里别的会话残留票据，并通知 Slot 0 上还在跑的旧任务中止。
        同前缀等长的重发（extended 且非 rewound）是客户端超时重试，旧任务结果仍然有用，不中止，排队等它跑完。"""
        if not category.is_main:
            return
        # 1. 新主会话请求到来：别的会话排队中的票据直接作废；同会话的票据保留（可能是同一次重试，FIFO 依次服务）
        for t in list(pool.waiting):
            if t.category.is_main and t.category.session_id != category.session_id and not t.granted.done():
                t.granted.cancel()
                pool.waiting.remove(t)
                warn("[调度] 清理残留的主会话排队请求", f"旧#{t.category.session_id} -> 新#{category.session_id}")
        # 2. 只有"旧任务确定作废"才掐断：ESC 回退重编辑（rewound），或 /clear、/resume 换了会话（无血缘新会话）
        for lane in pool.lanes:
            active = lane.active_leases.get(0)
            if not active or active.cancel.is_set():
                continue
            if not category.rewound and (category.extended or active.sid == category.session_id):
                info("[调度] 同前缀重发（客户端重试/追加）不中止在跑的旧任务，排队等待 Slot 0", f"#{category.session_id}")
                continue
            warn("[调度] 主会话重发或切换(/clear|/resume)，中止 Slot 0 旧任务", f"旧#{active.sid} -> 新#{category.session_id}")
            active.cancel.set()

    def _pump(self, pool: _pool) -> None:
        for ticket in sorted(pool.waiting, key=self._rank):
            if ticket.granted.done():
                pool.waiting.remove(ticket)
                continue
            picked = self._pick(pool, ticket)
            if picked is None:
                continue
            lane, slot = picked
            pool.waiting.remove(ticket)
            lane.running += 1
            lane.used.add(slot)
            ticket.granted.set_result(picked)

    def _release(self, pool: _pool, lane: _lane, slot: int) -> None:
        lane.running = max(0, lane.running - 1)
        lane.used.discard(slot)
        lane.active_leases.pop(slot, None)
        self._pump(pool)

    @asynccontextmanager
    async def _acquire(self, pool: _pool, category: sessionCategory, want: int | None, warning: str = "") -> AsyncIterator[lease]:
        # self._abort_stale_same_session(pool, category)
        self._abort_stale_main_session(pool, category)
        ticket = _ticket(
            category,
            want,
            next(self._seq),
            time.monotonic(),
            asyncio.get_running_loop().create_future(),
        )
        ahead = sum(lane.running for lane in pool.lanes) + len(pool.waiting)
        pool.waiting.append(ticket)
        self._pump(pool)
        grant: lease | None = None
        try:
            lane, slot = await ticket.granted
            prev_bound, prev_sid, lane.last_sid[slot] = (
                category.model_id,
                lane.last_sid.get(slot, ""),
                category.session_id,
            )
            self._classifier.bind(category, lane.spec.model_id)
            grant = lease(
                lane,
                pool.isMain,
                time.monotonic() - ticket.enq_at,
                ahead,
                prev_sid,
                prev_bound,
                slot,  # 槽号始终下发给引擎：支路清理要知道清哪个槽，主/支路也才不会被引擎随机挑槽
                [warning] if warning else [],
                sid=category.session_id,
            )
            lane.active_leases[slot] = grant
            yield grant
        finally:
            if ticket in pool.waiting:
                pool.waiting.remove(ticket)
                self._pump(pool)
            elif not ticket.granted.cancelled():
                granted_lane, granted_slot = ticket.granted.result()
                job = grant.job if grant else None
                if job is not None and not job.done():
                    grant.cancel.set()
                self._after(job, lambda: self._release(pool, granted_lane, granted_slot))

    # ------------------------------------------------------------------ 执行
    async def run(self, category: sessionCategory, messages: list[dict[str, Any]], overrides: dict[str, Any]) -> GenerationResult:
        pool, want, warning = self._route(category)
        if warning and category.session_id not in self._warned:
            self._warned.add(category.session_id)
            warn("[调度]", self._who(category), warning)
        async with self._acquire(pool, category, want, warning) as grant:
            cfg = dict(overrides)
            cfg["branch"] = not category.is_main
            if grant.slot is not None:
                cfg["id_slot"] = grant.slot
            self._log_dispatch(category, grant)
            # 发生回退重发或主会话时开启陈旧思考指纹清理
            prune = category.is_main or category.rewound
            grant.job = asyncio.ensure_future(
                asyncio.to_thread(
                    grant.lane.spec.generate,
                    messages,
                    cfg,
                    prune,
                    grant.cancel,
                )
            )
            result = await asyncio.shield(grant.job)
        result.warnings = grant.warnings
        self._log_result(category, grant, messages, overrides, result)
        return result

    # ------------------------------------------------------------------ 日志
    @staticmethod
    def _who(category: sessionCategory) -> str:
        kind = "主会话" if category.is_main else "支路"
        state = "回退" if category.rewound else ("延长" if category.extended else "新会话")
        return f"{kind}#{category.session_id or '-'}·{state}"

    def _log_dispatch(self, category: sessionCategory, grant: lease) -> None:
        model = grant.lane.spec.model_id
        if grant.prev_bound == model:
            how = "按指纹绑定"
        elif grant.prev_bound:
            how = f"原绑定 {grant.prev_bound} 已不在池，重新挑选"
        else:
            how = "按空闲度挑选"
        info(
            "[调度]",
            self._who(category),
            f"-> {'主池' if grant.isMain else '副池'}:{model}",
            how,
            f"id_slot={grant.slot}",
            f"各槽占用={sorted(grant.lane.used)}/{grant.lane.slots}",
            f"本槽上次会话=#{grant.prev_sid or '-'}",
            f"各槽最近会话={dict(sorted(grant.lane.last_sid.items()))}",
            f"排队{grant.waited:.1f}s(前方{grant.ahead})",
            f"原因={category.reason}",
            f"注意: {'；'.join(grant.warnings)}" if grant.warnings else "",
        )

    def _log_result(self, category: sessionCategory, grant: lease, messages: list[dict[str, Any]], overrides: dict[str, Any], result: GenerationResult) -> None:
        model = grant.lane.spec.model_id
        info(f"[生成] {model} 速率: {result.tokens_per_second:.2f} token/s, 生成: {result.tokens_generated} tokens(思考 {len(result.reasoning)} 字, 工具调用 {len(result.tool_calls)} 个), " f"耗时: {result.time_seconds:.2f}s, 剩余 KV Cache: {result.kv_remaining} tokens")

        verdict, ratio = cache_verdict(result.cached_tokens, result.prompt_tokens)
        dropped = self._classifier.forget(model, category.session_id) if category.is_main and verdict == "未命中" else 0
        restored_all = result.memo_restored + result.memo_missed
        parts = [
            f"[缓存] {verdict} {ratio:.1%}",
            (f"复用 {result.cached_tokens}/{result.prompt_tokens}" if verdict != "未知" else "引擎未提供缓存数据"),
            self._who(category),
            f"模型={model}" + (f" 槽={grant.slot}" if grant.slot is not None else ""),
            f"排队{grant.waited:.1f}s 生成{result.time_seconds:.1f}s",
            f"消息={len(messages)} 工具={len(overrides.get('tools') or [])} 思考还原={result.memo_restored}/{restored_all}",
        ]
        if verdict in {"未命中", "部分命中"}:
            hints = []
            if grant.prev_sid and category.session_id and grant.prev_sid != category.session_id:
                hints.append(f"模型上一会话为#{grant.prev_sid}，KV 被覆盖")
            if result.memo_missed:
                hints.append(f"{result.memo_missed} 条 assistant 思考未还原，前缀被改写")
            if category.rewound:
                hints.append("用户回退/重编辑了消息")
            elif not category.extended:
                hints.append("新会话/压缩后首轮")
            elif not hints:
                hints.append("同会话历史被客户端改写，对比 debug/prompt diff")
            parts.append("可能原因: " + "；".join(hints))
        if dropped or result.memo_pruned:
            parts.append(f"已清理 会话指纹={dropped} 思考记忆={result.memo_pruned}")
        (info if verdict in {"命中", "未知"} else warn)(*parts)