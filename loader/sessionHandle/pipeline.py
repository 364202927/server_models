from __future__ import annotations

import asyncio
import itertools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

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
    last_sid: str = ""         # 上一个在此模型上跑的会话，用于判断 KV 是否被其他会话覆盖


@dataclass(eq=False)
class _ticket:
    category: sessionCategory
    seq: int
    enq_at: float
    granted: asyncio.Future    # 放行后结果为分到的 _lane


@dataclass(eq=False)
class _pool:
    isMain: bool
    lanes: list[_lane] = field(default_factory=list)
    waiting: list[_ticket] = field(default_factory=list)
    side_running: int = 0      # 池内正在跑的支路任务数（用于给主会话预留槽位）

    @property
    def alive(self) -> list[_lane]:
        return [lane for lane in self.lanes if lane.spec.isAlive]

    @property
    def total_slots(self) -> int:
        return sum(lane.slots for lane in self.alive)


@dataclass
class lease:
    lane: _lane
    isMain: bool               # 是否来自主池
    waited: float = 0.0        # 排队等待秒数
    ahead: int = 0             # 入队时池里已有的任务数
    prev_sid: str = ""         # 该模型上一个会话
    prev_bound: str = ""       # 入队前会话绑定的模型
    slot: int | None = None    # KV 槽位提示：主会话固定 0，支路 1；None 交给引擎自选


class pipeline:
    """
    会话调度器：只拿 modelSpec 的 generate，不管模型怎么生、怎么睡。
    - 主池/副池结构相同，都可放多个模型；addModel 整体替换，池里还有任务时拒绝切换（避免打断正在跑的会话）。
    - 路由只看分级：主会话与 Lv4 走主池；Lv1~3 走副池，副池没有可用模型时回落主池。
    - 主会话按指纹绑定模型：绑定的模型即使忙也排队等它，别的模型空闲也不接；没有绑定才按空闲度挑。
    - 排队：主会话最先，Lv4 支路不插队，支路按等级排且等得越久越靠前（防饥饿），运行中的任务不抢占。
    """

    _AGING_SEC = 30.0   # 支路每多等这么久，优先级提升一档

    def __init__(self) -> None:
        self._classifier = classifier()
        self._seq = itertools.count()
        self._main = _pool(True)
        self._sub = _pool(False)

    # ------------------------------------------------------------------ 登记与查询
    def classify(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        model_hint: str = "",
    ) -> sessionCategory:
        return self._classifier.classify(messages, tools, max_tokens, model_hint)

    def busy(self) -> bool:
        """整个 pipeline 只要还有任务在跑或在排队就算忙。"""
        pools = (self._main, self._sub)
        return any(lane.running for pool in pools for lane in pool.lanes) or any(pool.waiting for pool in pools)

    def addModel(self, isMain: bool, models: list[modelSpec]) -> bool:
        """用 models 整体替换主池/副池（传 [] 即清空）。pipeline 忙时不切换，返回 False。"""
        if self.busy():
            return False
        pool, other = (self._main, self._sub) if isMain else (self._sub, self._main)
        unique = list({spec.model_id: spec for spec in models}.values())
        removed = {lane.spec.model_id for lane in pool.lanes} - {s.model_id for s in unique} - {l.spec.model_id for l in other.lanes}
        pool.lanes = [_lane(spec, max(1, int(spec.full_load().get("parallel_slots", 1)))) for spec in unique]
        cleared = self._classifier.unbind(removed)
        info("[调度] 登记", "主池" if isMain else "副池", [s.model_id for s in unique],
             f"解除会话绑定 {cleared} 个" if cleared else "")
        return True

    def hasMain(self) -> bool:
        return bool(self._main.alive)

    def has(self, spec: modelSpec) -> bool:
        """spec 已在某个池里且仍存活（休眠算存活，请求到达时 generate 会唤醒它）。"""
        return spec.isAlive and any(lane.spec is spec for pool in (self._main, self._sub) for lane in pool.lanes)

    def describe(self) -> str:
        fmt = lambda pool: "[" + ", ".join(f"{lane.spec.model_id}({lane.spec.state})" for lane in pool.lanes) + "]"
        return f"主池={fmt(self._main)} 副池={fmt(self._sub)}"

    def stats(self) -> dict[str, Any]:
        return {
            name: {
                "models": [{"model": l.spec.model_id, "state": l.spec.state, "slots": l.slots, "running": l.running}
                           for l in pool.lanes],
                "waiting": len(pool.waiting),
            }
            for name, pool in (("main", self._main), ("sub", self._sub))
        }

    # ------------------------------------------------------------------ 路由与排队
    def _route(self, category: sessionCategory) -> _pool:
        if category.target_model_tier == "light" and self._sub.alive:
            return self._sub
        if self._main.alive:
            return self._main
        raise RuntimeError("没有可用的主模型")

    def _rank(self, ticket: _ticket, now: float) -> tuple[int, float, int]:
        category = ticket.category
        if category.is_main:
            return 0, 0.0, ticket.seq
        return 1, category.priority * self._AGING_SEC - (now - ticket.enq_at), ticket.seq

    @staticmethod
    def _pick(pool: _pool, category: sessionCategory) -> _lane | None:
        """挑一个有空槽的模型：主会话优先它绑定的模型（忙就等，不换别的）；否则取负载最低的。"""
        lanes = pool.alive
        bound = [lane for lane in lanes if category.is_main and lane.spec.model_id == category.model_id]
        free = [lane for lane in (bound or lanes) if lane.running < lane.slots]
        # 主池总槽位 >= 2 时，给主会话预留 1 个槽位，支路占不满
        reserve = 1 if pool.isMain and pool.total_slots >= 2 else 0
        if not category.is_main and pool.side_running >= pool.total_slots - reserve:
            return None
        return min(free, key=lambda lane: lane.running / lane.slots, default=None)

    def _pump(self, pool: _pool) -> None:
        """按优先级放行等待者；某张票要的模型没空槽时跳过它，不挡住去别的模型的票。"""
        now = time.monotonic()
        for ticket in sorted(pool.waiting, key=lambda t: self._rank(t, now)):
            if ticket.granted.done():           # 等待方已被取消
                pool.waiting.remove(ticket)
                continue
            lane = self._pick(pool, ticket.category)
            if lane is None:
                continue
            pool.waiting.remove(ticket)
            lane.running += 1
            pool.side_running += 0 if ticket.category.is_main else 1
            ticket.granted.set_result(lane)

    def _release(self, pool: _pool, lane: _lane, category: sessionCategory) -> None:
        lane.running -= 1
        pool.side_running -= 0 if category.is_main else 1
        self._pump(pool)

    @asynccontextmanager
    async def _acquire(self, pool: _pool, category: sessionCategory) -> AsyncIterator[lease]:
        """申请执行许可；取消/超时/异常都会正确归还名额并唤醒后续任务。"""
        ticket = _ticket(category, next(self._seq), time.monotonic(), asyncio.get_running_loop().create_future())
        ahead = sum(lane.running for lane in pool.lanes) + len(pool.waiting)
        pool.waiting.append(ticket)
        self._pump(pool)
        try:
            lane = await ticket.granted
        except asyncio.CancelledError:
            if ticket in pool.waiting:
                pool.waiting.remove(ticket)
            elif ticket.granted.done() and not ticket.granted.cancelled():
                self._release(pool, ticket.granted.result(), category)   # 已放行但调用方在恢复前被取消
            self._pump(pool)
            raise

        prev_bound, prev_sid, lane.last_sid = category.model_id, lane.last_sid, category.session_id
        slot = None
        if pool.isMain and lane.slots >= 2:   # 主池多槽位时主会话固定槽 0；仅 2 槽时支路固定槽 1
            slot = 0 if category.is_main else (1 if lane.slots == 2 else None)
        self._classifier.bind(category, lane.spec.model_id)
        try:
            yield lease(lane, pool.isMain, time.monotonic() - ticket.enq_at, ahead, prev_sid, prev_bound, slot)
        finally:
            self._release(pool, lane, category)

    # ------------------------------------------------------------------ 执行
    async def run(self, category: sessionCategory, messages: list[dict[str, Any]], overrides: dict[str, Any]) -> GenerationResult:
        """排队 -> 在分到的模型上生成。失败（含副模型）原样上抛，由调用方报给客户端。"""
        pool = self._route(category)
        async with self._acquire(pool, category) as grant:
            cfg = dict(overrides)
            if grant.slot is not None:
                cfg["id_slot"] = grant.slot
            self._log_dispatch(category, grant)
            result = await asyncio.to_thread(grant.lane.spec.generate, messages, cfg, category.is_main)
        self._log_result(category, grant, messages, overrides, result)
        return result

    # ------------------------------------------------------------------ 日志
    @staticmethod
    def _who(category: sessionCategory) -> str:
        kind = "主会话" if category.is_main else f"支路Lv{category.importance_lv}"
        return f"{kind}#{category.session_id or '-'}·{'延长' if category.extended else '新会话'}"

    def _log_dispatch(self, category: sessionCategory, grant: lease) -> None:
        model = grant.lane.spec.model_id
        if grant.prev_bound == model:
            how = "按指纹绑定"
        elif grant.prev_bound:
            how = f"原绑定 {grant.prev_bound} 已不在池，重新挑选"
        else:
            how = "按空闲度挑选"
        info("[调度]", self._who(category), f"-> {'主池' if grant.isMain else '副池'}:{model}", how,
             f"排队{grant.waited:.1f}s(前方{grant.ahead})", f"原因={category.reason}",
             "注意: 支路占用主池模型，会打断主会话" if grant.isMain and not category.is_main else "")

    def _log_result(self, category: sessionCategory, grant: lease, messages: list[dict[str, Any]],
                    overrides: dict[str, Any], result: GenerationResult) -> None:
        """每次请求完成后打印生成指标与缓存结果；未命中/部分命中用 warn 突出，并给出可能原因。"""
        model = grant.lane.spec.model_id
        info(f"[生成] {model} 速率: {result.tokens_per_second:.2f} token/s, 生成: {result.tokens_generated} tokens, "
             f"耗时: {result.time_seconds:.2f}s, 剩余 KV Cache: {result.kv_remaining} tokens")

        verdict, ratio = cache_verdict(result.cached_tokens, result.prompt_tokens)
        dropped = self._classifier.forget(model, category.session_id) if category.is_main and verdict == "未命中" else 0
        restored_all = result.memo_restored + result.memo_missed
        parts = [
            f"[缓存] {verdict} {ratio:.1%}",
            f"复用 {result.cached_tokens}/{result.prompt_tokens}" if verdict != "未知" else "引擎未提供缓存数据",
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
            if not category.extended:
                hints.append("新会话/压缩后首轮")
            elif not hints:
                hints.append("同会话历史被客户端改写，对比 debug/prompt diff")
            parts.append("可能原因: " + "；".join(hints))
        if dropped or result.memo_pruned:
            parts.append(f"已清理 会话指纹={dropped} 思考记忆={result.memo_pruned}")
        (info if verdict in {"命中", "未知"} else warn)(*parts)
