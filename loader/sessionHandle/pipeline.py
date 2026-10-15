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
    used: set[int] = field(default_factory=set)  # 已被占用的槽位号：主会话占 0，支路占 1（或副模型上任一空槽）
    last_sid: dict[int, str] = field(default_factory=dict)  # 槽位号 -> 上一个在该槽位上跑的会话，用于判断 KV 是否被其他会话覆盖


@dataclass(eq=False)
class _ticket:
    category: sessionCategory
    want: int | None  # 要占的槽位号；None = 任一空槽
    seq: int
    enq_at: float
    granted: asyncio.Future  # 放行后结果为 (分到的 _lane, 槽位号)


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
    isMain: bool  # 是否来自主池
    waited: float = 0.0  # 排队等待秒数
    ahead: int = 0  # 入队时池里已有的任务数
    prev_sid: str = ""  # 该槽位上一个会话
    prev_bound: str = ""  # 入队前会话绑定的模型
    slot: int | None = None  # 传给引擎的 id_slot：仅主池多槽位时给；None 交给引擎自选
    warnings: list[str] = field(default_factory=list)  # 路由告警，随响应带给客户端
    job: asyncio.Future | None = None  # 正在跑的生成线程；调用方被取消时，名额等它结束才归还
    cancel: threading.Event = field(default_factory=threading.Event)  # 调用方离开（超时/断连）时置位，通知引擎中止


class pipeline:
    """
    会话调度器：只拿 modelSpec 的 generate，不管模型怎么生、怎么睡。
    - 主池/副池结构相同，都可放多个模型；addModel 整体替换，池里还有任务时拒绝切换（避免打断正在跑的会话）。
    - 路由只分主/支：主会话占主模型 slot 0；支路（不分级）走副模型，没有副模型则占主模型 slot 1；
      两者都没有（主模型 parallel_slots<2）时告警，并被迫占 slot 0（排在主会话之后）。
    - 主会话按指纹绑定模型：绑定的模型即使忙也排队等它，别的模型空闲也不接；没有绑定才按空闲度挑。
    - 排队：主会话最先，支路按到达顺序，运行中的任务不抢占。
    - 切换：ensure 负责"加载 + 登记"，整体串行（_switch_lock）；池里有任务时拒绝替换。调用方不需要自己加锁。
      加载期间关闸（_idle）：新请求先 settled() 等切换结束，再按最新的池状态决策。
    - 线程不可强杀：超时/断连时先置位 cancel 事件通知引擎中止；无论引擎是否响应，槽位、切换锁都等线程真正结束才归还。
    """

    def __init__(self) -> None:
        self._classifier = classifier()
        self._warned: set[str] = set()  # 已在服务端告警过的支路会话，同一会话只报一次，避免子代理刷屏
        self._seq = itertools.count()
        self._main = _pool(True)
        self._sub = _pool(False)
        self._switch_lock = asyncio.Lock()  # 加载/切换模型串行化，避免并发请求重复加载或互相替换
        self._idle = asyncio.Event()  # 置位 = 没有模型正在加载/切换；清除期间新请求在 settled() 处等待
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
        """整个 pipeline 只要还有任务在跑或在排队就算忙。"""
        return bool(self.running or self.pending)

    def addModel(self, isMain: bool, models: list[modelSpec]) -> bool:
        """用 models 整体替换主池/副池（传 [] 即清空）。pipeline 忙时不切换，返回 False。"""
        if self.busy():
            return False
        pool, other = (self._main, self._sub) if isMain else (self._sub, self._main)
        unique = list({spec.model_id: spec for spec in models}.values())
        removed = {lane.spec.model_id for lane in pool.lanes} - {s.model_id for s in unique} - {l.spec.model_id for l in other.lanes}
        pool.lanes = [_lane(spec, max(1, int(spec.full_load().get("parallel_slots", 1)))) for spec in unique]
        cleared = self._classifier.unbind(removed)
        info(
            "[调度] 登记",
            "主池" if isMain else "副池",
            [s.model_id for s in unique],
            f"解除会话绑定 {cleared} 个" if cleared else "",
        )
        return True

    async def settled(self) -> None:
        """等正在进行的模型加载/切换结束。调用方应在等完之后再用 has/hasMain 判断，避免拿过期状态决策。"""
        await self._idle.wait()

    async def ensure(self, specs: list[modelSpec], isMain: bool, args: dict[str, Any]) -> None:
        """加载 specs 并登记为主/副池（整体替换）。失败抛 RuntimeError / MemoryError，由入口层统一转响应。
        - 加载与切换整体串行，并在加载期间关闸：新请求在 settled() 处等待，不会撞上半截状态。
        - specs 不在池里而 pipeline 又忙时拒绝切换，避免打断正在跑的会话。
        - 调用方被取消不会中断加载线程：锁与闸门等线程结束才释放，避免同一模型被重复加载。"""
        await self._switch_lock.acquire()
        job = None
        try:
            register = not all(self.has(spec) for spec in specs)
            if register and self.busy():
                raise RuntimeError(f"pipeline 仍有任务在执行，暂不能切换{'主' if isMain else '副'}模型到 {[s.model_id for s in specs]}")
            self._idle.clear()
            job = asyncio.ensure_future(self._load(specs, args))
            await asyncio.shield(job)
            if register and not self.addModel(isMain, specs):  # 有闸门时不会发生；防御绕过 settled() 的调用方
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
        """线程任务 job 结束后再调 release；没有 job 或已结束则立即调。
        调用方被取消（超时/断连）时线程仍在跑，资源必须等线程真正结束才能归还。"""
        if job is None or job.done():
            release()
            return

        def _on_done(done: asyncio.Future) -> None:
            if not done.cancelled():
                done.exception()  # shield 在调用方取消后不再取走异常，这里取走，避免 "never retrieved" 告警
            release()

        job.add_done_callback(_on_done)

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
        """返回 (池, 要占的槽位号, 告警)。主会话占主模型 slot 0；支路优先副模型，其次主模型 slot 1，都没有则告警并退到 slot 0。"""
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
        """挑 (模型, 槽位)：主会话优先它绑定的模型（忙就等，不换别的）；want 为 None 时取任一空槽，多个模型取负载最低的。"""
        category, want = ticket.category, ticket.want
        lanes = pool.alive
        bound = [lane for lane in lanes if category.is_main and lane.spec.model_id == category.model_id]
        free: list[tuple[_lane, int]] = []
        for lane in bound or lanes:
            slot = want if want is not None else next((i for i in range(lane.slots) if i not in lane.used), None)
            if slot is not None and slot < lane.slots and slot not in lane.used:
                free.append((lane, slot))
        return min(free, key=lambda item: item[0].running / item[0].slots, default=None)

    def _pump(self, pool: _pool) -> None:
        """按优先级放行等待者；某张票要的槽位被占时跳过它，不挡住后面能放行的票。"""
        for ticket in sorted(pool.waiting, key=self._rank):
            if ticket.granted.done():  # 等待方已被取消
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
        lane.running -= 1
        lane.used.discard(slot)
        self._pump(pool)

    @asynccontextmanager
    async def _acquire(self, pool: _pool, category: sessionCategory, want: int | None, warning: str = "") -> AsyncIterator[lease]:
        """申请执行许可。退出（含取消/超时/异常）时退出队列或归还名额并唤醒后续任务；
        生成线程还没结束（被超时取消）时，名额推迟到线程结束再归还，避免实际并发超过模型槽位数。"""
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
                slot if pool.isMain and lane.slots >= 2 else None,  # 副模型不固定槽位，交给引擎自选
                [warning] if warning else [],
            )
            yield grant
        finally:
            if ticket in pool.waiting:  # 还没放行就被取消：退出队列
                pool.waiting.remove(ticket)
                self._pump(pool)
            elif not ticket.granted.cancelled():  # 已放行（含放行后、恢复前被取消）：归还名额
                granted_lane, granted_slot = ticket.granted.result()
                job = grant.job if grant else None
                if job is not None and not job.done():
                    grant.cancel.set()  # 调用方先走了：通知引擎中止，名额等线程结束再还
                self._after(job, lambda: self._release(pool, granted_lane, granted_slot))

    # ------------------------------------------------------------------ 执行
    async def run(self, category: sessionCategory, messages: list[dict[str, Any]], overrides: dict[str, Any]) -> GenerationResult:
        """排队 -> 在分到的模型上生成。失败（含副模型）原样上抛，由调用方报给客户端。"""
        pool, want, warning = self._route(category)
        if warning and category.session_id not in self._warned:
            self._warned.add(category.session_id)
            warn("[调度]", self._who(category), warning)
        async with self._acquire(pool, category, want, warning) as grant:
            cfg = dict(overrides)
            if grant.slot is not None:
                cfg["id_slot"] = grant.slot
            self._log_dispatch(category, grant)
            grant.job = asyncio.ensure_future(
                asyncio.to_thread(
                    grant.lane.spec.generate,
                    messages,
                    cfg,
                    category.is_main,
                    grant.cancel,
                )
            )
            result = await asyncio.shield(grant.job)  # 调用方被取消时 job 继续跑，名额由 _acquire 延后归还
        result.warnings = grant.warnings
        self._log_result(category, grant, messages, overrides, result)
        return result

    # ------------------------------------------------------------------ 日志
    @staticmethod
    def _who(category: sessionCategory) -> str:
        kind = "主会话" if category.is_main else "支路"
        return f"{kind}#{category.session_id or '-'}·{'延长' if category.extended else '新会话'}"

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
        """每次请求完成后打印生成指标与缓存结果；未命中/部分命中用 warn 突出，并给出可能原因。"""
        model = grant.lane.spec.model_id
        info(f"[生成] {model} 速率: {result.tokens_per_second:.2f} token/s, 生成: {result.tokens_generated} tokens(思考 {len(result.reasoning)} 字, 工具调用 {len(result.tool_calls)} 个), " f"耗时: {result.time_seconds:.2f}s, 剩余 KV Cache: {result.kv_remaining} tokens")

        verdict, ratio = cache_verdict(result.cached_tokens, result.prompt_tokens)
        dropped = 0  # 临时注释：先确认引擎能否正确缓存，再做清理逻辑
        # dropped = self._classifier.forget(model, category.session_id) if category.is_main and verdict == "未命中" else 0
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
            if not category.extended:
                hints.append("新会话/压缩后首轮")
            elif not hints:
                hints.append("同会话历史被客户端改写，对比 debug/prompt diff")
            parts.append("可能原因: " + "；".join(hints))
        if dropped or result.memo_pruned:
            parts.append(f"已清理 会话指纹={dropped} 思考记忆={result.memo_pruned}")
        (info if verdict in {"命中", "未知"} else warn)(*parts)
