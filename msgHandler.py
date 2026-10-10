from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import adim_ID
from .loader.models_mgr import ModelsMgr
from .utils.common import info, warn
from .utils.hardware import detect_hardware


@dataclass
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    deploy: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = 180.0


@dataclass
class AdminRequest:
    message_id: int
    model: str = ""
    args: Any = None


@dataclass
class _TaskItem:
    task_id: str
    message_id: int
    model: str
    messages: list[dict[str, Any]]
    deploy: dict[str, Any]
    timeout: float | None
    source: str
    future: asyncio.Future = field(default_factory=asyncio.Future)


class MsgHandler:
    def __init__(self, manager: ModelsMgr) -> None:
        self.manager = manager
        self._queue: asyncio.Queue[_TaskItem] | None = None
        self._worker_task: asyncio.Task | None = None
        self._current_task: _TaskItem | None = None

    def _ensure_worker(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker_loop())

    @property
    def pending(self) -> int:
        return self._queue.qsize() if self._queue else 0

    def queue_info(self) -> dict[str, Any]:
        return {
            "pending_count": self.pending,
            "current_task_id": self._current_task.task_id if self._current_task else None,
            "current_model": self._current_task.model if self._current_task else None,
        }

    async def _worker_loop(self) -> None:
        while True:
            task = await self._queue.get()
            if task.future.cancelled():
                self._queue.task_done()
                continue

            self._current_task = task
            try:
                info("队列任务开始执行", task.source, task.task_id, "model=", task.model)
                job = asyncio.ensure_future(self._dispatch(task))
                wait_s = task.timeout if task.timeout and task.timeout > 0 else None
                done, _ = await asyncio.wait({job}, timeout=wait_s)
                if not done:
                    # 先通知调用方超时，但 to_thread 里的线程无法中断；
                    # 必须等它真正结束再放行下一个任务，否则会并发操作同一个非线程安全的引擎对象
                    if not task.future.cancelled():
                        task.future.set_exception(TimeoutError(f"任务超时 ({task.timeout}s)"))
                    warn("任务超时，等待后台推理线程结束后再放行队列", task.task_id)
                    await asyncio.gather(job, return_exceptions=True)
                elif not task.future.cancelled():
                    task.future.set_result(job.result())
            except Exception as exc:
                if not task.future.cancelled():
                    task.future.set_exception(exc)
            finally:
                self._current_task = None
                self._queue.task_done()

    @staticmethod
    def _response(model: str, message_id: int, status: str, value: Any) -> dict[str, Any]:
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return {
            "request_id": str(uuid.uuid4()),
            "status": status,
            "model": model,
            "message_id": message_id,
            "response": payload,
        }

    async def chat(self, request: ChatRequest, *, source: str = "unknown") -> dict[str, Any]:
        return await self._enqueue_task(
            message_id=0,
            model=request.model,
            messages=request.messages,
            deploy=request.deploy,
            timeout=request.timeout,
            source=source,
        )

    async def admin(self, request: AdminRequest, *, source: str = "unknown") -> dict[str, Any]:
        admin_ids = {e.value for e in adim_ID}
        if request.message_id not in admin_ids:
            raise ValueError(f"不支持的管理指令: {request.message_id}")

        if request.message_id == adim_ID.eStatus.value:
            return self._response(request.model, adim_ID.eStatus.value, "ok", self.manager.status() | {"queue_length": self.pending})
        if request.message_id == adim_ID.eHardwareInfo.value:
            return self._response(request.model, adim_ID.eHardwareInfo.value, "ok", detect_hardware().to_dict())
        if request.message_id == adim_ID.eModelsList.value:
            return self._response(request.model, adim_ID.eModelsList.value, "ok", {"models": list(self.manager.specs)})

        return await self._enqueue_task(
            message_id=request.message_id,
            model=request.model,
            messages=[],
            deploy=request.args if isinstance(request.args, dict) else {},
            timeout=180.0,
            source=source,
        )

    async def _enqueue_task(
        self,
        message_id: int,
        model: str,
        messages: list[dict[str, Any]],
        deploy: dict[str, Any],
        timeout: float | None,
        source: str,
    ) -> dict[str, Any]:
        self._ensure_worker()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = _TaskItem(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            message_id=message_id,
            model=model,
            messages=messages,
            deploy=deploy,
            timeout=timeout,
            source=source,
            future=future,
        )
        await self._queue.put(task)
        return await future

    async def _dispatch(self, task: _TaskItem) -> dict[str, Any]:
        model = task.model
        msg_id = task.message_id

        if msg_id in {adim_ID.eSleep.value, adim_ID.eUnload.value}:
            ok = self.manager.sleep(model) if msg_id == adim_ID.eSleep.value else self.manager.unload(model)
            return self._response(model, msg_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")
        if msg_id == adim_ID.eEnsureLoaded.value:
            await asyncio.to_thread(self.manager.ensure_loaded, model, task.deploy)
            return self._response(model, msg_id, "ok", "操作成功")

        return await self._generate_chat(task)

    async def _generate_chat(self, task: _TaskItem) -> dict[str, Any]:
        model = task.model
        incoming_args = dict(task.deploy)

        # 1. 确保模型就绪
        await asyncio.to_thread(self.manager.ensure_loaded, model, incoming_args)

        # 2. 参数合并
        full_gen = self.manager.get_full_generation_config(model)
        for k, v in incoming_args.items():
            if k in full_gen and v is not None:
                full_gen[k] = v

        if "tools" in incoming_args:
            full_gen["tools"] = incoming_args["tools"]
        if "tool_choice" in incoming_args:
            full_gen["tool_choice"] = incoming_args["tool_choice"]

        # [DBG] 临时：system 前 120 字符 + 全部 messages 的前缀 hash，用于核对相邻请求前缀是否逐字一致
        first = task.messages[0] if task.messages else {}
        info("[DBG] 送入引擎", model, f"共{len(task.messages)}条", "system[:120]=", repr(first)[:120],
             "hash(首条)=", hashlib.md5(json.dumps(first, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8],
             "hash(去掉末条)=", hashlib.md5(json.dumps(task.messages[:-1], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8])

        # 3. 调度生成
        result = await asyncio.to_thread(self.manager.generate, model, task.messages, full_gen)

        # 4. 指标取自引擎返回的真值（事后统计，不做生成前估算）
        loader = self.manager.runtime[model].loader
        remaining_kv = loader.remaining_kvCache(result.prompt_tokens + result.tokens_generated)
        cached = result.cached_tokens
        cache_info = (f"缓存复用 {cached}/{result.prompt_tokens} tokens "
                      f"({cached / max(1, result.prompt_tokens) * 100:.1f}%), 本轮重算 {result.prompt_tokens - cached} tokens"
                      if cached >= 0 else "缓存复用: 引擎未提供")
        info(f"[{model}] 生成完成 -> {cache_info} | "
             f"速率: {result.tokens_per_second:.2f} token/s, "
             f"生成: {result.tokens_generated} tokens, "
             f"耗时: {result.time_seconds:.2f}s, "
             f"剩余 KV Cache: {remaining_kv} tokens")

        return self._response(model, 0, "ok", result.text) | {
            "tool_calls": result.tool_calls,
            "finish_reason": result.finish_reason,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "cached_tokens": result.cached_tokens,
                "time_seconds": result.time_seconds,
                "tokens_per_second": round(result.tokens_per_second, 2),
                "kv_cache_remaining": remaining_kv,
            },
        }