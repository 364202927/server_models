from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import adim_ID
from .loader.models_mgr import ModelsMgr
from .loader.chatDataFilter import chatDataFilter
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
                if task.timeout and task.timeout > 0:
                    result = await asyncio.wait_for(self._dispatch(task), timeout=task.timeout)
                else:
                    result = await self._dispatch(task)

                if not task.future.cancelled():
                    task.future.set_result(result)
            except asyncio.TimeoutError:
                if not task.future.cancelled():
                    task.future.set_exception(TimeoutError(f"任务超时 ({task.timeout}s)"))
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

    # async def _generate_chat(self, task: _TaskItem) -> dict[str, Any]:
    #     model = task.model
    #     incoming_args = dict(task.deploy)

    #     # 1. 确保模型就绪 (支持加载时覆写已存在字段)
    #     await asyncio.to_thread(self.manager.ensure_loaded, model, incoming_args)

    #     # 2. 闭包参数生成：只有在完整配置中存在的键才会被覆盖
    #     full_gen = self.manager.get_full_generation_config(model)
    #     for k, v in incoming_args.items():
    #         if k in full_gen and v is not None:
    #             full_gen[k] = v

    #     # 提取特殊参数
    #     if "tools" in incoming_args:
    #         full_gen["tools"] = incoming_args["tools"]
    #     if "tool_choice" in incoming_args:
    #         full_gen["tool_choice"] = incoming_args["tool_choice"]

    #     # 3. 调度生成
    #     result = await asyncio.to_thread(self.manager.generate, model, task.messages, full_gen)

    #     final_text = chatDataFilter.repair_think_tags(result.text)
    #     return self._response(model, 0, "ok", final_text) | {
    #         "tool_calls": result.tool_calls,
    #         "finish_reason": result.finish_reason,
    #         "usage": {
    #             "prompt_tokens": result.prompt_tokens,
    #             "completion_tokens": result.tokens_generated,
    #             "time_seconds": result.time_seconds,
    #             "tokens_per_second": result.tokens_per_second,
    #         },
    #     }

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

        # 3. [功能 1] 生成前预估 Token
        runtime = self.manager.runtime.get(model)
        req_max = full_gen.get("max_tokens")
        metrics = runtime.loader.estimate_genTokens(task.messages)
        info(f"[{model}] 预估输入: 总计 {metrics['total_prompt_tokens']} tokens | "
            f"缓存复用 {metrics['cached_tokens']} tokens ({metrics['hit_rate_pct']}%) | "
            f"本轮实际新增(Delta) {metrics['delta_tokens']} tokens | "
            f"可用生成空间: {metrics['available_generation_tokens']} tokens")

        # 4. 调度生成
        result = await asyncio.to_thread(self.manager.generate, model, task.messages, full_gen)

        # 5. [功能 2 & 3] 计算生成速率 (token/s) 与 剩余 KV Cache
        speed = result.tokens_per_second
        total_used = result.prompt_tokens + result.tokens_generated
        
        remaining_kv = 0
        remaining_kv = runtime.loader.remaining_kvCache(total_used)

        info(f"[{model}] 生成完成 -> 速率: {speed:.2f} token/s, "
            f"生成: {result.tokens_generated} tokens, "
            f"耗时: {result.time_seconds:.2f}s, "
            f"剩余 KV Cache: {remaining_kv} tokens")

        final_text = chatDataFilter.repair_think_tags(result.text)
        return self._response(model, 0, "ok", final_text) | {
            "tool_calls": result.tool_calls,
            "finish_reason": result.finish_reason,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "time_seconds": result.time_seconds,
                "tokens_per_second": round(speed, 2),
                "kv_cache_remaining": remaining_kv,
            },
        }