from __future__ import annotations

import asyncio
import hashlib
import json
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import adim_ID
from .loader.models_mgr import ModelsMgr
from .utils.common import error, info, warn
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
    def __init__(self, manager: ModelsMgr, max_concurrency: int = 32) -> None:
        self.manager = manager
        self._queue: asyncio.Queue[_TaskItem] | None = None
        self._worker_task: asyncio.Task | None = None
        self._running_tasks: set[_TaskItem] = set()
        self._prev_req: dict[str, Any] = {}
        # 允许并发将请求转发至已拉起的模型服务
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _cache_summary(self, task: _TaskItem, tools: list, result: Any, diag: dict[str, Any]) -> None:
        tools_hash = hashlib.md5(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:6]
        msgs = task.messages
        last = msgs[-1] if msgs else {}
        text = last.get("content")
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        prev, self._prev_req = self._prev_req, {"tools": tools_hash, "n": len(msgs), "source": task.source}

        verdict = diag.get("verdict", "未知")
        hint = ""
        if verdict == "分叉":
            if prev and prev["tools"] != tools_hash:
                hint = "工具集与上次不同→疑似旁路请求/换会话覆盖KV"
            elif prev and len(msgs) <= prev["n"]:
                hint = "消息数未增加→新会话/重试/回溯"
            else:
                hint = "同会话历史被改写→对比 debug/prompt diff"
        kv = {"延长": "严格延长", "首轮": "首轮", "分叉": f"分叉@{diag.get('idx')}(回退{diag.get('rollback')})"}.get(verdict, verdict)
        cached = result.cached_tokens
        info("[缓存摘要]", f"来源={task.source}", f"消息={len(msgs)}", f"末条={last.get('role')}",
             f"工具={len(tools)}#{tools_hash}", f"复用={cached}/{result.prompt_tokens}", f"KV={kv}",
             f"文件={diag.get('file', '-')}", f"提示={hint or '-'}", "| 末条开头:", repr((text or "")[:50]))

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
            "running_count": len(self._running_tasks),
            "running_task_ids": [t.task_id for t in self._running_tasks],
        }

    async def _worker_loop(self) -> None:
        while True:
            task = await self._queue.get()
            if task.future.cancelled():
                self._queue.task_done()
                continue

            # 使用任务池并发分发，不再单任务串行阻塞
            asyncio.create_task(self._execute_task(task))

    async def _execute_task(self, task: _TaskItem) -> None:
        self._running_tasks.add(task)
        async with self._semaphore:
            try:
                info("任务开始执行", task.source, task.task_id, "model=", task.model)
                job = asyncio.ensure_future(self._dispatch(task))
                wait_s = task.timeout if task.timeout and task.timeout > 0 else None
                done, _ = await asyncio.wait({job}, timeout=wait_s)
                if not done:
                    if not task.future.cancelled():
                        task.future.set_exception(TimeoutError(f"任务超时 ({task.timeout}s)"))
                    warn("任务执行超时", task.task_id)
                elif not task.future.cancelled():
                    task.future.set_result(job.result())
            except Exception as exc:
                error("[任务异常]", task.source, task.task_id, "model=", task.model, f"{type(exc).__name__}: {exc}",
                      "\n" + "".join(traceback.format_exception(exc)))
                if not task.future.cancelled():
                    task.future.set_exception(exc)
            finally:
                self._running_tasks.discard(task)
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

    async def handle(self, message_id: int, data: Any = None, *, source: str = "console") -> dict[str, Any]:
        """Console 入口适配:按 message_id 分发到 admin()/chat(),业务异常转为统一 error 响应。"""
        payload = data if isinstance(data, dict) else {}
        seq = data if isinstance(data, (list, tuple)) else []
        model = str(payload.get("model") or (seq[0] if seq else (data if isinstance(data, str) else "")))
        try:
            if message_id in {e.value for e in adim_ID}:
                return await self.admin(AdminRequest(message_id, model, data), source=source)
            prompt = str(payload.get("prompt") or (seq[1] if len(seq) > 1 else ""))
            return await self.chat(ChatRequest(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                deploy=dict(payload.get("deploy", {})),
            ), source=source)
        except (KeyError, RuntimeError, MemoryError, ValueError, TimeoutError) as exc:
            warn("消息处理失败:", exc)
            return self._response(model, message_id, "error", str(exc))

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

        # 1. 确保模型服务就绪
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

        first = task.messages[0] if task.messages else {}
        info("[DBG] 送入引擎", model, f"共{len(task.messages)}条", "system[:120]=", repr(first)[:120],
             "hash(首条)=", hashlib.md5(json.dumps(first, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8],
             "hash(去掉末条)=", hashlib.md5(json.dumps(task.messages[:-1], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8])

        tools = full_gen.get("tools") or []
        info("[DBG] 请求参数", model, f"tools数={len(tools)}",
             "tools_hash=", hashlib.md5(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8],
             "角色序列=", [m.get("role") for m in task.messages], "tool_choice=", full_gen.get("tool_choice"),
             "max_tokens=", full_gen.get("max_tokens"), "system_prompt长度=", len(str(full_gen.get("system_prompt") or "")))

        # 3. 调度生成（底层通过 HTTP 转发给对应 Server 实例）
        result = await asyncio.to_thread(self.manager.generate, model, task.messages, full_gen)

        # 4. 指标统计
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

        self._cache_summary(task, tools, result, getattr(loader, "last_diag", {}))

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