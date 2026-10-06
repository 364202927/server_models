from __future__ import annotations

import asyncio, json, uuid
from dataclasses import dataclass, field
from typing import Any

from . import adim_ID
from .loader.model_spec import LOAD_KEYS
from .loader.models_mgr import ModelsMgr
from .loader.chatDataFilter import (
    GENERATION_KEYS,
    chatDataFilter,
    flatten_messages,
    normalize_generation_params,
    resolve_think_level,
)
from .utils.common import info, warn
from .utils.hardware import detect_hardware


@dataclass
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    think: int = 0
    deploy: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    parallel_tool_calls: bool = True
    system_prompt: str = ""
    timeout: float | None = 180.0


@dataclass
class AdminRequest:
    message_id: int
    model: str = ""
    generation: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ChatContext:
    model: str = ""
    prompt: str = ""
    think: int = 0
    deploy: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    parallel_tool_calls: bool = True
    system_prompt: str = ""
    timeout: float | None = 180.0


@dataclass
class _TaskItem:
    task_id: str
    message_id: int
    args: Any
    ctx: _ChatContext
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
            "current_model": self._current_task.ctx.model if self._current_task else None,
        }

    async def _worker_loop(self) -> None:
        while True:
            task = await self._queue.get()
            if task.future.cancelled():
                info("队列任务在执行前已取消，直接跳过", task.task_id)
                self._queue.task_done()
                continue

            self._current_task = task
            try:
                info("队列任务开始执行", task.source, task.task_id, "model=", task.ctx.model)
                timeout = task.ctx.timeout
                if timeout and timeout > 0:
                    result = await asyncio.wait_for(
                        self._dispatch(task.message_id, task.args, task.ctx),
                        timeout=timeout,
                    )
                else:
                    result = await self._dispatch(task.message_id, task.args, task.ctx)

                if not task.future.cancelled():
                    task.future.set_result(result)
            except asyncio.TimeoutError:
                info("队列任务执行超时", task.task_id, f"timeout={task.ctx.timeout}s")
                if not task.future.cancelled():
                    task.future.set_exception(TimeoutError(f"任务执行超时 ({task.ctx.timeout}s)"))
            except Exception as exc:
                info("队列任务执行失败", task.source, task.task_id, exc)
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
        ctx = _ChatContext(
            model=request.model,
            prompt=flatten_messages(request.messages, request.system_prompt),
            think=request.think,
            deploy=request.deploy,
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            system_prompt=request.system_prompt,
            timeout=request.timeout,
        )
        return await self._enqueue_task(0, None, ctx, source)

    async def admin(self, request: AdminRequest, *, source: str = "unknown") -> dict[str, Any]:
        admin_ids = {e.value for e in adim_ID}
        if request.message_id not in admin_ids:
            raise ValueError(f"不支持的管理指令 message_id: {request.message_id}")

        if request.message_id == adim_ID.eStatus.value:
            return self._response(request.model, adim_ID.eStatus.value, "ok", self.manager.status() | {"queue_length": self.pending})
        if request.message_id == adim_ID.eHardwareInfo.value:
            return self._response(request.model, adim_ID.eHardwareInfo.value, "ok", detect_hardware().to_dict())
        if request.message_id == adim_ID.eModelsList.value:
            return self._response(request.model, adim_ID.eModelsList.value, "ok", {"models": list(self.manager.specs)})

        args = {"model": request.model, "generation": request.generation} if request.message_id == adim_ID.eUpdateGeneration.value else None
        return await self._enqueue_task(request.message_id, args, _ChatContext(model=request.model), source)

    async def _enqueue_task(self, message_id: int, args: Any, ctx: _ChatContext, source: str) -> dict[str, Any]:
        self._ensure_worker()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        task = _TaskItem(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            message_id=message_id,
            args=args,
            ctx=ctx,
            source=source,
            future=future,
        )
        await self._queue.put(task)
        return await future

    async def _dispatch(self, message_id: int, args: Any, ctx: _ChatContext) -> dict[str, Any]:
        model = ctx.model
        if message_id in {adim_ID.eSleep.value, adim_ID.eUnload.value}:
            target = model or (args.get("model") if isinstance(args, dict) else str(args or ""))
            ok = self.manager.sleep(target) if message_id == adim_ID.eSleep.value else self.manager.unload(target)
            return self._response(target, message_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")
        if message_id == adim_ID.eEnsureLoaded.value:
            target = model or (args.get("model") if isinstance(args, dict) else str(args or ""))
            await asyncio.to_thread(self.manager.ensure_loaded, target)
            return self._response(target, adim_ID.eEnsureLoaded.value, "ok", "操作成功")
        if message_id == adim_ID.eUpdateGeneration.value:
            payload = args if isinstance(args, dict) else {}
            target = model or str(payload.get("model", ""))
            changes = {k: v for k, v in dict(payload.get("generation", payload.get("deploy", {}))).items() if k in GENERATION_KEYS}
            merged = await asyncio.to_thread(self.manager.update_generation, target, changes)
            return self._response(target, adim_ID.eUpdateGeneration.value, "ok", merged)

        payload = args if isinstance(args, dict) else {}
        model = model or str(payload.get("model", ""))
        prompt = ctx.prompt or str(payload.get("prompt", args if isinstance(args, str) else ""))
        return await self._generate_chat(model, prompt, payload, ctx)

    async def _generate_chat(self, model: str, prompt: str, payload: dict[str, Any], ctx: _ChatContext) -> dict[str, Any]:
        deploy = {**dict(payload.get("deploy", {})), **ctx.deploy}
        changes = {k: v for k, v in deploy.items() if k in LOAD_KEYS}
        if changes:
            await asyncio.to_thread(self.manager.reconfigure, model, changes)

        params = self.manager.generation_params(model)
        params.update({k: v for k, v in deploy.items() if k in GENERATION_KEYS})
        sampling = {k: v for k, v in params.items() if k in GENERATION_KEYS}
        if "max_tokens" in sampling:
            sampling["max_new_tokens"] = sampling.pop("max_tokens")

        runtime = await asyncio.to_thread(self.manager.ensure_loaded, model)
        supported_modalities = runtime.loader.supported_modalities if runtime.loader else {"text"}

        think_prompt = f"请以推理等级 {ctx.think}/5 分析后给出最终答案。" if ctx.think else ""
        default_system = str(sampling.pop("system_prompt", "") or "")

        cleaned_messages = chatDataFilter.preprocess_messages( messages=ctx.messages,
                                                        supported_modalities=supported_modalities,
                                                        default_system=default_system,
                                                        user_prompt=prompt,
                                                        think_instruction=think_prompt)

        info("[MsgHandler调度]",
             f"model={model}",
             f" ctx_tools_len={len(ctx.tools)}",
             f" tool_choice={ctx.tool_choice}")
        
        result = await asyncio.to_thread(self.manager.generate,
                                        model,
                                        prompt,
                                        sampling=sampling,
                                        messages=cleaned_messages,
                                        tools=ctx.tools or None,
                                        tool_choice=ctx.tool_choice,
                                        parallel_tool_calls=ctx.parallel_tool_calls)
        warn("[MsgHandler完成]",
             f" finish_reason={result.finish_reason}",
             f" tool_calls_len={len(result.tool_calls)}",
             f" response={result}")
            #  f" response_prefix={result.text if result.text else ''}")
        
        final_text = chatDataFilter.repair_think_tags(result.text)

        # 判定工具链闭环，沉淀持久化瘦身工具列表
        invoked_tools = chatDataFilter.extract_invoked_tool_names(cleaned_messages, final_text)
        pruned_tools = chatDataFilter.prune_tools_for_storage(ctx.tools, invoked_tools)

        return self._response(model, 0, "ok", final_text) | {
            "tool_calls": result.tool_calls,
            "finish_reason": result.finish_reason,
            "pruned_tools": pruned_tools,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "time_seconds": result.time_seconds,
                "tokens_per_second": result.tokens_per_second,
            },
        }


msgHandler = MsgHandler
__all__ = [
    "GENERATION_KEYS", "ChatRequest", "AdminRequest", "MsgHandler", "msgHandler",
    "resolve_think_level", "normalize_generation_params", "flatten_messages",
]