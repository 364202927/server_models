from __future__ import annotations

import asyncio
import hashlib
import json
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import adim_ID
from .loader.model_spec import modelSpec
from .loader.models_mgr import models_mgr
from .loader.sessionHandle import pipeline
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


@dataclass(eq=False)  # 要放进 set，必须按身份哈希
class _taskItem:
    message_id: int
    model: str
    messages: list[dict[str, Any]]
    deploy: dict[str, Any]
    timeout: float | None = 180.0
    source: str = "unknown"
    client: str = "other"      # 消息来源归类：claude_code / codex / … / other
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    future: asyncio.Future | None = None


class msgHandler:
    def __init__(self, max_concurrency: int = 32) -> None:
        self._queue: asyncio.Queue[_taskItem] | None = None
        self._worker_task: asyncio.Task | None = None
        self._running_tasks: set[_taskItem] = set()
        # 允许并发将请求转发至已拉起的模型服务
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # 会话分级 + 排队调度；模型由本类创建后通过 addModel 登记，pipeline 不持有 manager
        self.pipeline = pipeline()
        self._create_lock = asyncio.Lock()   # 创建/切换模型串行化，避免并发请求重复加载或互相替换

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
            "lanes": self.pipeline.stats(),
        }

    async def _worker_loop(self) -> None:
        while True:
            task = await self._queue.get()
            if task.future.cancelled():
                self._queue.task_done()
                continue

            # 使用任务池并发分发，不再单任务串行阻塞
            asyncio.create_task(self._execute_task(task))

    async def _execute_task(self, task: _taskItem) -> None:
        self._running_tasks.add(task)
        async with self._semaphore:
            try:
                info("任务开始执行", task.source, task.task_id, "model=", task.model)
                job = asyncio.ensure_future(self._dispatch(task))
                wait_s = task.timeout if task.timeout and task.timeout > 0 else None
                done, _ = await asyncio.wait({job}, timeout=wait_s)
                if not done:
                    job.cancel()  # 让仍在排队的任务退出通道，不再占着名额
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

    async def chat(self, request: ChatRequest, *, source: str = "unknown", client: str = "other") -> dict[str, Any]:
        spec = models_mgr.getModel(request.model)
        # 消息入口：来源归类不是已知 agent 时用 warn，方便核对 UA 关键词表
        (warn if client == "other" else info)(
            "[入口]", f"model={request.model}", f"来源={source}", f"归类={client}",
            f"models.json={'命中' if spec else '未命中'}", self.pipeline.describe())
        return await self._enqueue(_taskItem(
            message_id=0, model=request.model, messages=request.messages, deploy=request.deploy,
            timeout=request.timeout, source=source, client=client,
        ))

    async def admin(self, request: AdminRequest, *, source: str = "unknown") -> dict[str, Any]:
        admin_ids = {e.value for e in adim_ID}
        if request.message_id not in admin_ids:
            raise ValueError(f"不支持的管理指令: {request.message_id}")

        if request.message_id == adim_ID.eStatus.value:
            return self._response(request.model, adim_ID.eStatus.value, "ok", models_mgr.status() | {"queue_length": self.pending})
        if request.message_id == adim_ID.eHardwareInfo.value:
            return self._response(request.model, adim_ID.eHardwareInfo.value, "ok", detect_hardware().to_dict())
        if request.message_id == adim_ID.eModelsList.value:
            return self._response(request.model, adim_ID.eModelsList.value, "ok", {"models": list(models_mgr.specs)})

        return await self._enqueue(_taskItem(
            message_id=request.message_id, model=request.model, messages=[],
            deploy=request.args if isinstance(request.args, dict) else {}, timeout=180.0, source=source,
        ))

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

    async def _enqueue(self, task: _taskItem) -> dict[str, Any]:
        self._ensure_worker()
        task.future = asyncio.get_running_loop().create_future()
        await self._queue.put(task)
        return await task.future

    # ------------------------------------------------------------------ 按 id 调用模型的统一入口
    def _spec(self, model_id: str) -> modelSpec:
        spec = models_mgr.getModel(model_id)
        if spec is None:
            raise KeyError(f"models.json 中不存在模型: {model_id}")
        return spec

    async def _createModel(self, ids: list[str], args: dict[str, Any], isMain: bool) -> str:
        """加载 ids 并登记到 pipeline（整体替换对应池）。chat 与管理指令共用；成功返回空串，失败返回原因。"""
        async with self._create_lock:
            specs = [models_mgr.getModel(model_id) for model_id in ids]
            if None in specs:
                return f"模型不存在: {[i for i, s in zip(ids, specs) if s is None]}"
            register = not all(self.pipeline.has(spec) for spec in specs)
            if register and self.pipeline.busy():
                return f"pipeline 仍有任务在执行，暂不能切换{'主' if isMain else '副'}模型到 {ids}"
            try:
                for spec in specs:
                    await asyncio.to_thread(spec.ensure, args)
            except (RuntimeError, MemoryError) as exc:
                return str(exc)
            if register and not self.pipeline.addModel(isMain, specs):
                return "加载期间 pipeline 被占用，请稍后重试"
            return ""

    def _fail(self, task: _taskItem, code: int, reason: str) -> dict[str, Any]:
        error("[请求失败]", task.source, task.task_id, f"model={task.model}", f"code={code}", reason)
        return self._response(task.model, task.message_id, "error", reason) | {"code": code}

    async def _dispatch(self, task: _taskItem) -> dict[str, Any]:
        model = task.model
        msg_id = task.message_id

        if msg_id in {adim_ID.eSleep.value, adim_ID.eUnload.value}:
            spec = self._spec(model)
            ok = await asyncio.to_thread(spec.sleep if msg_id == adim_ID.eSleep.value else spec.unload)
            return self._response(model, msg_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")
        if msg_id == adim_ID.eEnsureLoaded.value:
            reason = await self._createModel([model], task.deploy, not self.pipeline.hasMain())
            if reason:
                return self._fail(task, 503, reason)
            return self._response(model, msg_id, "ok", "操作成功")

        return await self._generate_chat(task)

    async def _generate_chat(self, task: _taskItem) -> dict[str, Any]:
        """models.json 里没有的 id：非 agent 来源或没有主模型直接报错，agent 来源且已有主模型则当作客户端别名直接分级调度；
        有的 id：已在池里就直接调度，否则先创建并登记（没有主模型或本次是主会话时登记为主池，否则登记为副池）。"""
        args = dict(task.deploy)
        spec = models_mgr.getModel(task.model)
        if spec is None and (task.client == "other" or not self.pipeline.hasMain()):
            return self._fail(task, 404, f"模型不存在: {task.model}（来源归类={task.client}）")

        category = self.pipeline.classify(task.messages, args.get("tools"), args.get("max_tokens"), task.model)
        if spec is not None and not (self.pipeline.has(spec) and self.pipeline.hasMain()):
            reason = await self._createModel([task.model], args, category.is_main or not self.pipeline.hasMain())
            if reason:
                return self._fail(task, 503, reason)

        first = task.messages[0] if task.messages else {}
        tools = args.get("tools") or []
        info("[DBG] 送入引擎", task.model, f"共{len(task.messages)}条", "system[:120]=", repr(first)[:120],
             "hash(首条)=", hashlib.md5(json.dumps(first, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8],
             "hash(去掉末条)=", hashlib.md5(json.dumps(task.messages[:-1], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8])
        info("[DBG] 请求参数", task.model, f"tools数={len(tools)}",
             "tools_hash=", hashlib.md5(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8],
             "角色序列=", [m.get("role") for m in task.messages], "tool_choice=", args.get("tool_choice"),
             "max_tokens=", args.get("max_tokens"))

        try:
            result = await self.pipeline.run(category, task.messages, args)
        except (RuntimeError, MemoryError, httpx.HTTPError) as exc:   # 模型/引擎失败：不回落，直接报给客户端
            return self._fail(task, 502, f"{type(exc).__name__}: {exc}")

        return self._response(task.model, 0, "ok", result.text) | {
            "tool_calls": result.tool_calls,
            "finish_reason": result.finish_reason,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "cached_tokens": result.cached_tokens,
                "time_seconds": result.time_seconds,
                "tokens_per_second": round(result.tokens_per_second, 2),
                "kv_cache_remaining": result.kv_remaining,
            },
        }