from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from . import adim_ID
from .loader.model_spec import modelSpec
from .loader.models_mgr import models_mgr
from .loader.sessionHandle import pipeline
from .utils.common import error, info, warn
from .utils.hardware import detect_hardware

_ADMIN_IDS = {e.value for e in adim_ID}


def _digest(obj: Any) -> str:
    return hashlib.md5(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8]


@dataclass
class chatReq:
    model: str
    messages: list[dict[str, Any]]
    deploy: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = 240.0


@dataclass
class adminReq:
    message_id: int
    model: str = ""
    args: Any = None


@dataclass
class _taskItem:
    """一次请求的上下文：入口处解析一次，之后各层只读它。"""

    message_id: int  # 0 = chat，其余为管理指令
    model: str
    spec: modelSpec | None  # models.json 里的定义；None = 未登记（agent 别名或不存在）
    messages: list[dict[str, Any]] = field(default_factory=list)
    deploy: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = 180.0  # 仅 chat 的「进入通道后」使用：排队 + 生成，不含模型加载/唤醒
    source: str = "unknown"
    client: str = "other"  # 消息来源归类：claude_code / codex / … / other
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    fail_code: int = 503  # 模型/引擎失败时的响应码：准备模型阶段 503，开始生成后改 502


_opFn = Callable[[_taskItem], Awaitable[dict[str, Any]]]


class msgHandler:
    """协议无关的消息入口：解析请求 -> 确保模型就绪 -> 交给 pipeline -> 组装响应。

    排队、并发上限、模型切换互斥全部由 pipeline 负责，这里不持有队列 / 信号量 / 锁。
    异常边界只有 _run 一处：业务失败（超时 / 模型加载或引擎出错）在那里统一转成带 code 的 error 响应，
    其余层级一律不写 try/except，直接 return / raise。

    调用链（每层只做一件事）：
        handle()  控制台适配（只解析 data）
          └─ chat() / admin()   解析请求 -> _taskItem
               └─ _run()        唯一的异常边界
                    └─ _chat() / _model_ops[id]()   具体业务
    """

    def __init__(self) -> None:
        self.pipeline = pipeline()
        # 只读指令：同步返回，不计超时
        self._info_ops: dict[int, Callable[[], Any]] = {
            adim_ID.eStatus.value: lambda: models_mgr.status() | {"queue_length": self.pending},
            adim_ID.eHardwareInfo.value: lambda: detect_hardware().to_dict(),
            adim_ID.eModelsList.value: lambda: {"models": list(models_mgr.specs)},
        }
        # 操作具体模型的指令：入口已保证 task.spec 存在
        self._model_ops: dict[int, _opFn] = {
            adim_ID.eSleep.value: self._sleep_or_unload,
            adim_ID.eUnload.value: self._sleep_or_unload,
            adim_ID.eEnsureLoaded.value: self._ensure_loaded,
        }

    # ------------------------------------------------------------------ 状态
    @property
    def pending(self) -> int:
        return self.pipeline.pending

    def queue_info(self) -> dict[str, Any]:
        return {
            "pending_count": self.pending,
            "running_count": self.pipeline.running,
            "lanes": self.pipeline.stats(),
        }

    # ------------------------------------------------------------------ 入口
    async def chat(self, request: chatReq, *, source: str = "unknown", client: str = "other") -> dict[str, Any]:
        spec = models_mgr.getModel(request.model)
        # 来源归类不是已知 agent 时用 warn，方便核对 UA 关键词表
        (warn if client == "other" else info)("[入口]", f"model={request.model}", f"来源={source}", f"归类={client}", f"models.json={'命中' if spec else '未命中'}", self.pipeline.describe())
        return await self._run(
            _taskItem(
                message_id=0,
                model=request.model,
                spec=spec,
                messages=request.messages,
                deploy=request.deploy,
                timeout=request.timeout,
                source=source,
                client=client,
            ),
            self._chat,
        )

    async def admin(self, request: adminReq, *, source: str = "unknown") -> dict[str, Any]:
        mid, model = request.message_id, request.model
        if mid in self._info_ops:
            return self._response(model, mid, "ok", self._info_ops[mid]())

        task = _taskItem(message_id=mid, model=model, spec=models_mgr.getModel(model), source=source, deploy=request.args if isinstance(request.args, dict) else {})
        if mid not in self._model_ops:
            return self._fail(task, 400, f"不支持的管理指令: {mid}")
        if task.spec is None:
            return self._fail(task, 404, f"模型不存在: {model}")
        return await self._run(task, self._model_ops[mid])

    async def handle(self, message_id: int, data: Any = None, *, source: str = "console") -> dict[str, Any]:
        """Console 入口适配：只负责把 data 解析成 admin()/chat() 的请求；失败由它们返回统一 error 响应。"""
        payload = data if isinstance(data, dict) else {}
        seq = data if isinstance(data, (list, tuple)) else []
        model = str(payload.get("model") or (seq[0] if seq else (data if isinstance(data, str) else "")))
        if message_id in _ADMIN_IDS:
            return await self.admin(adminReq(message_id, model, data), source=source)

        prompt = str(payload.get("prompt") or (seq[1] if len(seq) > 1 else ""))
        return await self.chat(
            chatReq(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                deploy=dict(payload.get("deploy", {})),
            ),
            source=source,
        )

    # ------------------------------------------------------------------ 执行壳
    async def _run(self, task: _taskItem, op: _opFn) -> dict[str, Any]:
        """统一执行壳，也是唯一的异常边界：生成超时 -> 504；模型加载/引擎失败 -> task.fail_code，
        不回落，直接报给客户端。其余异常属于 bug，原样上抛，由框架打印堆栈。
        这里不计时：计时只围住 _chat 里「进入 pipeline 之后」的那一段，模型加载/唤醒不受超时影响。"""
        info("任务开始执行", task.source, task.task_id, "model=", task.model)
        try:
            return await op(task)
        except asyncio.TimeoutError:
            return self._fail(task, 504, f"生成超时 ({task.timeout}s，从进入 pipeline 起算，不含模型加载)")
        except (RuntimeError, MemoryError, httpx.HTTPError) as exc:
            return self._fail(task, task.fail_code, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ 管理指令
    async def _sleep_or_unload(self, task: _taskItem) -> dict[str, Any]:
        action = task.spec.sleep if task.message_id == adim_ID.eSleep.value else task.spec.unload
        ok = await asyncio.to_thread(action)
        return self._response(task.model, task.message_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")

    async def _ensure_loaded(self, task: _taskItem) -> dict[str, Any]:
        await self.pipeline.ensure([task.spec], not self.pipeline.hasMain(), task.deploy)
        return self._response(task.model, task.message_id, "ok", "操作成功")

    # ------------------------------------------------------------------ chat
    async def _chat(self, task: _taskItem) -> dict[str, Any]:
        """① 校验模型 ② 分级 ③ 确保模型就绪（无超时） ④ 进入 pipeline 后开始计时：排队 + 生成 ⑤ 组装响应。

        models.json 里没有的 id：非 agent 来源或没有主模型直接报错；agent 来源且已有主模型则当作客户端别名直接调度。
        有的 id：已在池里就直接调度，否则先创建并登记（没有主模型或本次是主会话时登记为主池，否则登记为副池）。"""
        pipe, spec, args = self.pipeline, task.spec, dict(task.deploy)

        await pipe.settled()  # 模型正在加载/切换时先等它结束，再按最新的池状态判断
        if spec is None and (task.client == "other" or not pipe.hasMain()):
            return self._fail(task, 404, f"模型不存在: {task.model}（来源归类={task.client}）")

        category = pipe.classify(task.messages, args.get("tools"))

        # 模型不在池里 / 没有主模型 / 处于休眠：都在计时开始前加载或唤醒好，冷启动再久也不会撞上超时
        if spec is not None and not (pipe.has(spec) and pipe.hasMain() and spec.isRunning):
            await pipe.ensure([spec], category.is_main or not pipe.hasMain(), args)

        self._log_request(task, args)
        task.fail_code = 502  # 模型已就绪，之后的失败属于引擎
        wait_s = task.timeout if task.timeout and task.timeout > 0 else None
        result = await asyncio.wait_for(pipe.run(category, task.messages, args), wait_s)  # 从这里起才计时

        return self._response(task.model, 0, "ok", result.text) | {
            "tool_calls": result.tool_calls,
            "finish_reason": result.finish_reason,
            "reasoning": result.reasoning,
            "warnings": result.warnings,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "cached_tokens": result.cached_tokens,
                "time_seconds": result.time_seconds,
                "tokens_per_second": round(result.tokens_per_second, 2),
                "kv_cache_remaining": result.kv_remaining,
            },
        }

    @staticmethod
    def _log_request(task: _taskItem, args: dict[str, Any]) -> None:
        """缓存命中排查用的请求指纹日志；排查完可整体删除。"""
        tools = args.get("tools") or []
        info("[DBG] 请求", task.model, f"{len(task.messages)}条", [m.get("role") for m in task.messages], f"tools={len(tools)}#{_digest(tools)}", "首条#", _digest(task.messages[0] if task.messages else {}), "去末条#", _digest(task.messages[:-1]), "字段=", sorted(args))

    # ------------------------------------------------------------------ 响应
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

    def _fail(self, task: _taskItem, code: int, reason: str) -> dict[str, Any]:
        error("[请求失败]", task.source, task.task_id, f"model={task.model}", f"code={code}", reason)
        return self._response(task.model, task.message_id, "error", reason) | {"code": code}
