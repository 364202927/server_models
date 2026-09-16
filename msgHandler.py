"""Console 与 FastAPI 共用的统一消息处理器。"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any

from .hardware import detect_hardware
from .loader.base import ToolCapabilityError, ToolOutputError
from .loader.model_spec import LOAD_KEYS
from .loader.models_mgr import ModelsMgr
from .utils.common import info, log

# 每次请求可覆盖的采样参数；``max_tokens`` 在传给 Loader 前改名为 ``max_new_tokens``。
GENERATION_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
    "max_tokens", "stop_sequences", "system_prompt", "seed", "logit_bias",
    "frequency_penalty", "presence_penalty", "repeat_last_n", "tfs_z",
    "mirostat", "mirostat_eta", "mirostat_tau",
})
SUPPORTED_MESSAGE_IDS = frozenset({0, 1001, 1002, 1003, 1004, 1005, 1006, 1007})
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_tool_request(messages: list[dict[str, Any]],tools: list[dict[str, Any]] | None,tool_choice: Any) -> tuple[list[dict[str, Any]], str | dict[str, Any]]:
    """Validate modern function tools and tool-call message history."""
    tools = tools or []
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("当前仅支持 type=function 的工具")
        name = function.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise ValueError("工具名称必须是 1~64 位字母、数字、下划线或连字符")
        if name in names:
            raise ValueError(f"工具名称重复: {name}")
        if function.get("strict") is True:
            raise ValueError(f"工具 {name} 的 strict=true 当前不支持")
        if "parameters" in function and not isinstance(function["parameters"], dict):
            raise ValueError(f"工具 {name} 的 parameters 必须是 JSON Schema 对象")
        names.add(name)

    choice = tool_choice if tool_choice is not None else ("auto" if tools else "none")
    if isinstance(choice, str):
        if choice not in {"none", "auto", "required"}:
            raise ValueError(f"不支持的 tool_choice: {choice}")
        if choice == "required" and not tools:
            raise ValueError("tool_choice=required 时 tools 不能为空")
    elif isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if choice.get("type") != "function" or name not in names:
            raise ValueError("tool_choice 指定的函数不在 tools 中")
    else:
        raise ValueError("tool_choice 必须是 none/auto/required 或函数选择对象")

    pending: set[str] = set()
    known: set[str] = set()
    completed: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages 中的每一项必须是对象")
        role = message.get("role")
        if pending and role != "tool":
            raise ValueError("工具调用结果未补齐，不能开始下一轮消息")
        calls = message.get("tool_calls")
        if calls is not None:
            if role != "assistant" or not isinstance(calls, list) or not calls:
                raise ValueError("tool_calls 只能出现在 assistant 消息且不能为空")
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                call_id = call.get("id") if isinstance(call, dict) else None
                if (not isinstance(call_id, str) or not call_id or call_id in known
                        or not isinstance(function, dict) or not function.get("name")):
                    raise ValueError("历史 tool_calls 的 ID 或函数无效")
                arguments = function.get("arguments", "{}")
                try:
                    decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError as exc:
                    raise ValueError(f"工具调用 {call_id} 的 arguments 不是合法 JSON") from exc
                if not isinstance(decoded, dict):
                    raise ValueError(f"工具调用 {call_id} 的 arguments 必须是 JSON 对象")
                known.add(call_id)
                pending.add(call_id)
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                suffix = "已重复回传" if call_id in completed else "不存在"
                raise ValueError(f"tool_call_id {call_id!r} {suffix}")
            pending.remove(call_id)
            completed.add(call_id)
    if pending:
        raise ValueError("工具调用结果未补齐: " + ", ".join(sorted(pending)))
    return tools, choice


def validate_tool_output(calls: list[dict[str, Any]],tools: list[dict[str, Any]],choice: str | dict[str, Any],parallel: bool) -> list[dict[str, Any]]:
    """Normalize model calls and enforce the request's tool constraints."""
    names = {tool["function"]["name"] for tool in tools}
    if choice == "none" and calls:
        raise ToolOutputError("模型在 tool_choice=none 时返回了工具调用")
    if choice == "required" and not calls:
        raise ToolOutputError("模型未按 tool_choice=required 调用工具")
    if not parallel and len(calls) > 1:
        raise ToolOutputError("模型返回了多个工具调用，但 parallel_tool_calls=false")
    forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
    normalized = []
    call_ids: set[str] = set()
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name not in names or (forced and name != forced):
            raise ToolOutputError(f"模型调用了未允许的工具: {name}")
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        try:
            if not isinstance(json.loads(arguments), dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError) as exc:
            raise ToolOutputError(f"工具 {name} 的 arguments 不是 JSON 对象") from exc
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex}")
        if call_id in call_ids:
            raise ToolOutputError(f"模型返回了重复的工具调用 ID: {call_id}")
        call_ids.add(call_id)
        normalized.append({"id": call_id,
                           "type": "function",
                           "function": {"name": name, "arguments": arguments}})
    return normalized


class MsgHandler:
    """把不同入口的消息串行路由到 ModelsMgr。"""

    def __init__(self, manager: ModelsMgr) -> None:
        self.manager = manager
        self._lock = asyncio.Lock()
        self._pending = 0

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def length(self) -> int:
        return self._pending

    @staticmethod
    def _response(model: str, message_id: int, status: str, value: Any) -> dict[str, Any]:
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return {"request_id": str(uuid.uuid4()), "status": status, "model": model,
                "message_id": message_id, "response": payload}

    async def handle(self, message_id: int, args: Any = None, *, model: str = "", prompt: str = "",think: int = 0, deploy: dict[str, Any] | None = None, source: str = "unknown",messages: list[dict[str, Any]] | None = None,tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,parallel_tool_calls: bool = True, system_prompt: str = "") -> dict[str, Any]:
        if message_id not in SUPPORTED_MESSAGE_IDS:
            raise ValueError(f"不支持的 message_id: {message_id}")
        if message_id != 0 and (tools or messages is not None or tool_choice is not None):
            raise ValueError("管理指令不支持工具参数或工具消息")
        if messages is not None:
            tools, tool_choice = validate_tool_request(messages, tools, tool_choice)
        self._pending += 1
        try:
            async with self._lock:
                info("消息处理", source, message_id, model)
                result = await self._dispatch(
                    message_id, args, model=model, prompt=prompt, think=think,
                    deploy=deploy or {}, messages=messages, tools=tools or [],
                    tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
                    system_prompt=system_prompt,
                )
                info("消息处理完成", source, message_id,
                     "status=", result.get("status", "unknown"),
                     "response_chars=", len(str(result.get("response", ""))))
                return result
        finally:
            self._pending -= 1

    async def _dispatch(self, message_id: int, args: Any, *, model: str, prompt: str, think: int,deploy: dict[str, Any], messages: list[dict[str, Any]] | None,tools: list[dict[str, Any]], tool_choice: Any, parallel_tool_calls: bool,system_prompt: str) -> dict[str, Any]:
        try:
            if message_id == 1001:
                value = self.manager.status()
                value["queue_length"] = self.pending
                return self._response(model, message_id, "ok", value)
            if message_id == 1002:
                return self._response(model, message_id, "ok", detect_hardware().to_dict())
            if message_id == 1003:
                return self._response(model, message_id, "ok", {"models": list(self.manager.specs)})
            if message_id in {1004, 1005}:
                target = model or self._model_from(args)
                ok = self.manager.sleep(target) if message_id == 1004 else self.manager.unload(target)
                return self._response(target, message_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")
            if message_id == 1006:
                target = model or self._model_from(args)
                await asyncio.to_thread(self.manager.ensure_loaded, target)
                return self._response(target, message_id, "ok", "操作成功")
            if message_id == 1007:
                # 持久化生成参数；请求里的 deploy 只影响当次，改常驻值走这里。
                payload = args if isinstance(args, dict) else {}
                target = model or str(payload.get("model", ""))
                changes = {key: value for key, value in
                           dict(payload.get("generation", payload.get("deploy", {}))).items()
                           if key in GENERATION_KEYS}
                merged = await asyncio.to_thread(self.manager.update_generation, target, changes)
                return self._response(target, message_id, "ok", merged)

            payload = args if isinstance(args, dict) else {}
            model = model or str(payload.get("model", ""))
            prompt = prompt or str(payload.get("prompt", args if isinstance(args, str) else ""))
            if isinstance(args, (list, tuple)):
                if not model and args:
                    model = str(args[0])
                if not prompt and len(args) > 1:
                    prompt = str(args[1])
            info("请求参数解析", "id=", message_id, "model=", model,
                 "prompt_chars=", len(prompt), "args_type=", type(args).__name__)
            deploy = {**dict(payload.get("deploy", {})), **deploy}
            if deploy:
                # reconfigure 内部会与当前 load 值 diff，值没变就不会重载模型。
                changes = {key: value for key, value in deploy.items() if key in LOAD_KEYS}
                if changes:
                    await asyncio.to_thread(self.manager.reconfigure, model, changes)
            # defaults.generation ← 模型 generation ← 本次 deploy（不落盘）。
            params = self.manager.generation_params(model)
            params.update({key: value for key, value in deploy.items() if key in GENERATION_KEYS})
            params = {key: value for key, value in params.items() if key in GENERATION_KEYS}
            if "max_tokens" in params:
                params["max_new_tokens"] = params.pop("max_tokens")
            if messages is not None:
                messages = [dict(item) for item in messages]
                instructions = [item for item in (system_prompt,
                                f"请以推理等级 {think}/5 分析后给出最终答案。" if think else "") if item]
                if instructions:
                    messages.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
                params.update({"messages": messages, "tools": tools,
                               "tool_choice": tool_choice,
                               "parallel_tool_calls": parallel_tool_calls})
            elif think:
                prompt = f"请以推理等级 {think}/5 分析后给出最终答案。\n\n{prompt}"
            result = await asyncio.to_thread(self.manager.generate, model, prompt, **params)
            calls = validate_tool_output(result.tool_calls, tools, tool_choice, parallel_tool_calls) \
                if messages is not None else result.tool_calls
            return self._response(model, 0, "ok", result.text) | {
                "tool_calls": calls,
                "finish_reason": "tool_calls" if calls else result.finish_reason,
                "usage": {
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.tokens_generated,
                "time_seconds": result.time_seconds, "tokens_per_second": result.tokens_per_second,
            }}
        except ToolCapabilityError as exc:
            return self._response(model, message_id, "error", str(exc)) | {"error_type": "request"}
        except ToolOutputError as exc:
            return self._response(model, message_id, "error", str(exc)) | {"error_type": "tool_output"}
        except (KeyError, RuntimeError, MemoryError, ValueError, FileNotFoundError) as exc:
            log("消息处理失败:", exc)
            return self._response(model, message_id, "error", str(exc))
        except Exception as exc:
            # 推理后端可能抛出自定义异常；统一转成可见的错误响应，避免
            # Console/FastAPI 只看到“消息处理开始”却没有结束结果。
            log("消息处理异常:", type(exc).__name__, exc)
            return self._response(model, message_id, "error", f"推理失败: {exc}")

    @staticmethod
    def _model_from(args: Any) -> str:
        if isinstance(args, dict):
            return str(args.get("model", ""))
        if isinstance(args, (list, tuple)) and args:
            return str(args[0])
        return str(args or "")

msgHandler = MsgHandler
__all__ = ["GENERATION_KEYS", "SUPPORTED_MESSAGE_IDS", "MsgHandler", "msgHandler",
           "validate_tool_request", "validate_tool_output"]
