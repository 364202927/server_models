"""Console 与 FastAPI 共用的统一消息处理器。

单流程说明：无论请求是 Console 的裸 prompt 还是 HTTP 的结构化 messages，
最终都会在 _generate_chat -> _build_messages 这一步统一成同一份 messages
交给 Loader；Loader 内部只按"有没有 tools"决定走普通聊天模板还是工具模板，
不再区分"有没有 messages"。

工具数量较多时（超过 _TOOL_SEARCH_THRESHOLD），_generate_with_tool_search
会切换成按需检索模式：只把 __search_tools__ 元工具的完整定义交给模型，
其余工具收进一份精简目录，模型需要时自己调用 __search_tools__ 换取完整
定义。这个过程完全在 MsgHandler 内部完成，对客户端和各个 Loader 透明——
见 loader/tool_format.py。
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .hardware import detect_hardware
from .loader.llmFramework.baseInference import ToolCapabilityError, ToolOutputError
from .loader.model_spec import LOAD_KEYS
from .loader.models_mgr import ModelsMgr
from .loader.tool_format import SEARCH_TOOL_NAME, merge_tools, prepare_tool_view, search_tools
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

# 工具数超过这个值才启用 __search_tools__ 按需检索；数量不多时直接原样透传更简单。
_TOOL_SEARCH_THRESHOLD = 8
# 每次 __search_tools__ 命中后最多带回几个工具的完整定义。
_TOOL_SEARCH_TOP_K = 3
# 最多允许几轮"调用 __search_tools__ -> 重新生成"；超过后直接把全量工具兜底发一次。
_MAX_TOOL_SEARCH_HOPS = 2


# --------------------------------------------------------------------------
# 工具调用请求/响应的校验；均为纯函数，可脱离 MsgHandler 独立测试。
# --------------------------------------------------------------------------

def _validate_tools(tools: list[dict[str, Any]]) -> set[str]:
    """校验 tools 列表中每个函数工具定义是否合法，返回已出现过的工具名集合。"""
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
    return names


def _resolve_tool_choice(tool_choice: Any, tools: list[dict[str, Any]], names: set[str]) -> str | dict[str, Any]:
    """解析并校验 tool_choice，返回规范化后的取值。"""
    choice = tool_choice if tool_choice is not None else ("auto" if tools else "none")
    if isinstance(choice, str):
        if choice not in {"none", "auto", "required"}:
            raise ValueError(f"不支持的 tool_choice: {choice}")
        if choice == "required" and not tools:
            raise ValueError("tool_choice=required 时 tools 不能为空")
        return choice
    if isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if choice.get("type") != "function" or name not in names:
            raise ValueError("tool_choice 指定的函数不在 tools 中")
        return choice
    raise ValueError("tool_choice 必须是 none/auto/required 或函数选择对象")


def _validate_message_history(messages: list[dict[str, Any]]) -> None:
    """校验对话历史里 tool_calls 与 role=tool 结果的配对是否完整、合法。"""
    known: set[str] = set()
    pending: set[str] = set()
    completed: set[str] = set()

    def register_calls(calls: Any, role: str) -> None:
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

    def consume_tool_result(message: dict[str, Any]) -> None:
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or call_id not in pending:
            suffix = "已重复回传" if call_id in completed else "不存在"
            raise ValueError(f"tool_call_id {call_id!r} {suffix}")
        pending.remove(call_id)
        completed.add(call_id)

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages 中的每一项必须是对象")
        role = message.get("role")
        if pending and role != "tool":
            raise ValueError("工具调用结果未补齐，不能开始下一轮消息")
        calls = message.get("tool_calls")
        if calls is not None:
            register_calls(calls, role)
        if role == "tool":
            consume_tool_result(message)
    if pending:
        raise ValueError("工具调用结果未补齐: " + ", ".join(sorted(pending)))


def validate_tool_request(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                           tool_choice: Any) -> tuple[list[dict[str, Any]], str | dict[str, Any]]:
    """校验现代 function 工具定义与历史消息中的工具调用配对，返回规范化后的 (tools, tool_choice)。

    这里始终对 messages 做结构校验（不管本次是否带 tools），因为服务端始终
    把结构化 messages 交给 Loader；tools 为空时 tool_choice 会被归一为
    "none"，下游据此判断"这是一次普通对话"。
    """
    tools = tools or []
    names = _validate_tools(tools)
    choice = _resolve_tool_choice(tool_choice, tools, names)
    _validate_message_history(messages)
    return tools, choice


def validate_tool_output(calls: list[dict[str, Any]], tools: list[dict[str, Any]],
                          choice: str | dict[str, Any], parallel: bool) -> list[dict[str, Any]]:
    """校验并规范化模型返回的工具调用，确保符合本次请求的 tool_choice/parallel_tool_calls 约束。"""
    if choice == "none" and calls:
        raise ToolOutputError("模型在 tool_choice=none 时返回了工具调用")
    if choice == "required" and not calls:
        raise ToolOutputError("模型未按 tool_choice=required 调用工具")
    if not parallel and len(calls) > 1:
        raise ToolOutputError("模型返回了多个工具调用，但 parallel_tool_calls=false")

    names = {tool["function"]["name"] for tool in tools}
    forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
    call_ids: set[str] = set()

    def normalize(call: dict[str, Any]) -> dict[str, Any]:
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
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

    return [normalize(call) for call in calls]


@dataclass
class _ChatContext:
    """message_id == 0（聊天）分支所需的参数集合；把 handle() 的一长串关键字参数收敛成一个对象，
    避免 _dispatch/_generate_chat 各自携带十几个参数。"""
    model: str = ""
    prompt: str = ""
    think: int = 0
    deploy: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    parallel_tool_calls: bool = True
    system_prompt: str = ""


class MsgHandler:
    """把不同入口的消息串行路由到 ModelsMgr。"""

    def __init__(self, manager: ModelsMgr) -> None:
        self.manager = manager
        self._lock = asyncio.Lock()
        self._pending = 0

    @property
    def pending(self) -> int:
        return self._pending

    length = pending  # 兼容外部代码按 length 属性访问排队数（如确认无外部引用可删除）

    @staticmethod
    def _response(model: str, message_id: int, status: str, value: Any) -> dict[str, Any]:
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return {"request_id": str(uuid.uuid4()), "status": status, "model": model,
                "message_id": message_id, "response": payload}

    @staticmethod
    def _model_from(args: Any) -> str:
        if isinstance(args, dict):
            return str(args.get("model", ""))
        if isinstance(args, (list, tuple)) and args:
            return str(args[0])
        return str(args or "")

    async def handle(self, message_id: int, args: Any = None, *, model: str = "", prompt: str = "",
                      think: int = 0, deploy: dict[str, Any] | None = None, source: str = "unknown",
                      messages: list[dict[str, Any]] | None = None,
                      tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
                      parallel_tool_calls: bool = True, system_prompt: str = "") -> dict[str, Any]:
        """统一入口：校验请求合法性后串行分发给 ModelsMgr。message_id=0 为聊天，1001~1007 为管理指令。"""
        if message_id not in SUPPORTED_MESSAGE_IDS:
            raise ValueError(f"不支持的 message_id: {message_id}")
        if message_id != 0 and (tools or messages is not None or tool_choice is not None):
            raise ValueError("管理指令不支持工具参数或工具消息")
        if messages is not None:
            tools, tool_choice = validate_tool_request(messages, tools, tool_choice)

        ctx = _ChatContext(model=model, prompt=prompt, think=think, deploy=deploy or {}, messages=messages,
                            tools=tools or [], tool_choice=tool_choice,
                            parallel_tool_calls=parallel_tool_calls, system_prompt=system_prompt)
        self._pending += 1
        try:
            async with self._lock:
                info("消息处理", source, message_id, model)
                result = await self._dispatch(message_id, args, ctx)
                info("消息处理完成", source, message_id,
                     "status=", result.get("status", "unknown"),
                     "response_chars=", len(str(result.get("response", ""))))
                return result
        finally:
            self._pending -= 1

    async def _dispatch(self, message_id: int, args: Any, ctx: _ChatContext) -> dict[str, Any]:
        model = ctx.model
        try:
            if message_id == 1001:
                return self._status(model)
            if message_id == 1002:
                return self._hardware_info(model)
            if message_id == 1003:
                return self._list_models(model)
            if message_id in {1004, 1005}:
                return self._sleep_or_unload(message_id, model or self._model_from(args))
            if message_id == 1006:
                return await self._ensure_loaded(model or self._model_from(args))
            if message_id == 1007:
                return await self._update_generation(model, args)

            payload = args if isinstance(args, dict) else {}
            model = model or str(payload.get("model", ""))
            prompt = ctx.prompt or str(payload.get("prompt", args if isinstance(args, str) else ""))
            if isinstance(args, (list, tuple)) and not model and args:
                model = str(args[0])
            if isinstance(args, (list, tuple)) and not prompt and len(args) > 1:
                prompt = str(args[1])
            info("请求参数解析", "id=", message_id, "model=", model,
                 "prompt_chars=", len(prompt), "args_type=", type(args).__name__)
            return await self._generate_chat(model, prompt, payload, ctx)
        except ToolCapabilityError as exc:
            return self._response(model, message_id, "error", str(exc)) | {"error_type": "request"}
        except ToolOutputError as exc:
            return self._response(model, message_id, "error", str(exc)) | {"error_type": "tool_output"}
        except (KeyError, RuntimeError, MemoryError, ValueError, FileNotFoundError) as exc:
            log("消息处理失败:", exc)
            return self._response(model, message_id, "error", str(exc))
        except Exception as exc:
            # 推理后端可能抛出自定义异常；统一转成可见的错误响应，避免
            # Console/FastAPI 只看到"消息处理开始"却没有结束结果。
            log("消息处理异常:", type(exc).__name__, exc)
            return self._response(model, message_id, "error", f"推理失败: {exc}")

    # ---- 管理指令：每个 message_id 对应一个独立方法，便于单独测试/复用 ----

    def _status(self, model: str) -> dict[str, Any]:
        value = self.manager.status()
        value["queue_length"] = self.pending
        return self._response(model, 1001, "ok", value)

    def _hardware_info(self, model: str) -> dict[str, Any]:
        return self._response(model, 1002, "ok", detect_hardware().to_dict())

    def _list_models(self, model: str) -> dict[str, Any]:
        return self._response(model, 1003, "ok", {"models": list(self.manager.specs)})

    def _sleep_or_unload(self, message_id: int, target: str) -> dict[str, Any]:
        ok = self.manager.sleep(target) if message_id == 1004 else self.manager.unload(target)
        return self._response(target, message_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")

    async def _ensure_loaded(self, target: str) -> dict[str, Any]:
        await asyncio.to_thread(self.manager.ensure_loaded, target)
        return self._response(target, 1006, "ok", "操作成功")

    async def _update_generation(self, model: str, args: Any) -> dict[str, Any]:
        """持久化生成参数；请求里的 deploy 只影响当次，改常驻值走这里。"""
        payload = args if isinstance(args, dict) else {}
        target = model or str(payload.get("model", ""))
        changes = {key: value for key, value in
                   dict(payload.get("generation", payload.get("deploy", {}))).items()
                   if key in GENERATION_KEYS}
        merged = await asyncio.to_thread(self.manager.update_generation, target, changes)
        return self._response(target, 1007, "ok", merged)

    # ---- message_id == 0：聊天/工具调用（单流程：统一走结构化 messages）----

    @staticmethod
    def _clean_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """客户端（如 OpenWebUI）会把模型输出的推理段原样存回历史再传回来；
        清掉历史 assistant 消息里残留的 ``...</think>`` 前缀，避免脏文本被当
        成正文再次拼进下一轮 prompt、越攒越长最终把上下文撑爆。"""
        cleaned = []
        for item in messages:
            item = dict(item)
            content = item.get("content")
            if item.get("role") == "assistant" and isinstance(content, str) and "</think>" in content:
                item["content"] = content.split("</think>", 1)[1].strip()
            cleaned.append(item)
        return cleaned

    def _build_messages(self, prompt: str, ctx: _ChatContext, default_system: str,
                        extra_system: str = "") -> list[dict[str, Any]]:
        """把 Console 的裸 prompt 与 HTTP 的结构化 messages 统一成同一份 messages。

        system 段来源（从低到高优先级，同时给出时依次拼接）：
        模型配置的默认 system_prompt → 请求级 system_prompt → think 等级指令 →
        工具检索目录说明（仅工具数超过阈值时存在，见 tool_format.prepare_tool_view）。
        与 messages 里已有的首条 system 消息合并，避免出现两条 system。
        """
        messages = (self._clean_history([dict(item) for item in ctx.messages]) if ctx.messages is not None
                    else [{"role": "user", "content": prompt}])

        def think_instruction() -> str:
            return f"请以推理等级 {ctx.think}/5 分析后给出最终答案。" if ctx.think else ""

        segments = [text for text in (default_system, ctx.system_prompt, think_instruction(), extra_system)
                   if text]
        if not segments:
            return messages
        if messages and messages[0].get("role") == "system":
            existing = str(messages[0].get("content") or "")
            messages[0]["content"] = "\n\n".join([*segments, existing] if existing else segments)
        else:
            messages.insert(0, {"role": "system", "content": "\n\n".join(segments)})
        return messages

    async def _generate_chat(self, model: str, prompt: str, payload: dict[str, Any],
                              ctx: _ChatContext) -> dict[str, Any]:
        """构建生成参数并调用 Loader；处理 message_id == 0 的核心逻辑。"""
        deploy = {**dict(payload.get("deploy", {})), **ctx.deploy}
        changes = {key: value for key, value in deploy.items() if key in LOAD_KEYS}
        if changes:
            # reconfigure 内部会与当前 load 值 diff，值没变就不会重载模型。
            await asyncio.to_thread(self.manager.reconfigure, model, changes)

        # defaults.generation ← 模型 generation ← 本次 deploy（不落盘）。
        params = self.manager.generation_params(model)
        params.update({key: value for key, value in deploy.items() if key in GENERATION_KEYS})
        params = {key: value for key, value in params.items() if key in GENERATION_KEYS}
        if "max_tokens" in params:
            params["max_new_tokens"] = params.pop("max_tokens")

        # 单流程核心：所有聊天请求统一走结构化 messages。system_prompt 从这里
        # 摘出来折进 messages，避免 Loader 端对同一份系统提示做两次处理。
        default_system = str(params.pop("system_prompt", "") or "")
        tools_for_model, catalog_notice = prepare_tool_view(ctx.tools, ctx.tool_choice, _TOOL_SEARCH_THRESHOLD)
        messages = self._build_messages(prompt, ctx, default_system, catalog_notice or "")

        result = await self._generate_with_tool_search(model, prompt, messages, tools_for_model, ctx, params)

        # 没有声明 tools 时不校验工具输出：普通对话里模型偶发吐出类似
        # <tool_call> 的文本不该被当成协议违规而返回 502，原样按文本处理。
        calls = (validate_tool_output(result.tool_calls, ctx.tools, ctx.tool_choice, ctx.parallel_tool_calls)
                 if ctx.tools else [])
        return self._response(model, 0, "ok", result.text) | {
            "tool_calls": calls,
            "finish_reason": "tool_calls" if calls else result.finish_reason,
            "usage": {
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.tokens_generated,
                "time_seconds": result.time_seconds, "tokens_per_second": result.tokens_per_second,
            },
        }

    async def _generate_with_tool_search(self, model: str, prompt: str, messages: list[dict[str, Any]],
                                         tools_for_model: list[dict[str, Any]], ctx: _ChatContext,
                                         params: dict[str, Any]) -> Any:
        """调用 Loader 生成一个回复。

        工具数没超过阈值时 tools_for_model 就是 ctx.tools，第一轮必然不会
        命中 SEARCH_TOOL_NAME，行为和不做检索完全一样。工具数超过阈值时，
        模型调用 __search_tools__ 就在这里按需展开完整定义并重新生成，
        对客户端和 Loader 都透明——两者都只会看到真正的工具调用。
        """
        tools = tools_for_model
        for _ in range(_MAX_TOOL_SEARCH_HOPS):
            params.update({"messages": messages, "tools": tools, "tool_choice": ctx.tool_choice,
                          "parallel_tool_calls": ctx.parallel_tool_calls})
            result = await asyncio.to_thread(self.manager.generate, model, prompt, **params)
            search_calls = [call for call in result.tool_calls if call["function"]["name"] == SEARCH_TOOL_NAME]
            if not search_calls:
                return result

            messages = messages + [{"role": "assistant", "tool_calls": search_calls}]
            for call in search_calls:
                found = search_tools(self._search_query(call), ctx.tools, _TOOL_SEARCH_TOP_K)
                tools = merge_tools(tools, found)
                content = ("找到: " + ", ".join(tool["function"]["name"] for tool in found) if found else
                          "没有找到匹配的工具，换个关键词再试，或者直接回答用户。")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

        # 超过检索跳数上限：把完整工具集合一次性交给模型兜底，避免用户被卡在搜索循环里。
        params.update({"messages": messages, "tools": ctx.tools, "tool_choice": ctx.tool_choice,
                      "parallel_tool_calls": ctx.parallel_tool_calls})
        return await asyncio.to_thread(self.manager.generate, model, prompt, **params)

    @staticmethod
    def _search_query(call: dict[str, Any]) -> str:
        """安全解析 __search_tools__ 调用的 query 参数；解析失败就当作空查询（返回空结果）。"""
        try:
            arguments = call["function"]["arguments"]
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            return str(parsed.get("query", "")) if isinstance(parsed, dict) else ""
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return ""


msgHandler = MsgHandler
__all__ = ["GENERATION_KEYS", "SUPPORTED_MESSAGE_IDS", "MsgHandler", "msgHandler",
           "validate_tool_request", "validate_tool_output"]