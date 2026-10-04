from __future__ import annotations
import uuid, asyncio, json
from dataclasses import dataclass, field
from typing import Any

from . import adim_ID
from .hardware import detect_hardware
from .loader.model_spec import LOAD_KEYS
from .loader.models_mgr import ModelsMgr
from .utils.common import info, log

# 每次请求可覆盖的采样参数；``max_tokens`` 在传给 Loader 前改名为 ``max_new_tokens``。
GENERATION_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
    "max_tokens", "stop_sequences", "system_prompt", "seed", "logit_bias",
    "frequency_penalty", "presence_penalty", "repeat_last_n", "tfs_z",
    "mirostat", "mirostat_eta", "mirostat_tau", "regex", "json_schema",
})

#解析think强度serverapi 用
def resolve_think_level(*, think: Any = None, reasoning_effort: Any = None) -> int:
    REASONING_EFFORT_LEVELS = {"none": 0, "low": 1, "medium": 3, "high": 5}
    if think is not None:
        return max(0, min(5, int(think)))
    if isinstance(reasoning_effort, int):
        return max(0, min(5, reasoning_effort))
    return REASONING_EFFORT_LEVELS.get(str(reasoning_effort).lower(), 0)

#合并归一化字段 serverapi 用
def normalize_generation_params(*sources: dict[str, Any], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    GENERATION_ALIASES = {"stop": "stop_sequences", "repeat_penalty": "repetition_penalty"} #协议返回的
    alias_map = {**GENERATION_ALIASES, **(aliases or {})}
    values: dict[str, Any] = {}
    for source in sources:
        for key, value in sorted(source.items(), key=lambda kv: kv[0] in alias_map):
            if value is not None:
                values.setdefault(alias_map.get(key, key), value)
    return {key: value for key, value in values.items() if key in GENERATION_KEYS}

@dataclass#跨协议的规范化聊天请求
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    think: int = 0
    deploy: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    parallel_tool_calls: bool = True
    system_prompt: str = ""


@dataclass#管理员指令结构
class AdminRequest:
    message_id: int
    model: str = ""
    generation: dict[str, Any] = field(default_factory=dict)


@dataclass#聊天结构
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
        return {"request_id": str(uuid.uuid4()), "status": status, "model": model, "message_id": message_id, "response": payload}

    @staticmethod
    def _model_from(args: Any) -> str:
        if isinstance(args, dict):
            return str(args.get("model", ""))
        if isinstance(args, (list, tuple)) and args:
            return str(args[0])
        return str(args or "")

    # console输入参数触发的聊天
    async def handle(self, message_id: int, args: Any = None, *, model: str = "", prompt: str = "",
                      think: int = 0, deploy: dict[str, Any] | None = None, source: str = "unknown",
                      messages: list[dict[str, Any]] | None = None,
                      tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
                      parallel_tool_calls: bool = True, system_prompt: str = "") -> dict[str, Any]:
        valid_ids = {0, *(e.value for e in adim_ID)}
        if message_id not in valid_ids:
            raise ValueError(f"不支持的 message_id: {message_id}")
        if message_id != 0 and messages is not None:
            raise ValueError("管理指令不支持消息参数")

        ctx = _ChatContext(model=model, prompt=prompt, think=think, deploy=deploy or {}, messages=messages,
                            tools=tools or [], tool_choice=tool_choice,
                            parallel_tool_calls=parallel_tool_calls, system_prompt=system_prompt)
        self._pending += 1
        try:
            async with self._lock:
                info("消息处理", source, message_id, model)
                result = await self._dispatch(message_id, args, ctx)
                info("消息处理完成", source, message_id, "status=", result.get("status", "unknown"),
                     "response_chars=", len(str(result.get("response", ""))))
                return result
        finally:
            self._pending -= 1

    # 服务器正常过来的聊天
    async def chat(self, request: ChatRequest, *, source: str = "unknown") -> dict[str, Any]:
        def has_non_text_content(messages: list[dict[str, Any]]) -> bool:
            """检测消息中是否包含非文本内容分段（当前服务仅支持纯文本消息，与具体协议无关）。"""
            for item in messages:
                if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                    continue
                if any(isinstance(part, dict) and part.get("type") != "text" for part in item["content"]):
                    return True
            return False
        def flatten_messages(messages: list[dict[str, Any]], system_prompt: str | None = None) -> str:
            def text_of(content: Any) -> str:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content
                                if isinstance(part, dict) and part.get("type") == "text")
                return str(content or "")

            turns = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
            return "\n".join(f"{turn.get('role', 'user')}: {text_of(turn.get('content'))}" for turn in turns)
        if has_non_text_content(request.messages):
            raise ValueError("当前服务仅支持文本消息")
        #
        ctx = _ChatContext(model=request.model,
                            prompt=flatten_messages(request.messages, request.system_prompt),
                            think=request.think, 
                            deploy=request.deploy, 
                            messages=request.messages,
                            tools=request.tools, 
                            tool_choice=request.tool_choice,
                            parallel_tool_calls=request.parallel_tool_calls,
                            system_prompt=request.system_prompt)
        self._pending += 1
        try:
            async with self._lock:
                info("消息处理", source, 0, request.model)
                result = await self._dispatch(0, None, ctx)
                info("消息处理完成", source, 0,
                     "status=", result.get("status", "unknown"),
                     "response_chars=", len(str(result.get("response", ""))))
                return result
        finally:
            self._pending -= 1

    # 管理指令
    async def admin(self, request: AdminRequest, *, source: str = "unknown") -> dict[str, Any]:
        admin_ids = {e.value for e in adim_ID}
        if request.message_id not in admin_ids:
            raise ValueError(f"不支持的管理指令 message_id: {request.message_id}")
        args = {"model": request.model, "generation": request.generation} if request.message_id == adim_ID.eUpdateGeneration.value else None

        ctx = _ChatContext(model=request.model)
        self._pending += 1
        try:
            async with self._lock:
                info("消息处理", source, request.message_id, request.model)
                result = await self._dispatch(request.message_id, args, ctx)
                info("消息处理完成", source, request.message_id,
                     "status=", result.get("status", "unknown"))
                return result
        finally:
            self._pending -= 1

    async def _dispatch(self, message_id: int, args: Any, ctx: _ChatContext) -> dict[str, Any]:
        model = ctx.model
        #指令触发
        if message_id == adim_ID.eStatus.value:
            return self._status(model)
        if message_id == adim_ID.eHardwareInfo.value:
            return self._hardware_info(model)
        if message_id == adim_ID.eModelsList.value:
            return self._list_models(model)
        if message_id in {adim_ID.eSleep.value, adim_ID.eUnload.value}:
            return self._sleep_or_unload(message_id, model or self._model_from(args))
        if message_id == adim_ID.eEnsureLoaded.value:
            return await self._ensure_loaded(model or self._model_from(args))
        if message_id == adim_ID.eUpdateGeneration.value:
            return await self._update_generation(model, args)
        #正常聊天
        payload = args if isinstance(args, dict) else {}
        model = model or str(payload.get("model", ""))
        prompt = ctx.prompt or str(payload.get("prompt", args if isinstance(args, str) else ""))
        if isinstance(args, (list, tuple)) and not model and args:
            model = str(args[0])
        if isinstance(args, (list, tuple)) and not prompt and len(args) > 1:
            prompt = str(args[1])
        info("请求参数解析", "id=", message_id, "model=", model,"prompt_chars=", len(prompt), "args_type=", type(args).__name__)
        return await self._generate_chat(model, prompt, payload, ctx)

    # ---- 管理指令：每个 message_id 对应一个独立方法，便于单独测试/复用 ----
    def _status(self, model: str) -> dict[str, Any]:
        value = self.manager.status()
        value["queue_length"] = self.pending
        return self._response(model, adim_ID.eStatus.value, "ok", value)

    def _hardware_info(self, model: str) -> dict[str, Any]:
        return self._response(model, adim_ID.eHardwareInfo.value, "ok", detect_hardware().to_dict())

    def _list_models(self, model: str) -> dict[str, Any]:
        return self._response(model, adim_ID.eModelsList.value, "ok", {"models": list(self.manager.specs)})

    def _sleep_or_unload(self, message_id: int, target: str) -> dict[str, Any]:
        ok = self.manager.sleep(target) if message_id == adim_ID.eSleep.value else self.manager.unload(target)
        return self._response(target, message_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")

    async def _ensure_loaded(self, target: str) -> dict[str, Any]:
        await asyncio.to_thread(self.manager.ensure_loaded, target)
        return self._response(target, adim_ID.eEnsureLoaded.value, "ok", "操作成功")

    async def _update_generation(self, model: str, args: Any) -> dict[str, Any]:
        payload = args if isinstance(args, dict) else {}
        target = model or str(payload.get("model", ""))
        changes = {key: value for key, value in
                   dict(payload.get("generation", payload.get("deploy", {}))).items()
                   if key in GENERATION_KEYS}
        merged = await asyncio.to_thread(self.manager.update_generation, target, changes)
        return self._response(target, adim_ID.eUpdateGeneration.value, "ok", merged)
    #
    async def _generate_chat(self, model: str, prompt: str, payload: dict[str, Any], ctx: _ChatContext) -> dict[str, Any]:
        def _clean_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            cleaned = []
            for item in messages:
                item = dict(item)
                content = item.get("content")
                if item.get("role") == "assistant" and isinstance(content, str) and "</think>" in content:
                    item["content"] = content.split("</think>", 1)[1].strip() #将历史字段里的</think>去掉
                cleaned.append(item)
            return cleaned
        def think_instruction() -> str:
            return f"请以推理等级 {ctx.think}/5 分析后给出最终答案。" if ctx.think else ""
        def _build_messages(self, prompt: str, ctx: _ChatContext, default_system: str,extra_system: str = "") -> list[dict[str, Any]]:
            messages = (_clean_history([dict(item) for item in ctx.messages]) if ctx.messages is not None
                        else [{"role": "user", "content": prompt}])
            segments = [text for text in (default_system, ctx.system_prompt, think_instruction(), extra_system)
                        if text]
            if not segments:
                return messages
            #构建标准消息上下文
            if messages and messages[0].get("role") == "system":
                existing = str(messages[0].get("content") or "")
                messages[0]["content"] = "\n\n".join([*segments, existing] if existing else segments)
            else:
                messages.insert(0, {"role": "system", "content": "\n\n".join(segments)})
            return messages
        #处理聊天参数
        deploy = {**dict(payload.get("deploy", {})), **ctx.deploy}
        changes = {key: value for key, value in deploy.items() if key in LOAD_KEYS}
        if changes:
            await asyncio.to_thread(self.manager.reconfigure, model, changes)
        # 合并与校准文本生成参数后重构聊天信息
        params = self.manager.generation_params(model)
        params.update({key: value for key, value in deploy.items() if key in GENERATION_KEYS})
        params = {key: value for key, value in params.items() if key in GENERATION_KEYS}
        if "max_tokens" in params:
            params["max_new_tokens"] = params.pop("max_tokens")
        default_system = str(params.pop("system_prompt", "") or "")
        messages = _build_messages(prompt, ctx, default_system)
        # 将工具发给模型调用处理
        params.update({"messages": messages, "tools": ctx.tools or None,
                      "tool_choice": ctx.tool_choice,
                      "parallel_tool_calls": ctx.parallel_tool_calls})
        result = await asyncio.to_thread(self.manager.generate, model, prompt, **params)

        return self._response(model, 0, "ok", result.text) | {
                        "tool_calls": result.tool_calls,
                        "finish_reason": result.finish_reason,
                        "usage": {
                            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.tokens_generated,
                            "time_seconds": result.time_seconds, "tokens_per_second": result.tokens_per_second,}
                        }


msgHandler = MsgHandler
__all__ = ["GENERATION_KEYS",
           "ChatRequest", "AdminRequest", "MsgHandler", "msgHandler",
           "resolve_think_level", "normalize_generation_params", "", "has_non_text_content"]