"""OpenAI / Open WebUI 兼容的 FastAPI 网关。

把 OpenAI Chat Completions 协议的请求翻译为内部 MsgHandler 调用，
并把内部生成结果重新包装为 OpenAI 兼容的响应（含 SSE 流式重放）。

单流程说明：所有聊天请求（message_id == 0）统一把结构化 messages 交给
MsgHandler，不再区分"要不要走工具"；是否拼普通 prompt、是否渲染工具
模板，全部下沉到 MsgHandler / Loader 决定（依据是否有 tools，而不是
是否有 messages）。API 层不再做这个分流。
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from itertools import chain
from typing import Any, AsyncIterator, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, StreamingResponse

from ..loader.models_mgr import ModelsMgr
from ..msgHandler import MsgHandler

# 会实际影响生成结果的采样/惩罚参数；用于从 extra_body/custom_parameters 中过滤出有效字段。
_GENERATION_FIELDS = {
    "temperature", "top_p", "top_k", "min_p", "max_tokens", "stop",
    "seed", "logit_bias", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "repeat_penalty", "repeat_last_n", "tfs_z",
    "mirostat", "mirostat_eta", "mirostat_tau",
}


class OpenAIChatRequest(BaseModel):
    """OpenAI / Open WebUI 兼容的聊天请求体。

    只显式声明会影响生成行为、或被服务实际使用的字段；未识别的字段仍会被接受（见 model_config），
    但不再显式声明纯粹"接收后忽略"的参数（如部分 Ollama 专属选项），以保持结构简洁。
    """
    model_config = ConfigDict(extra="allow")

    # ---- 服务层：路由与管理指令，不影响生成内容 ----
    message_id: int = 0                                            # 管理指令 ID：0 为普通聊天；1001~1007 转发给 MsgHandler 执行管理操作
    args: Any = None                                               # 管理指令参数，仅在 message_id 非 0 时使用

    # ---- 会话内容 ----
    model: str = ""                                                # 目标模型 ID，决定由哪个 Loader/权重生成回复
    messages: list[dict[str, Any]] = Field(default_factory=list)   # 对话历史；始终整体透传给 MsgHandler，由其决定拼接方式
    system_prompt: str | None = None                               # 追加系统提示词，插入到 messages 之前，用于设定角色、语气与行为约束

    # ---- 输出方式：只影响返回节奏/格式，不改变生成内容本身 ----
    stream: bool = False                                           # 是否以 SSE 分块返回
    stream_options: dict[str, Any] | None = None                   # 流式选项；include_usage=true 时额外多推一个 usage 分块
    stream_delta_chunk_size: int | None = Field(default=None, ge=0)  # 流式分块的字符数，越小分块越多、越接近逐字输出

    # ---- 采样与惩罚参数：直接影响回复内容 ----
    temperature: float | None = Field(default=None, ge=0)          # 采样温度，越高回复越发散随机，越低越保守确定
    top_p: float | None = Field(default=None, ge=0, le=1)          # 核采样概率阈值，越小候选词越集中、回复越保守
    top_k: int | None = Field(default=None, ge=0)                  # 候选 token 数量上限，越小回复用词越受限
    min_p: float | None = Field(default=None, ge=0, le=1)          # 最小概率采样阈值，过滤掉概率过低的候选 token（HF 后端）
    max_tokens: int | None = None                                  # 最大生成 token 数，决定回复长度上限
    max_completion_tokens: int | None = None                       # max_tokens 的 OpenAI 新版别名；与 max_tokens 同时给出且不一致时报错
    stop: str | list[str] | None = None                            # 停止序列，命中即立即终止生成
    seed: int | None = None                                        # 采样随机种子，固定后相同输入可复现相同输出
    repetition_penalty: float | None = None                        # 重复惩罚系数，越高越抑制重复用词
    repeat_penalty: float | None = None                            # repetition_penalty 的 llama.cpp/Ollama 命名别名
    frequency_penalty: float | None = None                         # 按 token 出现频率施加惩罚，抑制高频重复表达
    presence_penalty: float | None = None                          # 按 token 是否已出现过施加惩罚，鼓励话题展开而非重复
    logit_bias: dict[str, float] | None = None                     # 按 token ID 直接调整其被采样到的概率
    repeat_last_n: int | None = None                               # 重复惩罚回看的历史 token 窗口大小（GGUF）
    tfs_z: float | None = None                                     # Tail Free Sampling 参数，过滤低置信度的尾部候选（GGUF）
    mirostat: int | None = None                                    # Mirostat 自适应采样模式开关，用于稳定困惑度（GGUF）
    mirostat_eta: float | None = None                              # Mirostat 学习率，影响困惑度收敛速度（GGUF）
    mirostat_tau: float | None = None                              # Mirostat 目标熵，影响输出的可预测程度（GGUF）

    # ---- 推理强度 ----
    reasoning_effort: str | int | None = None                      # OpenAI 风格推理强度：none/low/medium/high，或直接给 0~5 等级
    think: bool | int | None = None                                # 内部推理开关/等级，优先级高于 reasoning_effort

    # ---- 参数透传通道：用于携带未在顶层声明、但属于 _GENERATION_FIELDS 的参数 ----
    extra_body: dict[str, Any] = Field(default_factory=dict)         # OpenAI SDK 常用的扩展参数透传字段，优先级高于 custom_parameters
    custom_parameters: dict[str, Any] = Field(default_factory=dict)  # 自定义参数透传字段，优先级低于 extra_body

    # ---- 工具调用：唯一的工具能力来源；服务端不猜测、不用 @ 关键字判断 ----
    tools: list[dict[str, Any]] | None = None                      # 可用函数工具定义；为空/未提供时模型只会普通回答
    tool_choice: Any = None                                        # none/auto/required 或指定函数，控制是否强制/禁止调用工具
    parallel_tool_calls: bool | None = None                        # 为 false 时模型一轮最多只返回一个工具调用

    # ---- 已废弃协议：仅用于识别并拒绝，不参与生成 ----
    function_call: Any = None                                      # 旧版函数调用配置；传入即拒绝，提示改用 tool_choice
    functions: list[dict[str, Any]] | None = None                  # 旧版函数定义；传入即拒绝，提示改用 tools


def _chat_prompt(messages: list[dict[str, Any]], system_prompt: str | None) -> str:
    """为旧版 handler 保留扁平 prompt，同时完整 messages 仍单独透传。"""
    def text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content
                           if isinstance(part, dict) and part.get("type") == "text")
        return str(content or "")

    turns = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    return "\n".join(f"{turn.get('role', 'user')}: {text_of(turn.get('content'))}" for turn in turns)


def _has_non_text_content(messages: list[dict[str, Any]]) -> bool:
    """检测消息中是否包含非文本内容分段（当前服务仅支持纯文本消息）。"""
    for item in messages:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        if any(isinstance(part, dict) and part.get("type") != "text" for part in item["content"]):
            return True
    return False


def _resolve_think_level(request: OpenAIChatRequest) -> int:
    """解析推理强度：think 优先于 reasoning_effort，统一转换为内部等级。"""
    if request.think is not None:
        value = int(request.think)
        return max(0, min(5, value))
    if isinstance(request.reasoning_effort, int):
        return max(0, min(5, request.reasoning_effort))
    return {"none": 0, "low": 1, "medium": 3, "high": 5}.get(str(request.reasoning_effort).lower(), 0)


def _generation_params(request: OpenAIChatRequest) -> dict[str, Any]:
    """合并顶层字段与 extra_body/custom_parameters 中的生成参数，并统一停止序列、长度、重复惩罚等别名。"""
    values = {field: getattr(request, field) for field in _GENERATION_FIELDS
              if getattr(request, field) is not None}
    for key, value in chain(request.extra_body.items(), request.custom_parameters.items()):
        if key in _GENERATION_FIELDS:
            values.setdefault(key, value)

    if request.max_tokens is not None and request.max_completion_tokens not in (None, request.max_tokens):
        raise HTTPException(status_code=400, detail="max_tokens 与 max_completion_tokens 冲突")
    if request.max_tokens is None and request.max_completion_tokens is not None:
        values["max_tokens"] = request.max_completion_tokens

    if "stop" in values:
        stop = values.pop("stop")
        values["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    if values.get("repeat_penalty") is not None:
        values["repetition_penalty"] = values.pop("repeat_penalty")
    if request.repetition_penalty is not None:
        values["repetition_penalty"] = request.repetition_penalty

    return values


def _build_payload(result: dict[str, Any], model_id: str) -> dict[str, Any]:
    """把内部生成结果转换为 OpenAI `chat.completion` 响应体。"""
    tool_calls = result.get("tool_calls") or []
    usage = result.get("usage", {})
    message: dict[str, Any] = {"role": "assistant", "content": None if tool_calls else result["response"]}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(time.time()), "model": result.get("model") or model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": result.get("finish_reason", "stop")}],
        "usage": {**usage, "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)},
    }


def _sse(payload: dict[str, Any]) -> str:
    """把一个 JSON 数据包装为 SSE `data:` 行。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunks(text: str, size: int) -> Iterator[str]:
    """按固定字符数切分文本，用于流式分块输出。"""
    for start in range(0, len(text), size):
        yield text[start:start + size]


class serverApi:
    """可嵌入或独立运行的 FastAPI 服务。"""

    def __init__(self, manager: ModelsMgr, handler: MsgHandler | None = None) -> None:
        self.manager = manager
        self.handler = handler or MsgHandler(manager)
        self.queue = self.handler  # 兼容外部代码按 queue 属性访问 handler（如无外部引用可删除）
        self._server: Any = None
        self.app = self._create_app()

    def _server_settings(self) -> dict[str, Any]:
        """合并根节点与 settings 节点下的 server 配置（settings 优先，兼容旧版把 server 放在根节点的写法）。"""
        config = getattr(self.manager, "_config", {})
        root = config.get("server", {}) if isinstance(config, dict) else {}
        settings = self.manager.settings.get("server", {})
        return {**(root if isinstance(root, dict) else {}), **(settings if isinstance(settings, dict) else {})}

    def _port(self) -> int:
        settings = self._server_settings()
        return int(settings.get("api_port", settings.get("port", 8000)))

    def _require_api_key(self, x_api_key: str | None = Header(default=None),
                          authorization: str | None = Header(default=None)) -> None:
        """FastAPI 依赖：校验请求头中的 API Key；未配置 api_key 时放行所有请求。"""
        expected = str(self._server_settings().get("api_key", ""))
        if not expected:
            return
        bearer = authorization or ""
        token = x_api_key or (bearer[7:].strip() if bearer.lower().startswith("bearer ") else None)
        if token != expected:
            raise HTTPException(status_code=401, detail="API key 无效")

    @asynccontextmanager
    async def _lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        # 空闲回收由 main.py 独立启动，纯 Console 模式（AI_KAPI=false）同样需要。
        yield

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="AI Multi-Model Service", lifespan=self._lifespan)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._server_settings().get("allow_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/v1/health")  # 连接检测
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/", response_model=None)  # 服务状态、OpenAI Base URL 和监听端口
        async def root() -> Any:
            return JSONResponse({"status": "ok", "openai_base_url": "/v1", "port": self._port()})

        @app.api_route("/v1", methods=["GET", "HEAD"])
        async def api_root() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models(_: None = Depends(self._require_api_key)) -> dict[str, Any]:
            now = int(time.time())
            return {"object": "list", "data": [
                {"id": model_id, "object": "model", "created": now, "owned_by": "local"}
                for model_id in self.manager.specs]}

        @app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(request: OpenAIChatRequest,
                                    _: None = Depends(self._require_api_key)) -> Any:
            if request.function_call is not None or request.functions:
                raise HTTPException(status_code=400,
                                     detail="旧版 functions/function_call 不受支持，请使用 tools/tool_choice")

            messages = request.messages or []
            is_chat = request.message_id == 0
            if is_chat and not request.model:
                raise HTTPException(status_code=400, detail="model 不能为空")
            if is_chat and not messages:
                raise HTTPException(status_code=400, detail="messages 不能为空")
            if is_chat and _has_non_text_content(messages):
                raise HTTPException(status_code=400, detail="当前服务仅支持文本消息")

            # 单流程：不再判断"要不要走结构化"，聊天请求一律把 messages
            # 原样交给 MsgHandler；是否调用工具完全由 tools/tool_choice 决定。
            try:
                result = await self.handler.handle(
                    request.message_id, request.args, model=request.model,
                    prompt=_chat_prompt(messages, request.system_prompt) if is_chat else "",
                    deploy=_generation_params(request) if is_chat else {},
                    think=_resolve_think_level(request) if is_chat else 0,
                    source="openwebui",
                    messages=messages if is_chat else None,
                    tools=request.tools, tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls is not False,
                    system_prompt=request.system_prompt or "")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if result.get("status") != "ok":
                status = {"request": 400, "tool_output": 502}.get(result.get("error_type"), 500)
                raise HTTPException(status_code=status, detail=result.get("response", "模型生成失败"))

            payload = _build_payload(result, request.model)
            if not request.stream:
                return payload

            tool_calls = result.get("tool_calls") or []
            content = "" if tool_calls else result["response"]
            chunk_size = request.stream_delta_chunk_size or len(content) or 1
            base = {"id": payload["id"], "object": "chat.completion.chunk",
                    "created": payload["created"], "model": payload["model"]}
            finish_reason = payload["choices"][0]["finish_reason"]

            async def events() -> AsyncIterator[str]:
                """把一次性生成结果按 OpenAI SSE 协议重放为分块流；后续 Loader 支持真流式后可在此接入。"""

                def chunk(delta: dict[str, Any], reason: str | None = None) -> str:
                    return _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})

                if tool_calls:
                    initial = [{"index": i, "id": call["id"], "type": "function",
                                "function": {"name": call["function"]["name"], "arguments": ""}}
                               for i, call in enumerate(tool_calls)]
                    yield chunk({"role": "assistant", "tool_calls": initial})
                    for call_index, call in enumerate(tool_calls):
                        for piece in _chunks(call["function"]["arguments"], chunk_size):
                            yield chunk({"tool_calls": [{"index": call_index, "function": {"arguments": piece}}]})

                for index, piece in enumerate(_chunks(content, chunk_size)):
                    yield chunk({"content": piece, **({"role": "assistant"} if index == 0 else {})})

                yield chunk({}, finish_reason)
                if (request.stream_options or {}).get("include_usage"):
                    yield _sse({**base, "choices": [], "usage": payload["usage"]})
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        return app

    async def run(self) -> None:
        import uvicorn
        settings = self._server_settings()
        config = uvicorn.Config(self.app, host=str(settings.get("host", "0.0.0.0")),
                                 port=self._port(), log_level="info")
        self._server = uvicorn.Server(config)
        await self._server.serve()


web = serverApi
