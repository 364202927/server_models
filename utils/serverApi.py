"""FastAPI 服务封装。

服务对象负责创建路由、鉴权和单用户排队；``main.py`` 负责启动它并驱动空闲回收。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, StreamingResponse

from ..loader.models_mgr import ModelsMgr
from ..msgHandler import MsgHandler


class ChatRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    think: int = Field(default=0, ge=0, le=5)
    special: int = Field(default=0, ge=0)
    stream: bool = False
    deploy: dict[str, Any] = Field(default_factory=dict)


class MessageRequest(BaseModel):
    id: int
    args: Any = None


class OpenAIChatRequest(BaseModel):
    """OpenAI/Open WebUI 文本聊天请求；后端不支持的字段保持兼容接收。"""

    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    logit_bias: dict[str, float] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    tfs_z: float | None = None
    mirostat: int | None = None
    mirostat_eta: float | None = None
    mirostat_tau: float | None = None
    reasoning_effort: str | int | None = None
    think: bool | int | None = None
    system_prompt: str | None = None
    stream_delta_chunk_size: int | None = Field(default=None, ge=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    custom_parameters: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    function_call: Any = None
    context_compression_threshold: int | float | None = None
    format: Any = None
    num_keep: int | None = None
    num_ctx: int | None = None
    num_batch: int | None = None
    num_thread: int | None = None
    num_gpu: int | None = None
    keep_alive: Any = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None


_REQUEST_GENERATION_FIELDS = {
    "temperature", "top_p", "top_k", "min_p", "max_tokens", "stop",
    "seed", "logit_bias", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "repeat_penalty", "repeat_last_n", "tfs_z",
    "mirostat", "mirostat_eta", "mirostat_tau",
}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content
                       if isinstance(item, dict) and item.get("type") == "text")
    return str(content or "")


def _reasoning_level(value: str | int | None) -> int:
    if isinstance(value, int):
        return max(0, min(5, value))
    return {"none": 0, "low": 1, "medium": 3, "high": 5}.get(str(value).lower(), 0)


def _chat_prompt(messages: list[dict[str, Any]], system_prompt: str | None) -> str:
    items = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    return "\n".join(f"{item.get('role', 'user')}: {_message_text(item.get('content'))}"
                     for item in items if isinstance(item, dict))


def _generation_params(request: OpenAIChatRequest) -> dict[str, Any]:
    values = {key: getattr(request, key) for key in _REQUEST_GENERATION_FIELDS
              if getattr(request, key) is not None}
    values.update({key: value for key, value in request.extra_body.items()
                   if key in _REQUEST_GENERATION_FIELDS and key not in values})
    values.update({key: value for key, value in request.custom_parameters.items()
                   if key in _REQUEST_GENERATION_FIELDS and key not in values})
    if request.max_tokens is not None and request.max_completion_tokens is not None \
            and request.max_tokens != request.max_completion_tokens:
        raise HTTPException(status_code=400, detail="max_tokens 与 max_completion_tokens 冲突")
    if request.max_tokens is None and request.max_completion_tokens is not None:
        values["max_tokens"] = request.max_completion_tokens
    if isinstance(values.get("stop"), str):
        values["stop_sequences"] = [values.pop("stop")]
    elif "stop" in values:
        values["stop_sequences"] = values.pop("stop")
    if values.get("repeat_penalty") is not None:
        values["repetition_penalty"] = values.pop("repeat_penalty")
    if request.repeat_penalty is not None:
        values["repetition_penalty"] = request.repeat_penalty
    if request.repetition_penalty is not None:
        values["repetition_penalty"] = request.repetition_penalty
    return values


def response(model: str, special: int, status: str, value: Any, **extra: Any) -> dict[str, Any]:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return {"request_id": str(uuid.uuid4()), "status": status, "model": model,
            "special": special, "response": payload, **extra}


class serverApi:
    "可嵌入或独立运行的 FastAPI 服务"

    def __init__(self, manager: ModelsMgr, handler: MsgHandler | None = None) -> None:
        self.manager = manager
        self.handler = handler or MsgHandler(manager)
        self.queue = self.handler
        self._server: Any = None
        self.app = self._create_app()

    def _port(self) -> int:
        settings = self._server_settings()
        return int(settings.get("api_port", settings.get("port", 8000)))

    def _server_settings(self) -> dict[str, Any]:
        """读取服务配置；兼容旧配置把 ``server`` 放在根节点的写法。"""
        settings = self.manager.settings.get("server", {})
        config = getattr(self.manager, "_config", {})
        root_settings = config.get("server", {}) if isinstance(config, dict) else {}
        merged: dict[str, Any] = {}
        if isinstance(root_settings, dict):
            merged.update(root_settings)
        if isinstance(settings, dict):
            merged.update(settings)
        return merged

    def _check_key(self, provided: str | None, authorization: str | None = None) -> None:
        expected = str(self._server_settings().get("api_key", ""))
        bearer = authorization or ""
        if not provided and bearer.lower().startswith("bearer "):
            provided = bearer[7:].strip()
        if expected and provided != expected:
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

        @app.get("/v1/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/", response_model=None)
        async def root() -> Any:
            return JSONResponse({"status": "ok", "openai_base_url": "/v1", "port": self._port()})

        @app.api_route("/v1", methods=["GET", "HEAD"])
        async def api_root() -> dict[str, str]:
            """Open WebUI 会先用 HEAD /v1 探测 OpenAI 服务是否可达。"""
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models(
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            """Open WebUI 首次连接时调用的模型列表接口。"""
            self._check_key(x_api_key, authorization)
            now = int(time.time())
            return {"object": "list", "data": [
                {"id": model_id, "object": "model", "created": now,
                 "owned_by": "local"}
                for model_id in self.manager.specs
            ]}

        @app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(
            request: OpenAIChatRequest,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ) -> Any:
            """将 OpenAI 格式转换为共享 MsgHandler。"""
            self._check_key(x_api_key, authorization)
            if not request.model:
                raise HTTPException(status_code=400, detail="model 不能为空")
            messages = request.messages or []
            if any(isinstance(item.get("content"), list) and
                   any(part.get("type") != "text" for part in item["content"]
                       if isinstance(part, dict)) for item in messages if isinstance(item, dict)):
                raise HTTPException(status_code=400, detail="当前服务仅支持文本消息")
            prompt = _chat_prompt(messages, request.system_prompt)
            deploy = _generation_params(request)
            result = await self.handler.handle(
                0, {"model": request.model, "prompt": prompt, "deploy": deploy},
                model=request.model, prompt=prompt, deploy=deploy,
                think=(int(request.think) if isinstance(request.think, bool) else request.think)
                if request.think is not None else _reasoning_level(request.reasoning_effort),
                source="openwebui",
            )
            if result.get("status") != "ok":
                raise HTTPException(status_code=500, detail=result.get("response", "模型生成失败"))
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["response"]},
                             "finish_reason": "stop"}],
                "usage": {**result.get("usage", {}),
                          "total_tokens": result.get("usage", {}).get("prompt_tokens", 0)
                          + result.get("usage", {}).get("completion_tokens", 0)},
            }
            if not request.stream:
                return payload

            async def events() -> AsyncIterator[str]:
                # Generation is currently one-shot; SSE keeps clients compatible
                # and can become token streaming when loaders expose an iterator.
                content = result["response"]
                size = request.stream_delta_chunk_size or len(content) or 1
                for index in range(0, len(content), size):
                    chunk = {"id": payload["id"], "object": "chat.completion.chunk",
                             "created": payload["created"], "model": request.model,
                             "choices": [{"index": 0, "delta": {
                                 **({"role": "assistant"} if index == 0 else {}),
                                 "content": content[index:index + size]},
                                 "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                end = {"id": payload["id"], "object": "chat.completion.chunk",
                       "created": payload["created"], "model": request.model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
                if (request.stream_options or {}).get("include_usage"):
                    usage = {"id": payload["id"], "object": "chat.completion.chunk",
                             "created": payload["created"], "model": request.model,
                             "choices": [], "usage": payload["usage"]}
                    yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        @app.post("/v1/chat")
        async def chat(
            request: ChatRequest,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            self._check_key(x_api_key, authorization)
            if request.stream:
                return response(request.model, request.special, "error", "stream 暂未实现")
            return await self.handler.handle(request.special, {
                "model": request.model, "prompt": request.prompt, "deploy": request.deploy,
            }, model=request.model, prompt=request.prompt, think=request.think,
                deploy=request.deploy, source="api")

        @app.post("/api/postMessage")
        async def post_message(
            message: MessageRequest,
            x_api_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            self._check_key(x_api_key, authorization)
            return await self.handler.handle(message.id, message.args, source="api")

        return app

    async def run(self) -> None:
        """按照配置启动 Uvicorn，供 ``asyncio.run(serverApi.run())`` 调用。"""
        import uvicorn

        settings = self._server_settings()
        config = uvicorn.Config(self.app, host=str(settings.get("host", "0.0.0.0")),
                                port=self._port(), log_level="info")
        self._server = uvicorn.Server(config)
        await self._server.serve()


# 兼容旧代码：新入口使用 ``serverApi``，旧调用仍可使用 ``ModelServer``/``web``。
# ModelServer = serverApi
web = serverApi
