from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, StreamingResponse

from ..loader.models_mgr import ModelsMgr
from ..msgHandler import AdminRequest, ChatRequest, MsgHandler, normalize_generation_params, resolve_think_level


_GENERATION_FIELDS = {
    "temperature", "top_p", "top_k", "min_p", "max_tokens", "stop",
    "seed", "logit_bias", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "repeat_penalty", "repeat_last_n", "tfs_z",
    "mirostat", "mirostat_eta", "mirostat_tau",
}


class openChatReq(BaseModel):
    model_config = ConfigDict(extra="allow")

    # 服务层路由
    message_id: int = 0
    args: Any = None

    # 会话内容
    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    system_prompt: str | None = None

    # 输出方式
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    stream_delta_chunk_size: int | None = Field(default=None, ge=0)

    # 采样参数
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    repetition_penalty: float | None = None
    repeat_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    repeat_last_n: int | None = None
    tfs_z: float | None = None
    mirostat: int | None = None
    mirostat_eta: float | None = None
    mirostat_tau: float | None = None

    # 推理强度
    reasoning_effort: str | int | None = None
    think: bool | int | None = None

    # 透传参数
    extra_body: dict[str, Any] = Field(default_factory=dict)
    custom_parameters: dict[str, Any] = Field(default_factory=dict)

    # 工具调用
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None

    # 废弃工具字段
    function_call: Any = None
    functions: list[dict[str, Any]] | None = None


def _to_chat_request(request: openChatReq) -> ChatRequest:
    if request.max_tokens is not None and request.max_completion_tokens not in (None, request.max_tokens):
        raise HTTPException(status_code=400, detail="max_tokens 与 max_completion_tokens 冲突")

    direct_fields = {f: getattr(request, f) for f in _GENERATION_FIELDS if getattr(request, f) is not None}
    deploy = normalize_generation_params(direct_fields, request.extra_body, request.custom_parameters)
    if "max_tokens" not in deploy and request.max_completion_tokens is not None:
        deploy["max_tokens"] = request.max_completion_tokens

    return ChatRequest(
        model=request.model,
        messages=request.messages,
        think=resolve_think_level(think=request.think, reasoning_effort=request.reasoning_effort),
        deploy=deploy,
        tools=request.tools or [],
        tool_choice=request.tool_choice,
        parallel_tool_calls=request.parallel_tool_calls is not False,
        system_prompt=request.system_prompt or "",
    )

def _to_admin_request(request: openChatReq) -> AdminRequest:
    payload = request.args if isinstance(request.args, dict) else {}
    generation = dict(payload.get("generation", payload.get("deploy", {})))
    return AdminRequest(
        message_id=request.message_id,
        model=request.model or str(payload.get("model", "")),
        generation=generation,)

def _build_payload(result: dict[str, Any], model_id: str) -> dict[str, Any]:
    tool_calls = result.get("tool_calls") or []
    usage = result.get("usage", {})
    message: dict[str, Any] = {"role": "assistant", "content": None if tool_calls else result["response"]}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.get("model") or model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": result.get("finish_reason", "stop")}],
        "usage": {**usage, "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)},
    }


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncGenerator[None]:
    yield
class serverApi:
    """可嵌入或独立运行的 FastAPI 服务。"""

    def __init__(self, manager: ModelsMgr, handler: MsgHandler | None = None) -> None:
        self.manager = manager
        self.handler = handler or MsgHandler(manager)
        self.queue = self.handler
        self._server: Any = None
        self.app = self._create_app()

    def _server_settings(self) -> dict[str, Any]:
        config = getattr(self.manager, "_config", {})
        root = config.get("server", {}) if isinstance(config, dict) else {}
        settings = self.manager.settings.get("server", {})
        return {**(root if isinstance(root, dict) else {}), **(settings if isinstance(settings, dict) else {})}

    def _port(self) -> int:
        settings = self._server_settings()
        return int(settings.get("api_port", settings.get("port", 8000)))

    def _require_api_key(
        self,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        expected = str(self._server_settings().get("api_key", ""))
        if not expected:
            return
        bearer = authorization or ""
        token = x_api_key or (bearer[7:].strip() if bearer.lower().startswith("bearer ") else None)
        if token != expected:
            raise HTTPException(status_code=401, detail="API key 无效")

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="AI Multi-Model Service", lifespan=_lifespan)
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
            return {"status": "ok"}

        @app.get("/v1/queue")
        async def queue_status(_: None = Depends(self._require_api_key)) -> dict[str, Any]:
            """查询当前排队长度与运行中任务状态。"""
            return self.handler.queue_info()

        @app.get("/v1/models")
        async def models(_: None = Depends(self._require_api_key)) -> dict[str, Any]:
            now = int(time.time())
            return {
                "object": "list",
                "data": [
                    {"id": model_id, "object": "model", "created": now, "owned_by": "local"}
                    for model_id in self.manager.specs
                ],
            }

        @app.get("/v1/props")
        async def props(model: str | None = None, _: None = Depends(self._require_api_key)) -> dict[str, Any]:
            model_id = model or next(iter(self.manager.specs), "")
            spec = self.manager.specs.get(model_id)
            n_ctx = (spec.load.context_length if spec else None) or 4096
            return {
                "model_path": spec.path if spec else "",
                "total_slots": 1,
                "default_generation_settings": {"n_ctx": n_ctx, "model": model_id},
            }

        @app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(request: openChatReq, _: None = Depends(self._require_api_key)) -> Any:
            async def events() -> AsyncGenerator[str]:
                def _sse(payload: dict[str, Any]) -> str:
                    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                def _chunks(text: str, size: int) -> Iterator[str]:
                    for start in range(0, len(text), size):
                        yield text[start:start + size]
                def chunk(delta: dict[str, Any], reason: str | None = None) -> str:
                    return _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})

                if tool_calls:
                    initial = [{"index": i,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["function"]["name"], "arguments": ""},}
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

            if request.function_call is not None or request.functions:
                raise HTTPException(status_code=400, detail="旧版 functions/function_call 不受支持，请使用 tools/tool_choice")
            is_chat = request.message_id == 0
            if is_chat and not request.model:
                raise HTTPException(status_code=400, detail="model 不能为空")
            if is_chat and not request.messages:
                raise HTTPException(status_code=400, detail="messages 不能为空")
            # 用户id指令
            if not is_chat:
                return await self.handler.admin(_to_admin_request(request), source="openwebui")
            # 正常聊天排队处理
            result = await self.handler.chat(_to_chat_request(request), source="openwebui")
            payload = _build_payload(result, request.model)
            if not request.stream:
                return payload
            # 工具调用
            tool_calls = result.get("tool_calls") or []
            content = "" if tool_calls else result["response"]
            chunk_size = request.stream_delta_chunk_size or len(content) or 1
            base = {"id": payload["id"],
                    "object": "chat.completion.chunk",
                    "created": payload["created"],
                    "model": payload["model"],
                }
            finish_reason = payload["choices"][0]["finish_reason"]
            return StreamingResponse(events(), media_type="text/event-stream")

        return app

    async def run(self) -> None:
        import uvicorn
        settings = self._server_settings()
        config = uvicorn.Config(
            self.app,
            host=str(settings.get("host", "0.0.0.0")),
            port=self._port(),
            log_level="info",  #warning,error就不会打印了
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None
        await self._server.serve()


web = serverApi