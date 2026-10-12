from __future__ import annotations

import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, StreamingResponse

from .clientAdapter import clientAdapter
from .common import error, info
from ..loader.models_mgr import ModelsMgr
from ..msgHandler import AdminRequest, ChatRequest, MsgHandler


# 聊天结构
class openChatReq(BaseModel):
    model_config = ConfigDict(extra="allow")
    # 服务层路由
    message_id: int = 0
    args: Any = None
    # 会话内容
    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    system_prompt: str | None = None
    # 输出与控制
    stream: bool = False
    stream_delta_chunk_size: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


def _to_chat_request(request: openChatReq) -> ChatRequest:
    raw_dict = request.model_dump(exclude_unset=True)
    model = raw_dict.pop("model", "")
    messages = raw_dict.pop("messages", [])
    raw_dict.pop("message_id", None)
    raw_dict.pop("stream", None)
    raw_dict.pop("stream_delta_chunk_size", None)

    return ChatRequest(
        model=model,
        messages=messages,
        deploy=raw_dict,
    )


def _build_payload(result: dict[str, Any], model_id: str) -> dict[str, Any]:
    tool_calls = result.get("tool_calls") or []
    usage = result.get("usage", {})
    finish_reason = "tool_calls" if tool_calls else result.get("finish_reason", "stop")
    message: dict[str, Any] = {"role": "assistant", "content": None if tool_calls else result.get("response", "")}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.get("model") or model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {**usage, "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)},
    }


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    yield
    # FastAPI 服务退出时，安全卸载所有后端常驻模型进程
    mgr = getattr(app.state, "manager", None)
    if mgr and hasattr(mgr, "unload_all"):
        info("FastAPI 停机中，正在销毁所有常驻推理引擎 Server 进程...")
        mgr.unload_all()


class serverApi:
    """FastAPI 服务。"""

    def __init__(self, manager: ModelsMgr, handler: MsgHandler | None = None) -> None:
        self.manager = manager
        self.handler = handler or MsgHandler(manager)
        self.app = self._create_app()
        self.app.state.manager = self.manager
        self._server: Any = None

    def _server_settings(self) -> dict[str, Any]:
        return self.manager.settings.get("server", {})

    def _port(self) -> int:
        return int(self._server_settings().get("api_port", 8666))

    def _require_api_key(
        self,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> bool:
        """API Key 鉴权：未配置 key 时直接放行返回 True；配置时验证一致性。"""
        expected = str(self._server_settings().get("api_key", "")).strip()
        if not expected:
            return True

        bearer = authorization or ""
        token = x_api_key or (bearer[7:].strip() if bearer.lower().startswith("bearer ") else None)
        if not token or token != expected:
            raise HTTPException(status_code=401, detail="API key 无效或未提供")
        return True

    # 聊天指令路由
    async def _chat(self, request: openChatReq, client_type: str = "openai", ua: str = "") -> Any:
        if request.message_id != 0:
            admin_req = AdminRequest(
                message_id=request.message_id,
                model=request.model,
                args=request.args,
            )
            return await self.handler.admin(admin_req, source=client_type)
        info("[DBG] 入站请求", f"client={client_type}", f"stream={request.stream}", f"消息数={len(request.messages)}",
             f"tools数={len(request.tools or [])}", "额外字段=", sorted((request.model_extra or {}).keys()))
        if not request.model or not request.messages:
            raise HTTPException(status_code=400, detail="model 和 messages 不能为空")

        result = await self.handler.chat(_to_chat_request(request), source=f"{client_type}/{ua[:30]}")
        payload = _build_payload(result, request.model)

        outbound_res = clientAdapter.outbound(
            openai_resp=payload,
            client_type=client_type,
            stream=request.stream,
            chunk_size=request.stream_delta_chunk_size
        )

        if request.stream:
            return StreamingResponse(outbound_res, media_type="text/event-stream")
        return outbound_res

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="AI Multi-Model Service", lifespan=_lifespan)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/v1/models")
        async def models(raw_req: Request, _: bool = Depends(self._require_api_key)) -> dict[str, Any]:
            now = int(time.time())
            info("[DBG] 模型列表请求", f"ua={raw_req.headers.get('user-agent', '')[:30]}", "返回=", list(self.manager.specs))
            return {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "local"}
                    for m in self.manager.specs
                ],
            }

        @app.exception_handler(Exception)
        async def unhandled_error(raw_req: Request, exc: Exception) -> JSONResponse:
            error("[请求异常]", raw_req.method, raw_req.url.path, f"{type(exc).__name__}: {exc}",
                  "\n" + "".join(traceback.format_exception(exc)))
            body = {"type": "error", "error": {"type": "api_error", "message": f"{type(exc).__name__}: {exc}"}}
            return JSONResponse(status_code=500, content=body)

        # 1. OpenAI 协议端点
        @app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(request: openChatReq, raw_req: Request, _: bool = Depends(self._require_api_key)) -> Any:
            return await self._chat(request, client_type="openai", ua=raw_req.headers.get("user-agent", ""))

        # 2. Claude / Anthropic 协议端点
        @app.post("/v1/messages", response_model=None)
        async def claude_messages(raw_req: Request, _: bool = Depends(self._require_api_key)) -> Any:
            try:
                body = await raw_req.json()
            except Exception:
                raise HTTPException(status_code=400, detail="无效的 JSON 请求体")
            info("[DBG] Claude原始system[:120]", repr(body.get("system"))[:120])
            info("[DBG] Claude入站", f"model={body.get('model')}", f"stream={body.get('stream')}",
                 f"原始角色序列={[m.get('role') for m in body.get('messages', [])]}", f"tools数={len(body.get('tools') or [])}",
                 f"字段={sorted(body)}", f"thinking={body.get('thinking')}", f"output_config={body.get('output_config')}")
            openai_body = clientAdapter.inbound_to_openai(body, client_type="claude")
            info("[DBG] Claude转换后system[:120]", repr(openai_body["messages"][0])[:120])
            info("[DBG] Claude转换后角色序列", [m.get("role") for m in openai_body["messages"]])
            request = openChatReq(**openai_body)
            return await self._chat(request, client_type="claude", ua=raw_req.headers.get("user-agent", ""))

        return app

    async def run(self) -> None:
        import uvicorn
        settings = self._server_settings()
        config = uvicorn.Config(
            self.app,
            host=str(settings.get("host", "0.0.0.0")),
            port=self._port(),
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None
        await self._server.serve()