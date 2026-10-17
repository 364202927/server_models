from __future__ import annotations

import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from urllib.parse import quote

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, StreamingResponse

from .clientAdapter import clientAdapter
from .common import error, info
from ..loader.models_mgr import models_mgr
from ..msgHandler import adminReq, chatReq, msgHandler


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
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


def _2chatReq(request: openChatReq) -> chatReq:
    raw_dict = request.model_dump(exclude_unset=True)
    model = raw_dict.pop("model", "")
    messages = raw_dict.pop("messages", [])
    raw_dict.pop("message_id", None)
    raw_dict.pop("stream", None)
    raw_dict.pop("stream_delta_chunk_size", None)
    raw_dict.pop("stream_options", None)

    return chatReq(
        model=model,
        messages=messages,
        deploy=raw_dict,
    )


class gateway:
    """FastAPI 网关服务。"""

    def __init__(self, handler: msgHandler | None = None) -> None:
        self.handler = handler or msgHandler()
        self.app = self._create_app()
        self._server: Any = None

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None]:
        yield
        # FastAPI 服务退出时，安全卸载所有后端常驻模型进程
        info("FastAPI 停机中，正在销毁所有常驻推理引擎 Server 进程...")
        models_mgr.unload_all()

    def _server_settings(self) -> dict[str, Any]:
        return models_mgr.settings.get("server", {})

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
    async def _dispatch(self, request: openChatReq, client_type: str = "openai", ua: str = "") -> Any:
        if request.message_id != 0:#指令处理
            admin_req = adminReq(
                message_id=request.message_id,
                model=request.model,
                args=request.args,
            )
            return await self.handler.admin(admin_req, source=client_type)
        # 聊天处理
        if not request.model or not request.messages:
            raise HTTPException(status_code=400, detail="model 和 messages 不能为空")
        result = await self.handler.chat(_2chatReq(request), source=f"{client_type}/{ua[:60]}", client=clientAdapter.detectClient(ua))
        if result.get("status") == "error":
            raise HTTPException(status_code=int(result.get("code", 500)), detail=result.get("response", "未知错误"))
        warnings = result.pop("warnings", None) or []
        include_usage = bool((request.stream_options or {}).get("include_usage"))
        outbound_res = clientAdapter.outbound(
            openai_resp=result,
            client_type=client_type,
            stream=request.stream,
            chunk_size=request.stream_delta_chunk_size,
            include_usage=include_usage,
        )
        # 调度告警走响应头：不进正文，不改变客户端回传的历史前缀（HTTP 头只能是 latin-1，中文告警用 URL 编码）
        headers = {"X-Gateway-Warning": quote("; ".join(warnings))} if warnings else None
        if request.stream:
            return StreamingResponse(outbound_res, media_type="text/event-stream", headers=headers)
        return JSONResponse(content=outbound_res, headers=headers) if headers else outbound_res

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="AI Multi-Model Service", lifespan=self._lifespan)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/v1/models")
        async def models(_: bool = Depends(self._require_api_key)) -> dict[str, Any]:
            now = int(time.time())
            return {
                "object": "list",
                "data": [{"id": m, "object": "model", "created": now, "owned_by": "local"} for m in models_mgr.specs],
            }

        @app.exception_handler(Exception)
        async def unhandled_error(raw_req: Request, exc: Exception) -> JSONResponse:
            error("[请求异常]", raw_req.method, raw_req.url.path, f"{type(exc).__name__}: {exc}", "\n" + "".join(traceback.format_exception(exc)))
            body = {"type": "error", "error": {"type": "api_error", "message": f"{type(exc).__name__}: {exc}"}}
            return JSONResponse(status_code=500, content=body)

        # 1. OpenAI 协议端点
        @app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(request: openChatReq, raw_req: Request, _: bool = Depends(self._require_api_key)) -> Any:
            return await self._dispatch(request, client_type="openai", ua=raw_req.headers.get("user-agent", ""))

        # 2. Claude / Anthropic 协议端点
        @app.post("/v1/messages", response_model=None)
        async def claude_messages(raw_req: Request, _: bool = Depends(self._require_api_key)) -> Any:
            try:
                body = await raw_req.json()
            except Exception:
                raise HTTPException(status_code=400, detail="无效的 JSON 请求体")
            info("[DBG] Claude入站", f"stream={body.get('stream')}", f"字段={sorted(body)}", f"thinking={body.get('thinking')}", f"output_config={body.get('output_config')}")
            request = openChatReq(**clientAdapter.inbound_to_openai(body, client_type="claude"))
            return await self._dispatch(request, client_type="claude", ua=raw_req.headers.get("user-agent", ""))

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
