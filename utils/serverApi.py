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
from ..msgHandler import AdminRequest, ChatRequest, MsgHandler

#聊天结构
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
    # 基础逻辑字段单独处理
    model = raw_dict.pop("model", "")
    messages = raw_dict.pop("messages", [])
    raw_dict.pop("message_id", None)
    raw_dict.pop("stream", None)
    raw_dict.pop("stream_delta_chunk_size", None)

    return ChatRequest(
        model=model,
        messages=messages,
        deploy=raw_dict,  # 所有剩余参数作为可动态过滤的部署/生成参数包
    )


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
    """FastAPI 服务。"""

    def __init__(self, manager: ModelsMgr, handler: MsgHandler | None = None) -> None:
        self.manager = manager
        self.handler = handler or MsgHandler(manager)
        self.app = self._create_app()
        self._server: Any = None

    def _server_settings(self) -> dict[str, Any]:
        return self.manager.settings.get("server", {})

    def _port(self) -> int:
        return int(self._server_settings().get("api_port", 8666))

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
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/v1/models")
        async def models(_: None = Depends(self._require_api_key)) -> dict[str, Any]:
            now = int(time.time())
            return {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "local"}
                    for m in self.manager.specs
                ],
            }

        @app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(request: openChatReq, _: None = Depends(self._require_api_key)) -> Any:
            if request.message_id != 0:
                admin_req = AdminRequest(
                    message_id=request.message_id,
                    model=request.model,
                    args=request.args,
                )
                return await self.handler.admin(admin_req, source="openwebui")

            if not request.model or not request.messages:
                raise HTTPException(status_code=400, detail="model 和 messages 不能为空")

            result = await self.handler.chat(_to_chat_request(request), source="openwebui")
            payload = _build_payload(result, request.model)

            if not request.stream:
                return payload

            tool_calls = result.get("tool_calls") or []
            content = "" if tool_calls else result["response"]
            chunk_size = request.stream_delta_chunk_size or len(content) or 1
            base = {
                "id": payload["id"],
                "object": "chat.completion.chunk",
                "created": payload["created"],
                "model": payload["model"],
            }
            finish_reason = payload["choices"][0]["finish_reason"]

            async def events() -> AsyncGenerator[str]:
                def _sse(d: dict[str, Any]) -> str:
                    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
                for i in range(0, len(content), chunk_size):
                    piece = content[i:i + chunk_size]
                    yield _sse({**base, "choices": [{"index": 0, "delta": {"content": piece, "role": "assistant"}, "finish_reason": None}]})
                yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        return app

    async def run(self) -> None:
        """启动 uvicorn 服务供 main.py 异步任务调用"""
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