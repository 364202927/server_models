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
from pydantic import BaseModel, Field

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
    """OpenAI/Open WebUI 发送的最小聊天请求结构。"""

    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False


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

    def _check_key(self, provided: str | None) -> None:
        expected = str(self.manager.settings.get("server", {}).get("api_key", ""))
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
            allow_origins=self.manager.settings.get("server", {}).get("allow_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/v1/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
            """Open WebUI 首次连接时调用的模型列表接口。"""
            self._check_key(x_api_key)
            now = int(time.time())
            return {"object": "list", "data": [
                {"id": model_id, "object": "model", "created": now,
                 "owned_by": "local"}
                for model_id in self.manager.specs
            ]}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: OpenAIChatRequest,
            x_api_key: str | None = Header(default=None),
        ) -> dict[str, Any]:
            """将 OpenAI 格式转换为共享 MsgHandler。"""
            self._check_key(x_api_key)
            if request.stream:
                return {"error": {"message": "stream 暂未实现", "type": "不支持的请求"}}
            if not request.model:
                raise HTTPException(status_code=400, detail="model 不能为空")
            messages = request.messages or []
            prompt_parts = [
                f"{item.get('role', 'user')}: {item.get('content', '')}"
                for item in messages if isinstance(item, dict)
            ]
            prompt = "\n".join(prompt_parts)
            deploy = {
                key: value for key, value in {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                }.items() if value is not None
            }
            result = await self.handler.handle(
                0, {"model": request.model, "prompt": prompt, "deploy": deploy},
                model=request.model, prompt=prompt, deploy=deploy, source="openwebui",
            )
            if result.get("status") != "ok":
                raise HTTPException(status_code=500, detail=result.get("response", "模型生成失败"))
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["response"]},
                             "finish_reason": "stop"}],
                "usage": result.get("usage", {}),
            }

        @app.post("/v1/chat")
        async def chat(request: ChatRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
            self._check_key(x_api_key)
            if request.stream:
                return response(request.model, request.special, "error", "stream 暂未实现")
            return await self.handler.handle(request.special, {
                "model": request.model, "prompt": request.prompt, "deploy": request.deploy,
            }, model=request.model, prompt=request.prompt, think=request.think,
                deploy=request.deploy, source="api")

        @app.post("/api/postMessage")
        async def post_message(message: MessageRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
            self._check_key(x_api_key)
            return await self.handler.handle(message.id, message.args, source="api")

        return app

    async def run(self) -> None:
        """按照配置启动 Uvicorn，供 ``asyncio.run(serverApi.run())`` 调用。"""
        import uvicorn

        settings = self.manager.settings.get("server", {})
        config = uvicorn.Config(self.app, host=str(settings.get("host", "0.0.0.0")),
                                port=int(settings.get("port", 8000)), log_level="info")
        self._server = uvicorn.Server(config)
        await self._server.serve()


# 兼容旧代码：新入口使用 ``serverApi``，旧调用仍可使用 ``ModelServer``/``web``。
# ModelServer = serverApi
web = serverApi
