"""FastAPI 服务封装。

服务对象负责创建路由、鉴权、单用户排队和后台空闲模型回收；``main.py`` 只负责启动它。
"""

from __future__ import annotations

import asyncio
import json
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


def response(model: str, special: int, status: str, value: Any, **extra: Any) -> dict[str, Any]:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return {"request_id": str(uuid.uuid4()), "status": status, "model": model,
            "special": special, "response": payload, **extra}


class serverApi:
    """可嵌入或独立运行的 FastAPI 服务，写法对应旧项目 ``webPort.web``。"""

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
        async def reaper() -> None:
            while True:
                await asyncio.sleep(30)
                await asyncio.to_thread(self.manager.reap_idle)

        task = asyncio.create_task(reaper())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

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
