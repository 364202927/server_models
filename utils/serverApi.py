"""FastAPI 服务封装。

服务对象负责创建路由、鉴权、单用户排队和后台空闲模型回收；``main.py`` 只负责启动它。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TypeVar

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..hardware import detect_hardware
from ..loader.models_mgr import ModelsMgr


T = TypeVar("T")


class RequestQueue:
    """单用户 FIFO 请求队列。

    队列和 API 生命周期绑定，确保模型加载、休眠、卸载及生成不会并发
    操作同一个 GPU/CPU 资源。这样可以避免单用户场景下的竞态和显存峰值。
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending = 0

    @property
    def length(self) -> int:
        return self._pending

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        self._pending += 1
        try:
            async with self._lock:
                return await operation()
        finally:
            self._pending -= 1


class ChatRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    think: int = Field(default=0, ge=0, le=5)
    special: int = Field(default=0, ge=0)
    stream: bool = False
    deploy: dict[str, Any] = Field(default_factory=dict)


def response(model: str, special: int, status: str, value: Any, **extra: Any) -> dict[str, Any]:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return {"request_id": str(uuid.uuid4()), "status": status, "model": model,
            "special": special, "response": payload, **extra}


class serverApi:
    """可嵌入或独立运行的 FastAPI 服务，写法对应旧项目 ``webPort.web``。"""

    def __init__(self, manager: ModelsMgr) -> None:
        self.manager = manager
        self.queue = RequestQueue()
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
            return await self.queue.run(lambda: self._handle_request(request))

        return app

    async def _handle_request(self, request: ChatRequest) -> dict[str, Any]:
        try:
            if request.special == 1001:
                value = self.manager.status()
                value["queue_length"] = self.queue.length
                return response(request.model, request.special, "ok", value)
            if request.special == 1002:
                return response(request.model, request.special, "ok", detect_hardware().to_dict())
            if request.special == 1003:
                return response(request.model, request.special, "ok", {"models": list(self.manager.specs)})
            if request.special in {1004, 1005}:
                ok = (self.manager.sleep(request.model) if request.special == 1004
                      else self.manager.unload(request.model))
                return response(request.model, request.special, "ok" if ok else "error",
                                "操作成功" if ok else "操作失败")
            if request.special == 1006:
                await asyncio.to_thread(self.manager.ensure_loaded, request.model)
                return response(request.model, request.special, "ok", "操作成功")

            # API 层只负责解析 deploy 并交给 ModelsMgr；models.json 的首次加载
            # 参数回写只发生在 ModelsMgr.ensure_loaded 的成功路径。
            load_changes = {key: value for key, value in request.deploy.items() if key in {
                "engine", "dtype", "context_length", "gpu_offload_layers", "batch_size",
                "flash_attention", "draft_model", "speculative_decoding", "tensor_parallel",
                "gpu_split", "trust_remote_code", "quantization",
            }}
            if load_changes:
                await asyncio.to_thread(self.manager.reconfigure, request.model, load_changes)
            params = dict(self.manager.settings.get("generation", {}))
            params.update({key: value for key, value in request.deploy.items() if key in {
                "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "stop_sequences",
            }})
            if "max_tokens" in params:
                params["max_new_tokens"] = params.pop("max_tokens")
            prompt = request.prompt
            if request.think:
                prompt = f"请以推理等级 {request.think}/5 分析后给出最终答案。\n\n{prompt}"
            result = await asyncio.to_thread(self.manager.generate, request.model, prompt, **params)
            return response(request.model, 0, "ok", result.text,
                            usage={"prompt_tokens": result.prompt_tokens,
                                   "completion_tokens": result.tokens_generated,
                                   "time_seconds": result.time_seconds,
                                   "tokens_per_second": result.tokens_per_second})
        except (KeyError, RuntimeError, MemoryError, ValueError) as exc:
            return response(request.model, request.special, "error", str(exc))

    async def run(self) -> None:
        """按照配置启动 Uvicorn，供 ``asyncio.run(serverApi.run())`` 调用。"""
        import uvicorn

        settings = self.manager.settings.get("server", {})
        config = uvicorn.Config(self.app, host=str(settings.get("host", "0.0.0.0")),
                                port=int(settings.get("port", 8000)), log_level="info")
        self._server = uvicorn.Server(config)
        await self._server.serve()


# 兼容旧代码：新入口使用 ``serverApi``，旧调用仍可使用 ``ModelServer``/``web``。
ModelServer = serverApi
web = serverApi
