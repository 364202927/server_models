"""多模型 FastAPI 服务入口。"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException

if __package__ in (None, ""):
    # 直接执行 ``python main.py`` 时补入包的父目录，仍使用包内相对导入。
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ai.hardware import detect_hardware
    from ai.loader import ModelsMgr
    from ai.utils.server import ChatRequest, response
else:
    from .hardware import detect_hardware
    from .loader import ModelsMgr
from .utils.server import ChatRequest, response
from .utils.request_queue import RequestQueue


manager = ModelsMgr(str(Path(__file__).resolve().parent / "assets" / "models.json"))
_queue = RequestQueue()


def _check_key(provided: str | None) -> None:
    expected = str(manager.settings.get("server", {}).get("api_key", ""))
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="API key 无效")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    async def reaper() -> None:
        while True:
            await asyncio.sleep(30)
            await asyncio.to_thread(manager.reap_idle)
    task = asyncio.create_task(reaper())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="AI Multi-Model Service", lifespan=lifespan)


@app.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat")
async def chat(request: ChatRequest, x_api_key: str | None = Header(default=None)) -> dict:
    _check_key(x_api_key)
    if request.stream:
        return response(request.model, request.special, "error", "stream 暂未实现")
    async def operation() -> dict:
        if request.special == 1001:
            value = manager.status()
            value["queue_length"] = _queue.length
            return response(request.model, request.special, "ok", value)
        if request.special == 1002:
            return response(request.model, request.special, "ok", detect_hardware().to_dict())
        if request.special == 1003:
            return response(request.model, request.special, "ok", {"models": list(manager.specs)})
        if request.special in {1004, 1005}:
            ok = manager.sleep(request.model) if request.special == 1004 else manager.unload(request.model)
            return response(request.model, request.special, "ok" if ok else "error", "操作成功" if ok else "操作失败")
        if request.special == 1006:
            manager.ensure_loaded(request.model)
            return response(request.model, request.special, "ok", "操作成功")
        try:
            load_changes = {key: value for key, value in request.deploy.items()
                            if key in {"engine", "dtype", "quantization", "max_model_len", "tensor_parallel_size", "trust_remote_code"}}
            if load_changes:
                await asyncio.to_thread(manager.reconfigure, request.model, load_changes)
            params = dict(manager.settings.get("generation", {}))
            params.update({key: value for key, value in request.deploy.items()
                           if key in {"temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "stop_sequences"}})
            if "max_tokens" in params:
                params["max_new_tokens"] = params.pop("max_tokens")
            prompt = request.prompt
            if request.think:
                # think 仅作为推理强度提示；不同引擎不保证存在同名原生参数。
                prompt = f"请以推理等级 {request.think}/5 分析后给出最终答案。\n\n{prompt}"
            result = await asyncio.to_thread(manager.generate, request.model, prompt, **params)
            return response(request.model, 0, "ok", result.text,
                            usage={"prompt_tokens": result.prompt_tokens, "completion_tokens": result.tokens_generated,
                                   "time_seconds": result.time_seconds, "tokens_per_second": result.tokens_per_second})
        except (KeyError, RuntimeError, MemoryError) as exc:
            return response(request.model, 0, "error", str(exc))
    return await _queue.run(operation)


if __name__ == "__main__":
    import uvicorn
    settings = manager.settings.get("server", {})
    uvicorn.run(app, host=str(settings.get("host", "0.0.0.0")), port=int(settings.get("port", 8000)))
