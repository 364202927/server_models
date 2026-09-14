"""FastAPI 服务封装。

服务对象负责创建路由、鉴权和单用户排队；``main.py`` 负责启动它并驱动空闲回收。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

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


# ``desktop`` is an optional lightweight web UI served by this process.  It is
# deliberately not the official Open WebUI backend; that backend can still use
# the OpenAI-compatible routes below when this service runs in ``api`` mode.
_DESKTOP_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Desktop Web UI</title><style>
:root{font:16px system-ui,sans-serif;color:#e8e8e8;background:#171717}body{margin:0;height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px;border-bottom:1px solid #333;display:flex;gap:12px;align-items:center}header strong{margin-right:auto}
select,button,textarea{font:inherit;border:1px solid #444;border-radius:6px;background:#242424;color:inherit;padding:8px}
button{cursor:pointer;background:#4b55c8;border-color:#5963e0}button.secondary{background:#242424}
#messages{flex:1;overflow:auto;padding:24px;display:flex;flex-direction:column;gap:12px}.message{max-width:850px;white-space:pre-wrap;line-height:1.5;padding:12px 15px;border-radius:8px;background:#242424}.user{align-self:flex-end;background:#303b77}
form{display:flex;gap:10px;padding:14px 20px;border-top:1px solid #333}textarea{resize:none;flex:1;min-height:44px}small{opacity:.65}
</style></head><body><header><strong>AI Desktop Web UI</strong><label>模型 <select id="model"></select></label><input id="api-key" type="password" placeholder="API Key（可选）" autocomplete="off"><button class="secondary" id="new">新建聊天</button></header>
<main id="messages"><small>正在加载模型...</small></main><form id="form"><textarea id="prompt" placeholder="输入消息..." required></textarea><button>发送</button></form>
<script>
const model=document.querySelector('#model'),messages=document.querySelector('#messages'),prompt=document.querySelector('#prompt'),apiKey=document.querySelector('#api-key'),history=[];
apiKey.value=sessionStorage.getItem('ai-api-key')||'';
apiKey.onchange=()=>sessionStorage.setItem('ai-api-key',apiKey.value);
function authHeaders(){const key=apiKey.value.trim();return key?{'X-API-Key':key}:{} }
function add(text,kind){const el=document.createElement('div');el.className='message '+kind;el.textContent=text;messages.append(el);messages.scrollTop=messages.scrollHeight;return el}
async function loadModels(){const r=await fetch('/v1/models',{headers:authHeaders()});if(!r.ok)throw Error('模型列表加载失败');const data=await r.json();model.replaceChildren(...(data.data||[]).map(x=>new Option(x.id,x.id)));if(!model.options.length)throw Error('models.json 中没有模型');messages.replaceChildren()}
document.querySelector('#new').onclick=()=>{history.length=0;messages.replaceChildren()};
document.querySelector('#form').onsubmit=async e=>{e.preventDefault();const text=prompt.value.trim();if(!text||!model.value)return;prompt.value='';history.push({role:'user',content:text});add(text,'user');const pending=add('正在生成...','assistant');try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json',...authHeaders()},body:JSON.stringify({model:model.value,messages:history,stream:false})});const data=await r.json();if(!r.ok)throw Error(data.detail||data.error?.message||'请求失败');const answer=data.choices?.[0]?.message?.content||'无返回内容';pending.textContent=answer;history.push({role:'assistant',content:answer})}catch(err){history.pop();pending.textContent='错误：'+err.message}};
loadModels().catch(err=>{messages.replaceChildren();add('错误：'+err.message,'assistant')});
</script></body></html>"""


class serverApi:
    "可嵌入或独立运行的 FastAPI 服务"

    def __init__(self, manager: ModelsMgr, handler: MsgHandler | None = None) -> None:
        self.manager = manager
        self.handler = handler or MsgHandler(manager)
        self.queue = self.handler
        self._server: Any = None
        self.interface = self._get_interface()
        self.app = self._create_app()

    def _get_interface(self) -> str:
        """Return the selected listener/UI mode: ``api`` or ``desktop``."""
        settings = self._server_settings()
        value = os.getenv("AI_SERVER_INTERFACE", settings.get("interface", "api"))
        value = str(value).strip().lower()
        if value not in {"api", "desktop"}:
            raise ValueError("server.interface 必须是 api 或 desktop")
        return value

    def _port(self) -> int:
        settings = self._server_settings()
        key, default = ("desktop_port", 3000) if self.interface == "desktop" else ("api_port", 8000)
        if key in settings:
            return int(settings[key])
        # ``port`` is retained for old configs in API mode only.
        return int(settings["port"]) if self.interface == "api" and "port" in settings else default

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
            if self.interface == "desktop":
                return HTMLResponse(_DESKTOP_HTML)
            return JSONResponse({"status": "ok", "interface": "api", "openai_base_url": "/v1"})

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
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["response"]},
                             "finish_reason": "stop"}],
                "usage": result.get("usage", {}),
            }
            if not request.stream:
                return payload

            async def events() -> AsyncIterator[str]:
                # Generation is currently one-shot; SSE keeps clients compatible
                # and can become token streaming when loaders expose an iterator.
                chunk = {"id": payload["id"], "object": "chat.completion.chunk",
                         "created": payload["created"], "model": request.model,
                         "choices": [{"index": 0, "delta": {"role": "assistant",
                                      "content": result["response"]}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
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
