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


class OpenAIChatRequest(BaseModel):
    """OpenAI/Open WebUI 文本聊天请求"""
    model_config = ConfigDict(extra="allow")
    #通用 / 服务层
    message_id: int = 0                                     #扩展指令 ID；0 为聊天，1001~1007 为管理指令。
    args: Any = None                                        #管理指令参数；message_id 非 0 时传给共享 MsgHandler。
    model: str = ""                                         #模型id
    messages: list[dict[str, Any]] = Field(default_factory=list)    #OpenAI 消息列表；所有当前后端，仅支持文本内容，随后转换为角色标记文本。
    stream: bool = False                                    #返回兼容 SSE；所有当前后端，但目前是生成完成后分块，并非实时 token 流
    stream_options: dict[str, Any] | None = None            #流式响应选项；当前仅解析 include_usage。
    temperature: float | None = Field(default=None, ge=0)   #采样温度；HF、vLLM、GGUF。
    top_p: float | None = Field(default=None, ge=0, le=1)   #核采样概率阈值；HF、vLLM、GGUF。
    top_k: int | None = Field(default=None, ge=0)           #候选 token 数量；HF、vLLM、GGUF。
    max_tokens: int | None = None                           #最大生成 token 数；HF、vLLM、GGUF。
    max_completion_tokens: int | None = None                #max_tokens 的 OpenAI 别名；HF、vLLM、GGUF，两者冲突时报错。
    stop: str | list[str] | None = None                     #停止序列，字符串或字符串列表；HF、vLLM、GGUF
    seed: int | None = None                                 #固定采样随机种子；HF、vLLM、GGUF。
    repetition_penalty: float | None = None                 #重复惩罚；HF、vLLM、GGUF。
    repeat_penalty: float | None = None                     #repetition_penalty 的 llama.cpp/Ollama 别名；HF、vLLM、GGUF。
    reasoning_effort: str | int | None = None               #推理强度 none/low/medium/high 或 0~5
    think: bool | int | None = None                         #推理开关或 0~5 等级
    system_prompt: str | None = None                        #附加系统提示词；置于 messages 前，所有当前后端。
    stream_delta_chunk_size: int | None = Field(default=None, ge=0) #SSE 文本分块字符数；仅服务层使用。
    extra_body: dict[str, Any] = Field(default_factory=dict)#扩展参数；只提取已登记的生成字段。
    custom_parameters: dict[str, Any] = Field(default_factory=dict) #自定义参数对象；只提取已登记的生成字段
    # HuggingFace Transformers可用
    min_p: float | None = Field(default=None, ge=0, le=1)   #最小概率采样；HF 版本支持时可用，vLLM/GGUF 也会下传。
    # vLLM gguf可用
    frequency_penalty: float | None = None                  #按 token 出现频率施加惩罚；
    presence_penalty: float | None = None                   #按 token 是否出现施加惩罚:
    logit_bias: dict[str, float] | None = None              #按 token ID 调整 logits
    # 仅 GGUF
    repeat_last_n: int | None = None                        #计算重复惩罚时回看的 token 数；
    tfs_z: float | None = None                              #Tail Free Sampling 参数；
    mirostat: int | None = None                             #Mirostat 采样模式
    mirostat_eta: float | None = None                       #Mirostat 学习率；
    mirostat_tau: float | None = None                       #Mirostat 目标熵；
    # Ollama 兼容字段
    format: Any = None                                      # 输出格式；当前无 Ollama Loader，仅接收并忽略。
    num_keep: int | None = None                             # 保留提示 token 数；当前仅接收并忽略。
    num_ctx: int | None = None                              # 上下文长度；当前仅接收并忽略。
    num_batch: int | None = None                            # 批大小；当前仅接收并忽略。
    num_thread: int | None = None                           # CPU 线程数；当前仅接收并忽略。
    num_gpu: int | None = None                              # GPU 层/设备设置；当前仅接收并忽略。
    keep_alive: Any = None                                  # 模型驻留时间；当前仅接收并忽略
    use_mmap: bool | None = None                            # 内存映射开关；当前请求链路仅接收并忽略。
    use_mlock: bool | None = None                           # 内存锁定开关；当前请求链路仅接收并忽略
    #工具调用协议
    tools: list[dict[str, Any]] | None = None               #现代函数工具定义；当前仅支持 type=function。
    tool_choice: Any = None                                 #none/auto/required 或指定函数对象。
    parallel_tool_calls: bool | None = None                 #false 时模型一轮最多返回一个工具调用。
    function_call: Any = None                               #旧版函数调用配置；非空时返回 400。
    functions: list[dict[str, Any]] | None = None           #旧版函数定义；非空时拒绝并提示迁移。
    context_compression_threshold: int | float | None = None#上下文压缩阈值

def _message_text(content: Any) -> str:
    """从 OpenAI 字符串或文本分段中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content
                       if isinstance(item, dict) and item.get("type") == "text")
    return str(content or "")


def _reasoning_level(value: str | int | None) -> int:
    """把 OpenAI 推理强度转换为内部 0~5 等级。"""
    if isinstance(value, int):
        return max(0, min(5, value))
    return {"none": 0, "low": 1, "medium": 3, "high": 5}.get(str(value).lower(), 0)


def _chat_prompt(messages: list[dict[str, Any]], system_prompt: str | None) -> str:
    """把附加系统提示词和多轮消息合并为 Loader 使用的文本提示。"""
    items = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    return "\n".join(f"{item.get('role', 'user')}: {_message_text(item.get('content'))}"
                     for item in items if isinstance(item, dict))

def _generation_params(request: OpenAIChatRequest) -> dict[str, Any]:
    """合并顶层和自定义参数，并统一长度、停止序列及重复惩罚别名。"""
    _REQUEST_GENERATION_FIELDS = {
        "temperature", "top_p", "top_k", "min_p", "max_tokens", "stop",
        "seed", "logit_bias", "frequency_penalty", "presence_penalty",
        "repetition_penalty", "repeat_penalty", "repeat_last_n", "tfs_z",
        "mirostat", "mirostat_eta", "mirostat_tau",
    }

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

        @app.get("/v1/health") #连接检测
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/", response_model=None) #服务状态、OpenAI Base URL 和监听端口。
        async def root() -> Any:
            return JSONResponse({"status": "ok", "openai_base_url": "/v1", "port": self._port()})

        #OpenAI支持协议
        @app.api_route("/v1", methods=["GET", "HEAD"])
        async def api_root() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models(x_api_key: str | None = Header(default=None),authorization: str | None = Header(default=None)) -> dict[str, Any]:
            self._check_key(x_api_key, authorization)
            now = int(time.time())
            return {"object": "list", "data": [{"id": model_id, "object": "model", "created": now,
                                        "owned_by": "local"}
                                        for model_id in self.manager.specs]}

        @app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(request: OpenAIChatRequest,x_api_key: str | None = Header(default=None),authorization: str | None = Header(default=None)) -> Any:
            self._check_key(x_api_key, authorization)
            if request.function_call is not None or request.functions:
                raise HTTPException(status_code=400, detail="旧版 functions/function_call 不受支持，请使用 tools/tool_choice")
            if request.message_id == 0 and not request.model:
                raise HTTPException(status_code=400, detail="model 不能为空")
            if request.message_id == 0 and not request.messages:
                raise HTTPException(status_code=400, detail="messages 不能为空")
            prompt, deploy, think = "", {}, 0
            structured = bool(request.tools or request.tool_choice not in (None, "none") or any(
                isinstance(item, dict) and (item.get("role") == "tool" or item.get("tool_calls") is not None)
                for item in request.messages))
            if request.message_id == 0:
                messages = request.messages or []
                if any(isinstance(item.get("content"), list) and
                       any(part.get("type") != "text" for part in item["content"]
                           if isinstance(part, dict)) for item in messages if isinstance(item, dict)):
                    raise HTTPException(status_code=400, detail="当前服务仅支持文本消息")
                if not structured:
                    prompt = _chat_prompt(messages, request.system_prompt)
                deploy = _generation_params(request)
                think = ((int(request.think) if isinstance(request.think, bool) else request.think)
                         if request.think is not None else _reasoning_level(request.reasoning_effort))
            try:
                result = await self.handler.handle(
                    request.message_id, request.args, model=request.model, prompt=prompt,
                    deploy=deploy, think=think, source="openwebui",
                    messages=messages if request.message_id == 0 and structured else None,
                    tools=request.tools, tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls is not False,
                    system_prompt=request.system_prompt or "")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if result.get("status") != "ok":
                status = {"request": 400, "tool_output": 502}.get(result.get("error_type"), 500)
                raise HTTPException(status_code=status, detail=result.get("response", "模型生成失败"))
            response_model = result.get("model") or request.model
            tool_calls = result.get("tool_calls") or []
            message = {"role": "assistant",
                       "content": None if tool_calls else result["response"]}
            if tool_calls:
                message["tool_calls"] = tool_calls
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": response_model,
                "choices": [{"index": 0, "message": message,
                             "finish_reason": result.get("finish_reason", "stop")}],
                "usage": {**result.get("usage", {}),
                          "total_tokens": result.get("usage", {}).get("prompt_tokens", 0)
                          + result.get("usage", {}).get("completion_tokens", 0)},
            }
            if not request.stream:
                return payload

            async def events() -> AsyncIterator[str]:
                # Generation is currently one-shot; SSE keeps clients compatible
                # and can become token streaming when loaders expose an iterator.
                content = "" if tool_calls else result["response"]
                size = request.stream_delta_chunk_size or len(content) or 1
                if tool_calls:
                    initial_calls = [{"index": index, "id": call["id"], "type": "function",
                                      "function": {"name": call["function"]["name"],
                                                   "arguments": ""}}
                                     for index, call in enumerate(tool_calls)]
                    first = {"id": payload["id"], "object": "chat.completion.chunk",
                             "created": payload["created"], "model": response_model,
                             "choices": [{"index": 0, "delta": {"role": "assistant",
                                          "tool_calls": initial_calls}, "finish_reason": None}]}
                    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                    for call_index, call in enumerate(tool_calls):
                        arguments = call["function"]["arguments"]
                        for index in range(0, len(arguments), size):
                            chunk = {"id": payload["id"], "object": "chat.completion.chunk",
                                     "created": payload["created"], "model": response_model,
                                     "choices": [{"index": 0, "delta": {"tool_calls": [{
                                         "index": call_index,
                                         "function": {"arguments": arguments[index:index + size]},
                                     }]}, "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                for index in range(0, len(content), size):
                    chunk = {"id": payload["id"], "object": "chat.completion.chunk",
                             "created": payload["created"], "model": response_model,
                             "choices": [{"index": 0, "delta": {
                                 **({"role": "assistant"} if index == 0 else {}),
                                 "content": content[index:index + size]},
                                 "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                end = {"id": payload["id"], "object": "chat.completion.chunk",
                       "created": payload["created"], "model": response_model,
                       "choices": [{"index": 0, "delta": {},
                                    "finish_reason": result.get("finish_reason", "stop")}]}
                yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
                if (request.stream_options or {}).get("include_usage"):
                    usage = {"id": payload["id"], "object": "chat.completion.chunk",
                             "created": payload["created"], "model": response_model,
                             "choices": [], "usage": payload["usage"]}
                    yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        return app

    async def run(self) -> None:
        import uvicorn
        settings = self._server_settings()
        config = uvicorn.Config(self.app, host=str(settings.get("host", "0.0.0.0")),
                                port=self._port(), log_level="info")
        self._server = uvicorn.Server(config)
        await self._server.serve()

web = serverApi
