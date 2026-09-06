"""Console 与 FastAPI 共用的统一消息处理器。"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from .hardware import detect_hardware
from .loader.model_spec import LOAD_KEYS
from .loader.models_mgr import ModelsMgr
from .utils.common import info, log

# 每次请求可覆盖的采样参数；``max_tokens`` 在传给 Loader 前改名为 ``max_new_tokens``。
GENERATION_KEYS = frozenset({
    "temperature", "top_p", "top_k", "repetition_penalty",
    "max_tokens", "stop_sequences", "system_prompt",
})


class MsgHandler:
    """把不同入口的消息串行路由到 ModelsMgr。"""

    def __init__(self, manager: ModelsMgr) -> None:
        self.manager = manager
        self._lock = asyncio.Lock()
        self._pending = 0

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def length(self) -> int:
        return self._pending

    @staticmethod
    def _response(model: str, special: int, status: str, value: Any) -> dict[str, Any]:
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return {"request_id": str(uuid.uuid4()), "status": status, "model": model,
                "special": special, "response": payload}

    async def handle(
        self,
        message_id: int,
        data: Any = None,
        *,
        model: str = "",
        prompt: str = "",
        think: int = 0,
        deploy: dict[str, Any] | None = None,
        source: str = "unknown",
    ) -> dict[str, Any]:
        self._pending += 1
        try:
            async with self._lock:
                info("消息处理", source, message_id, model)
                result = await self._dispatch(message_id, data, model=model, prompt=prompt,
                                              think=think, deploy=deploy or {})
                info("消息处理完成", source, message_id,
                     "status=", result.get("status", "unknown"),
                     "response_chars=", len(str(result.get("response", ""))))
                return result
        finally:
            self._pending -= 1

    async def _dispatch(self, message_id: int, data: Any, *, model: str, prompt: str,
                        think: int, deploy: dict[str, Any]) -> dict[str, Any]:
        try:
            if message_id == 1001:
                value = self.manager.status()
                value["queue_length"] = self.pending
                return self._response(model, message_id, "ok", value)
            if message_id == 1002:
                return self._response(model, message_id, "ok", detect_hardware().to_dict())
            if message_id == 1003:
                return self._response(model, message_id, "ok", {"models": list(self.manager.specs)})
            if message_id in {1004, 1005}:
                target = model or self._model_from(data)
                ok = self.manager.sleep(target) if message_id == 1004 else self.manager.unload(target)
                return self._response(target, message_id, "ok" if ok else "error", "操作成功" if ok else "操作失败")
            if message_id == 1006:
                target = model or self._model_from(data)
                await asyncio.to_thread(self.manager.ensure_loaded, target)
                return self._response(target, message_id, "ok", "操作成功")
            if message_id == 1007:
                # 持久化生成参数；请求里的 deploy 只影响当次，改常驻值走这里。
                payload = data if isinstance(data, dict) else {}
                target = model or str(payload.get("model", ""))
                changes = {key: value for key, value in
                           dict(payload.get("generation", payload.get("deploy", {}))).items()
                           if key in GENERATION_KEYS}
                merged = await asyncio.to_thread(self.manager.update_generation, target, changes)
                return self._response(target, message_id, "ok", merged)

            payload = data if isinstance(data, dict) else {}
            model = model or str(payload.get("model", ""))
            prompt = prompt or str(payload.get("prompt", data if isinstance(data, str) else ""))
            if isinstance(data, (list, tuple)):
                if not model and data:
                    model = str(data[0])
                if not prompt and len(data) > 1:
                    prompt = str(data[1])
            info("请求参数解析", "id=", message_id, "model=", model,
                 "prompt_chars=", len(prompt), "data_type=", type(data).__name__)
            deploy = {**dict(payload.get("deploy", {})), **deploy}
            if deploy:
                # reconfigure 内部会与当前 load 值 diff，值没变就不会重载模型。
                changes = {key: value for key, value in deploy.items() if key in LOAD_KEYS}
                if changes:
                    await asyncio.to_thread(self.manager.reconfigure, model, changes)
            # defaults.generation ← 模型 generation ← 本次 deploy（不落盘）。
            params = self.manager.generation_params(model)
            params.update({key: value for key, value in deploy.items() if key in GENERATION_KEYS})
            params = {key: value for key, value in params.items() if key in GENERATION_KEYS}
            if "max_tokens" in params:
                params["max_new_tokens"] = params.pop("max_tokens")
            if think:
                prompt = f"请以推理等级 {think}/5 分析后给出最终答案。\n\n{prompt}"
            result = await asyncio.to_thread(self.manager.generate, model, prompt, **params)
            return self._response(model, 0, "ok", result.text) | {"usage": {
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.tokens_generated,
                "time_seconds": result.time_seconds, "tokens_per_second": result.tokens_per_second,
            }}
        except (KeyError, RuntimeError, MemoryError, ValueError, FileNotFoundError) as exc:
            log("消息处理失败:", exc)
            return self._response(model, message_id, "error", str(exc))
        except Exception as exc:
            # 推理后端可能抛出自定义异常；统一转成可见的错误响应，避免
            # Console/FastAPI 只看到“消息处理开始”却没有结束结果。
            log("消息处理异常:", type(exc).__name__, exc)
            return self._response(model, message_id, "error", f"推理失败: {exc}")

    @staticmethod
    def _model_from(data: Any) -> str:
        if isinstance(data, dict):
            return str(data.get("model", ""))
        if isinstance(data, (list, tuple)) and data:
            return str(data[0])
        return str(data or "")


msgHandler = MsgHandler

__all__ = ["MsgHandler", "msgHandler"]
