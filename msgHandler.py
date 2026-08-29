"""Console 与 FastAPI 共用的统一消息处理器。"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from .hardware import detect_hardware
from .loader.models_mgr import ModelsMgr
from .utils.common import info, log


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
                return await self._dispatch(message_id, data, model=model, prompt=prompt,
                                            think=think, deploy=deploy or {})
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

            payload = data if isinstance(data, dict) else {}
            model = model or str(payload.get("model", ""))
            prompt = prompt or str(payload.get("prompt", data if isinstance(data, str) else ""))
            if isinstance(data, (list, tuple)):
                if not model and data:
                    model = str(data[0])
                if not prompt and len(data) > 1:
                    prompt = str(data[1])
            deploy = {**dict(payload.get("deploy", {})), **deploy}
            if deploy:
                load_keys = {"dtype", "context_length", "gpu_offload_layers", "batch_size",
                             "flash_attention", "draft_model", "speculative_decoding", "tensor_parallel",
                             "gpu_split", "trust_remote_code"}
                changes = {key: value for key, value in deploy.items() if key in load_keys}
                if changes:
                    await asyncio.to_thread(self.manager.reconfigure, model, changes)
            params = dict(self.manager.settings.get("generation", {}))
            params.update({key: value for key, value in deploy.items() if key in {
                "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "stop_sequences",
            }})
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
