from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import httpx

from ...utils.common import info, error
from .baseInference import RawOutput, baseInference


class llama(baseInference):
    """llama.cpp Server 转发引擎"""

    _FALLBACK_N_CTX = 8192

    def __init__(self) -> None:
        super().__init__()
        self._model_path: str = ""

    @staticmethod
    def _resolve_gguf_file(model_path: str) -> Path:
        source = Path(model_path)
        if source.is_dir():
            files = sorted(source.glob("*.gguf"))
            if not files:
                raise FileNotFoundError(f"目录中没有 .gguf 文件: {model_path}")
            source = files[0]
        if not source.is_file() or source.suffix.lower() != ".gguf":
            raise ValueError(f"不是有效的 GGUF 文件: {model_path}")
        return source

    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "llama":
        self._unload_engine()
        source = self._resolve_gguf_file(model_path)
        self._model_path = str(source)
        self._port = self._find_free_port()

        calculated_n_ctx = max(int(load_cfg.get("useVram", 0)), self._FALLBACK_N_CTX)
        gpu_layers = int(load_cfg.get("gpu_offload_layers", -1))
        batch_size = int(load_cfg.get("batch_size", 512))
        slots = max(1, int(load_cfg.get("parallel_slots", 1)))
        llama_bin = shutil.which("llama-server")
        if llama_bin:
            cmd = [
                llama_bin,
                "-m", str(source),
                "-c", str(calculated_n_ctx),
                "-ngl", str(gpu_layers),
                "-b", str(batch_size),
                "--host", self._host,
                "--port", str(self._port),
                "--alias", "default",
            ]
            if slots > 1:
                # 多槽位共享同一块 KV 池（统一 KV），否则 -c 会被均分给各槽位
                cmd += ["-np", str(slots), "--kv-unified"]
        else:
            raise FileNotFoundError("未找到 llama-server，请先在当前环境编译安装并加入 PATH")

        info(f"[llama-server] 正在拉起服务: {' '.join(cmd)}")
        self._spawn(cmd)

        health_url = f"{self.base_url}/health"
        if not self._wait_for_server_ready(health_url, timeout=90.0):
            err_msg = self._startup_error()
            self._unload_engine()
            raise RuntimeError(f"llama-server 启动失败: {err_msg}")

        self._http_client = httpx.Client(base_url=self.base_url, timeout=120.0)
        self._effective_load = {
            "engine": "llama",
            "context_length": calculated_n_ctx,
            "gpu_offload_layers": gpu_layers,
            "batch_size": batch_size,
            "parallel_slots": slots,
        }
        self._sleep_capable = True
        return self

    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> RawOutput:
        cfg = dict(gen_cfg)
        tools = cfg.pop("tools", None)
        tool_choice = cfg.pop("tool_choice", None)

        payload: dict[str, Any] = {
            "model": "default",
            "messages": messages,
            "stream": False,
            "max_tokens": cfg.pop("max_tokens", 512),
            "cache_prompt": True,
        }
        if cfg.get("id_slot") is not None:
            payload["id_slot"] = int(cfg["id_slot"])   # 固定槽位：主会话 KV 不被支路请求挤掉
        if "temperature" in cfg:
            payload["temperature"] = cfg["temperature"]
        if "top_p" in cfg:
            payload["top_p"] = cfg["top_p"]
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        resp = self._http_client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        calls = message.get("tool_calls") or []
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        # 新版 usage 带 cached_tokens；旧版只在 timings.cache_n 里给出复用的 token 数
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        if cached_tokens is None:
            cached_tokens = data.get("timings", {}).get("cache_n", -1)

        return RawOutput(
            text=str(message.get("content") or ""),
            tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            calls=calls,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            cached_tokens=cached_tokens,
        )