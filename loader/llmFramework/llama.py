from __future__ import annotations

import os
import shutil
import subprocess
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

        # 定位 llama-server 可执行文件，未找到则回退到 python -m llama_cpp.server
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
        else:
            cmd = [
                "python3", "-m", "llama_cpp.server",
                "--model", str(source),
                "--n_ctx", str(calculated_n_ctx),
                "--n_gpu_layers", str(gpu_layers),
                "--host", self._host,
                "--port", str(self._port),
            ]

        info(f"[llama-server] 正在拉起服务: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        health_url = f"{self.base_url}/health"
        if not self._wait_for_server_ready(health_url, timeout=90.0):
            err_msg = self._server_process.stderr.read().decode() if self._server_process.stderr else ""
            self._unload_engine()
            raise RuntimeError(f"llama-server 启动失败: {err_msg}")

        self._http_client = httpx.Client(base_url=self.base_url, timeout=120.0)
        self._effective_load = {
            "engine": "llama",
            "context_length": calculated_n_ctx,
            "gpu_offload_layers": gpu_layers,
            "batch_size": batch_size,
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
        }
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
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", -1)

        return RawOutput(
            text=str(message.get("content") or ""),
            tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            calls=calls,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            cached_tokens=cached_tokens,
        )