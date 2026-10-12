from __future__ import annotations

import os
import subprocess
from typing import Any

import httpx

from ...utils.common import info
from .baseInference import RawOutput, baseInference


class sglang(baseInference):
    """SGLang Server 推理转发引擎"""

    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "sglang":
        self._unload_engine()
        self._port = self._find_free_port()

        tp_size = load_cfg.get("tensor_parallel", 1)
        dtype = load_cfg.get("dtype", "bfloat16")
        mem_fraction = load_cfg.get("gpu_memory_utilization", 0.9)

        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", self._host,
            "--port", str(self._port),
            "--tp-size", str(tp_size),
            "--dtype", str(dtype),
            "--mem-fraction-static", str(mem_fraction),
        ]

        context_val = load_cfg.get("useVram", 0)
        if context_val and context_val > 0:
            cmd.extend(["--context-length", str(int(context_val))])

        info(f"[SGLang-server] 正在拉起服务: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        health_url = f"{self.base_url}/health"
        if not self._wait_for_server_ready(health_url, timeout=120.0):
            err_msg = self._server_process.stderr.read().decode() if self._server_process.stderr else ""
            self._unload_engine()
            raise RuntimeError(f"SGLang server 启动失败: {err_msg}")

        self._http_client = httpx.Client(base_url=self.base_url, timeout=120.0)
        self._effective_load = {
            "engine": "sglang",
            "context_length": int(context_val) if context_val else 4096,
            "tp_size": tp_size,
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