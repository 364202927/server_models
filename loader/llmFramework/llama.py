from __future__ import annotations

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

    @staticmethod
    def _supports_flag(llama_bin: str, flag: str) -> bool:
        """旧版 llama-server 不认识的参数会直接启动失败，先从 --help 里确认再传。"""
        try:
            out = subprocess.run([llama_bin, "--help"], capture_output=True, text=True, errors="replace", timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return flag in (out.stdout + out.stderr)

    def _create(self, model_path: str, load_cfg: dict[str, Any]) -> "llama":
        self._unload_engine()
        source = self._resolve_gguf_file(model_path)
        self._model_path = str(source)
        self._port = self._find_free_port()

        calculated_n_ctx = max(int(load_cfg.get("useVram", 0)), self._FALLBACK_N_CTX)
        gpu_layers = int(load_cfg.get("gpu_offload_layers", -1))
        batch_size = int(load_cfg.get("batch_size", 512))
        slots = max(1, int(load_cfg.get("parallel_slots", 1)))
        slot_ctx = calculated_n_ctx if slots == 1 else calculated_n_ctx // slots // 256 * 256  # 单槽配额
        llama_bin = shutil.which("llama-server")
        if llama_bin:
            cmd = [
                llama_bin,
                "-m",
                str(source),
                "-c",
                str(calculated_n_ctx),
                "-ngl",
                str(gpu_layers),
                "-b",
                str(batch_size),
                "--host",
                self._host,
                "--port",
                str(self._port),
                "--alias",
                "default",
            ]
            if slots > 1:  # 多槽位：给每槽封顶 pool/np，任一槽占满都挤不掉其他槽的 KV
                cmd += ["-np", str(slots)]
                if self._supports_flag(llama_bin, "--kv-unified-per-slot"):
                    cmd += ["--kv-unified", "--kv-unified-per-slot", str(slot_ctx)]
                else:  # 旧版没有单槽封顶，退回非统一 KV，按 np 硬等分
                    cmd.append("--no-kv-unified")
                if self._supports_flag(llama_bin, "--no-cache-idle-slots"):
                    cmd.append("--no-cache-idle-slots")  # 空闲槽 KV 留驻显存，配额已保证不会被清
        else:
            raise FileNotFoundError("未找到 llama-server，请先在当前环境编译安装并加入 PATH")

        info(f"[llama-server] 正在拉起服务: {' '.join(cmd)}")
        self._spawn(cmd)

        health_url = f"{self.base_url}/health"
        if not self._wait_for_server_ready(health_url):
            err_msg = self._startup_error()
            self._unload_engine()
            raise RuntimeError(f"llama-server 启动失败: {err_msg}")

        self._http_client = httpx.Client(base_url=self.base_url, timeout=120.0)
        try:  # 新版 llama-server 未传 -np 时槽位数由引擎自定，打印实际值以核对 id_slot / 缓存隔离假设
            total_slots = self._http_client.get("/props").json().get("total_slots")
        except httpx.HTTPError:
            total_slots = None
        info("[llama-server] 实际槽位数", total_slots, f"(配置 parallel_slots={slots})")
        self._effective_load = {
            "engine": "llama",
            "context_length": calculated_n_ctx,
            "gpu_offload_layers": gpu_layers,
            "batch_size": batch_size,
            "parallel_slots": slots,
            "slot_context_length": slot_ctx,
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
            payload["id_slot"] = int(cfg["id_slot"])  # 固定槽位：主会话 KV 不被支路请求挤掉
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
        reasoning = str(message.get("reasoning_content") or "")
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
            reasoning=reasoning,
        )

    def remaining_kvCache(self, used_tokens: int, slot: int | None = None) -> int:
        """只算主会话（slot 0）的剩余：按单槽配额扣，而不是整个 KV 池；其他槽位不适用，返回 0。"""
        if self._effective_load.get("parallel_slots", 1) > 1 and slot != 0:
            return 0
        return max(0, self._effective_load.get("slot_context_length", 0) - used_tokens)
