from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import httpx

from ...utils.common import info, warn
from .baseInference import RawOutput, baseInference


class ninfer(baseInference):
    """NInfer (ninfer-serve) 转发引擎。

    约束（来自官方 docs/serving.md）：单 Engine、单常驻序列、请求串行，无连续批处理；
    因此不存在多槽位 / id_slot 隔离，parallel_slots 恒为 1。
    """

    _FALLBACK_N_CTX = 8192
    _MAX_N_CTX = 262144  # 注册模型原生上限
    _MAX_MTP_DRAFT = 5  # --mtp-draft-tokens 取值 0..5
    _MODEL_ID = "default"  # 请求里的 model 必须与 --model-id 一致
    _TOP_K_MAX = 20  # 沿用原实现的钳制值（文档未给出上限，以 ninfer-serve --help 为准）
    _SUPPORTED_KV_DTYPES = {"bf16", "int8"}

    @property
    def supported_modalities(self) -> set[str]:
        return {"text", "image", "video"}  # 两个注册模型均原生支持图像/视频，无需启动参数

    def _create(self, model_path: str, load_cfg: dict[str, Any]) -> "ninfer":
        self._unload_engine()

        # 1. 解析 .ninfer 权重文件（目录则取第一个）
        source = Path(model_path)
        if source.is_dir():
            files = sorted(source.glob("*.ninfer"))
            if not files:
                raise FileNotFoundError(f"目录中没有 .ninfer 文件: {model_path}")
            source = files[0]
        if not source.is_file() or source.suffix.lower() != ".ninfer":
            raise ValueError(f"不是有效的 .ninfer 文件: {model_path}")

        # 官方无安装目标，二进制在 build/apps/ninfer-serve：环境变量优先，其次 PATH
        ninfer_bin = os.environ.get("NINFER_BIN") or shutil.which("ninfer-serve") or ""
        if not Path(ninfer_bin).is_file():
            raise FileNotFoundError("未找到 ninfer-serve，请设置 NINFER_BIN 指向 build/apps/ninfer-serve 或加入 PATH")

        kv_dtype = str(load_cfg.get("kv_dtype") or "bf16").lower()
        if kv_dtype not in self._SUPPORTED_KV_DTYPES:
            raise ValueError(f"ninfer 的 kv_dtype 仅支持 {sorted(self._SUPPORTED_KV_DTYPES)}，当前: {kv_dtype}")

        # 2. 公共参数 -> 启动参数（useVram 此处已是 token 数，与 llama.py 口径一致）
        n_ctx = min(max(int(load_cfg.get("useVram", 0)), self._FALLBACK_N_CTX), self._MAX_N_CTX)
        mtp_n = min(int(load_cfg.get("draft_tokens", 3)), self._MAX_MTP_DRAFT) if load_cfg.get("mtp") else 0
        if int(load_cfg.get("parallel_slots", 1)) > 1:
            warn("[ninfer-serve] 单 Engine 单序列，parallel_slots 已按 1 处理")

        self._model_path = str(source)
        self._port = self._find_free_port()
        cmd = [
            ninfer_bin, self._model_path,
            "--host", self._host,
            "--port", str(self._port),
            "--model-id", self._MODEL_ID,
            "--max-context", str(n_ctx),
        ]
        if kv_dtype == "int8":  # 默认 bf16，不传
            cmd += ["--kv-dtype", "int8"]
        if mtp_n > 0:  # MTP 默认关闭(0)；--lm-head-draft 只服务于 MTP 提议头
            cmd += ["--mtp-draft-tokens", str(mtp_n)]
            if load_cfg.get("lm_head_draft", True):
                cmd.append("--lm-head-draft")

        info(f"[ninfer-serve] 正在拉起服务: {' '.join(cmd)}")
        self._spawn(cmd)

        if not self._wait_for_server_ready(f"{self.base_url}/health"):
            err_msg = self._startup_error()
            self._unload_engine()
            raise RuntimeError(f"ninfer-serve 启动失败: {err_msg}")

        # 非流式长 prefill（27B 262k 约 160s）会超过 120s，读超时放宽
        self._http_client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(600.0, connect=10.0))
        self._effective_load = {
            "engine": "ninfer",
            "context_length": n_ctx,
            "kv_dtype": kv_dtype,
            "parallel_slots": 1,
            "slot_context_length": n_ctx,
            "mtp_draft_tokens": mtp_n,
        }
        self._sleep_capable = True
        return self

    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> RawOutput:
        payload: dict[str, Any] = {
            "model": self._MODEL_ID,
            "messages": messages,
            "stream": False,
            "max_tokens": int(gen_cfg.get("max_tokens", 2048)),
        }

        # 仅透传文档列出的采样字段；min_p / repetition_penalty 不在 HTTP 字段表内，不转发
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if gen_cfg.get(key) is not None:
                payload[key] = gen_cfg[key]
        if gen_cfg.get("top_k") is not None:
            payload["top_k"] = max(0, min(int(gen_cfg["top_k"]), self._TOP_K_MAX))

        stop = [s for s in (gen_cfg.get("stop") or []) if s][:4]
        if stop:
            payload["stop"] = stop
        for key in ("tools", "tool_choice"):
            if gen_cfg.get(key):
                payload[key] = gen_cfg[key]

        resp = self._http_client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            cached = (data.get("timings") or {}).get("cache_n", -1)

        return RawOutput(
            text=str(message.get("content") or ""),
            tokens=int(usage.get("completion_tokens", 0)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            calls=message.get("tool_calls") or [],
            finish_reason=str(choice.get("finish_reason") or "stop"),
            cached_tokens=int(cached),
            reasoning=str(message.get("reasoning_content") or ""),
        )