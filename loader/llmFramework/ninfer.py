from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import httpx

from ...utils.common import info
from .baseInference import RawOutput, baseInference


class ninfer(baseInference):
    """NInfer (ninfer-serve) 转发引擎"""

    _FALLBACK_N_CTX = 8192
    _DTYPE_MAP = {"bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16", "fp8": "fp8"}

    def __init__(self) -> None:
        super().__init__()
        self._model_path: str = ""
        self._model_id: str = "default"
        self._vision_enabled: bool = False

    @property
    def supported_modalities(self) -> set[str]:
        return {"text", "image", "video"} if self._vision_enabled else {"text"}

    def _create(self, model_path: str, load_cfg: dict[str, Any]) -> "ninfer":
        self._unload_engine()

        # 1. 解析 .ninfer 权重文件路径（内联单次调用）
        source = Path(model_path)
        if source.is_dir():
            files = sorted(source.glob("*.ninfer"))
            if not files:
                raise FileNotFoundError(f"目录中没有 .ninfer 文件: {model_path}")
            source = files[0]
        if not source.is_file() or source.suffix.lower() != ".ninfer":
            raise ValueError(f"不是有效的 .ninfer 文件: {model_path}")

        ninfer_bin = shutil.which("ninfer-serve") or os.environ.get("NINFER_BIN", "")
        if not ninfer_bin or not Path(ninfer_bin).is_file():
            raise FileNotFoundError("未找到 ninfer-serve，请将其加入 PATH 或设置 NINFER_BIN 环境变量")

        self._model_path = str(source)
        self._port = self._find_free_port()

        # 2. 转换 template 公共参数 -> ninfer 启动参数
        # useVram -> --max-context & --kv-capacity
        calculated_n_ctx = max(int(load_cfg.get("useVram", 0)), self._FALLBACK_N_CTX)
        slots = max(1, int(load_cfg.get("parallel_slots", 1)))
        default_pool = calculated_n_ctx if slots == 1 else calculated_n_ctx * slots
        kv_capacity = max(int(load_cfg.get("kv_capacity", 0)), default_pool)

        # dtype ("bfloat16") / kv_dtype ("fp8") -> --kv-dtype ("bf16" | "fp16" | "fp8")
        raw_dtype = str(load_cfg.get("kv_dtype") or load_cfg.get("dtype", "fp8")).lower()
        kv_dtype = self._DTYPE_MAP.get(raw_dtype, "fp8")

        host_slots = int(load_cfg.get("host_state_slots", 8))
        host_kv_mib = int(load_cfg.get("host_kv_mib", 8192))

        cmd = [
            ninfer_bin,
            self._model_path,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--model-id",
            self._model_id,
            "--max-context",
            str(calculated_n_ctx),
            "--kv-capacity",
            str(kv_capacity),
            "--max-concurrency",
            str(slots),
            "--kv-dtype",
            kv_dtype,
            "--device-state-slots",
            str(slots),
            "--host-state-slots",
            str(host_slots),
            "--host-kv-mib",
            str(host_kv_mib),
        ]

        if load_cfg.get("preserve_thinking", True):
            cmd.append("--preserve-thinking")

        self._vision_enabled = bool(load_cfg.get("vision", False))
        if self._vision_enabled:
            cmd.append("--vision")

        # 3. 转换 template 的 mtp (bool) 与 draft (list) -> --spec & --draft-tokens
        draft_cfg = load_cfg.get("draft") or []
        spec_type = str(draft_cfg[0]).lower() if draft_cfg else ("mtp" if load_cfg.get("mtp", False) else "")
        if spec_type in {"mtp", "dflash", "dflash2"}:
            default_draft_n = int(draft_cfg[1]) if len(draft_cfg) > 1 else (3 if spec_type == "mtp" else 7)
            draft_tokens = int(load_cfg.get("draft_tokens", default_draft_n))
            cmd += ["--spec", spec_type, "--draft-tokens", str(draft_tokens)]
            if load_cfg.get("lm_head_draft", True):
                cmd.append("--lm-head-draft")

        info(f"[ninfer-serve] 正在拉起服务: {' '.join(cmd)}")
        self._spawn(cmd)

        if not self._wait_for_server_ready(f"{self.base_url}/health"):
            err_msg = self._startup_error()
            self._unload_engine()
            raise RuntimeError(f"ninfer-serve 启动失败: {err_msg}")

        self._http_client = httpx.Client(base_url=self.base_url, timeout=120.0)
        try:
            models_data = self._http_client.get("/v1/models").json().get("data", [])
            if models_data and models_data[0].get("id"):
                self._model_id = str(models_data[0]["id"])
        except httpx.HTTPError:
            pass

        info(
            "[ninfer-serve] 实际并发槽位数",
            slots,
            f"(max_ctx={calculated_n_ctx}, kv_capacity={kv_capacity}, kv_dtype={kv_dtype})",
        )
        self._effective_load = {
            "engine": "ninfer",
            "context_length": calculated_n_ctx,
            "kv_capacity": kv_capacity,
            "kv_dtype": kv_dtype,
            "parallel_slots": slots,
            "slot_context_length": calculated_n_ctx,
            "spec_type": spec_type or "none",
        }
        self._sleep_capable = True
        return self

    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> RawOutput:
        cfg = dict(gen_cfg)
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "stream": False,
            "max_tokens": int(cfg.get("max_tokens", 2048)),
        }

        # 1. 透传兼容的采样参数
        for key in ("temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty"):
            if cfg.get(key) is not None:
                payload[key] = cfg[key]

        # 2. 转换 template.top_k (默认50) -> 钳制到 NInfer CUDA 算子上限 [0, 20]
        if cfg.get("top_k") is not None:
            payload["top_k"] = max(0, min(int(cfg["top_k"]), 20))

        # 3. 转换 template.repetition_penalty (默认1.05) -> 折算为 presence_penalty
        rep_penalty = float(cfg.get("repetition_penalty", 1.0))
        if rep_penalty != 1.0 and not payload.get("presence_penalty"):
            payload["presence_penalty"] = round(max(-2.0, min(2.0, (rep_penalty - 1.0) * 2.0)), 3)

        # 4. 转换 template.stop (list) -> 过滤空串并截断至 NInfer 限制的最多 4 个
        stop = [s for s in (cfg.get("stop") or []) if s]
        if stop:
            payload["stop"] = stop[:4]

        if cfg.get("tools"):
            payload["tools"] = cfg["tools"]
        if cfg.get("tool_choice"):
            payload["tool_choice"] = cfg["tool_choice"]

        resp = self._http_client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        if cached_tokens is None:
            cached_tokens = data.get("timings", {}).get("cache_n", -1)

        return RawOutput(
            text=str(message.get("content") or ""),
            tokens=int(usage.get("completion_tokens", 0)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            calls=message.get("tool_calls") or [],
            finish_reason=str(choice.get("finish_reason") or "stop"),
            cached_tokens=int(cached_tokens),
            reasoning=str(message.get("reasoning_content") or ""),
        )

    def remaining_kvCache(self, used_tokens: int, slot: int | None = None) -> int:
        """动态共享 KV 池下，主会话（slot 0）最大可用为单请求上下文上限与总池容量的较小值。"""
        if self._effective_load.get("parallel_slots", 1) > 1 and slot != 0:
            return 0
        limit = min(
            int(self._effective_load.get("context_length", 0)),
            int(self._effective_load.get("kv_capacity", 0)),
        )
        return max(0, limit - used_tokens)