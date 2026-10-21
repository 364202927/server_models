from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from ...utils.common import info, error, warn
from ..model_spec import normalize_model_path
from .baseInference import RawOutput, baseInference


@lru_cache(maxsize=4)
def _help_text(llama_bin: str) -> str:
    """llama-server --help 每个二进制只取一次：一次加载要探测多个可选参数。"""
    try:
        out = subprocess.run([llama_bin, "--help"], capture_output=True, text=True, errors="replace", timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout + out.stderr


class llama(baseInference):
    """llama.cpp Server 转发引擎"""

    _FALLBACK_N_CTX = 8192
    _DEFAULT_DRAFT_N = 3
    _MAX_MTP_DRAFT = 6  # MTP 的 --spec-draft-n-max 实测甜点上限，再高收益递减
    _MAX_DRAFT_N = 64  # 外部草稿模型可承受更长的草稿链

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
        return flag in _help_text(llama_bin)

    @classmethod
    def _draft_path(cls, load_cfg: dict[str, Any]) -> Path | None:
        """draft 兼容字符串/列表，取第一个非空项；目录则取首个 .gguf。"""
        raw = load_cfg.get("draft") or []
        items = [raw] if isinstance(raw, str) else list(raw)
        first = next((str(item).strip() for item in items if str(item).strip()), "")
        return cls._resolve_gguf_file(normalize_model_path(first)) if first else None

    @staticmethod
    def _lora_flags(load_cfg: dict[str, Any]) -> list[str]:
        """LoRA 只接受本地 GGUF 适配器；lora 兼容字符串/列表，可叠加多个。"""
        raw = load_cfg.get("lora") or ""
        items = [raw] if isinstance(raw, str) else list(raw)
        flags: list[str] = []
        for item in items:
            value = str(item or "").strip()
            if not value:
                continue
            path = Path(normalize_model_path(value))
            if not path.is_file():
                raise FileNotFoundError(f"LoRA 文件不存在: {path}")
            flags += ["--lora", str(path)]
        return flags

    @classmethod
    def _draft_tokens(cls, load_cfg: dict[str, Any], cap: int) -> int:
        return min(max(1, int(load_cfg.get("draft_tokens", cls._DEFAULT_DRAFT_N))), cap)

    def _spec_flags(self, llama_bin: str, load_cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        """草稿加速：显式 draft 模型优先，其次主模型内置 MTP；返回启动参数与生效信息。"""
        result = {"draft_model": "", "draft_tokens": 0, "mtp_draft_tokens": 0}
        draft = self._draft_path(load_cfg)
        if draft:
            tokens = self._draft_tokens(load_cfg, self._MAX_DRAFT_N)
            if self._supports_flag(llama_bin, "draft-simple"):
                spec = ["--spec-type", "draft-simple", "--spec-draft-n-max", str(tokens)]
            elif self._supports_flag(llama_bin, "--draft-max"):  # 旧版没有 --spec-type
                spec = ["--draft-max", str(tokens)]
            else:
                warn("[llama-server] 当前版本不支持外部草稿模型，已忽略 draft 配置；请更新 llama.cpp")
                return [], result
            if load_cfg.get("mtp"):
                warn("[llama-server] draft 与 mtp 同时开启，仅使用外部草稿模型")
            result.update(draft_model=str(draft), draft_tokens=tokens)
            return ["-md", str(draft), *spec], result
        if load_cfg.get("mtp"):
            tokens = self._draft_tokens(load_cfg, self._MAX_MTP_DRAFT)
            if not self._supports_flag(llama_bin, "draft-mtp"):
                warn("[llama-server] 当前版本 --spec-type 不支持 draft-mtp，已忽略 mtp；请更新 llama.cpp")
                return [], result
            result["mtp_draft_tokens"] = tokens
            return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(tokens)], result
        return [], result

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
        spec_info = {"draft_model": "", "draft_tokens": 0, "mtp_draft_tokens": 0}
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
            cmd += self._lora_flags(load_cfg)
            spec_cmd, spec_info = self._spec_flags(llama_bin, load_cfg)
            cmd += spec_cmd
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
        if spec_info["mtp_draft_tokens"]:
            info("[llama-server] MTP 已启用", f"draft_tokens={spec_info['mtp_draft_tokens']}")
        if spec_info["draft_model"]:
            info("[llama-server] 外部草稿模型已启用", spec_info["draft_model"], f"draft_tokens={spec_info['draft_tokens']}")
        self._effective_load = {
            "engine": "llama",
            "context_length": calculated_n_ctx,
            "gpu_offload_layers": gpu_layers,
            "batch_size": batch_size,
            "parallel_slots": slots,
            "slot_context_length": slot_ctx,
            "lora": load_cfg.get("lora") or "",
            **spec_info,
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

    def kv_cache_info(self) -> dict[str, int]:
        """slot 0 的占用取 /slots 实时值与最近一次请求用量的较大者：空闲槽 KV 被引擎释放后 n_past 归零，
        直接用它会把“已用过”误报成“全空闲”（日志里显示的剩余才是请求结束时的真实值）。"""
        info = super().kv_cache_info()
        if not info["total"] or self._http_client is None:
            return info
        try:
            slots = self._http_client.get("/slots", timeout=2.0).json()
        except (httpx.HTTPError, ValueError):
            return info
        slot0 = next((s for s in slots if isinstance(s, dict) and int(s.get("id", -1)) == 0), None)
        if not slot0:
            return info
        used = max(int(slot0.get("n_past") or 0), self._kv_used.get(0, 0))
        return {"total": info["total"], "remaining": max(0, info["total"] - used)}
