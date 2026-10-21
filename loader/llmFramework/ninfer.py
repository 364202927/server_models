from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import httpx

from ...utils.common import info, warn
from ..model_spec import normalize_model_path
from .baseInference import RawOutput, baseInference

# ninfer 官方无 install 目标：编译产物相对仓库根 / 构建目录的位置，按顺序探测
_BIN_NAME = "ninfer-serve"
_BIN_PATHS = tuple(f"{prefix}{_BIN_NAME}" for prefix in ("", "build/", "apps/", "build/apps/", "build/bin/"))


def _pick_bin(candidate: Path) -> str:
    """candidate 可以是二进制本身、build 目录或仓库根目录；命中返回路径，否则空串。"""
    for path in ([candidate] if candidate.is_file() else [candidate / rel for rel in _BIN_PATHS]):
        if not path.is_file():
            continue
        if not os.access(path, os.X_OK):
            raise PermissionError(f"{path} 存在但没有执行权限，请执行 chmod +x {path}")
        return str(path)
    return ""


def _search_roots() -> list[Path]:
    """可能存放 ninfer 检出的目录：当前目录、家目录、/opt、/usr/local 及其下的 llmEngine 布局（含 ninfer* 多份检出）。"""
    bases = [Path.cwd(), Path.home(), Path("/opt"), Path("/usr/local")]
    mount = Path("/mnt")
    if mount.is_dir():
        bases += sorted(mount.glob("*"))
    roots: list[Path] = []
    for base in bases:
        for parent in (base, base / "llmEngine"):
            roots.append(parent)
            roots += sorted(parent.glob("ninfer*"))
    return roots


def _stringify_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把历史 assistant.tool_calls 的 arguments 还原成 JSON 字符串。

    共用过滤器为了按 schema 修正参数类型，把它反序列化成了 dict；ninfer 严格校验
    "function tool_calls must contain string arguments"，否则整条请求 400。
    """
    fixed: list[dict[str, Any]] = []
    for msg in messages:
        calls = msg.get("tool_calls")
        if msg.get("role") != "assistant" or not isinstance(calls, list) or not calls:
            fixed.append(msg)
            continue
        item = {**msg, "tool_calls": []}
        for call in calls:
            fn = dict(call.get("function") or {})
            args = fn.get("arguments")
            if not isinstance(args, str):
                fn["arguments"] = json.dumps({} if args is None else args, ensure_ascii=False)
            item["tool_calls"].append({**call, "function": fn})
        fixed.append(item)
    return fixed


class ninfer(baseInference):
    """NInfer (ninfer-serve) 转发引擎。

    约束（来自官方 docs/serving.md）：单 Engine、单常驻序列、请求串行，无连续批处理；
    因此不存在多槽位 / id_slot 隔离，parallel_slots 恒为 1。
    """

    _FALLBACK_N_CTX = 8192
    _MAX_N_CTX = 262144  # 注册模型原生上限
    _MAX_MTP_DRAFT = 15  # --draft-tokens 取值 1..15
    _MODEL_ID = "default"  # 请求里的 model 必须与 --model-id 一致
    _TOP_K_MAX = 20  # 沿用原实现的钳制值（文档未给出上限，以 ninfer-serve --help 为准）
    _SUPPORTED_KV_DTYPES = {"bf16", "int8", "fp8", "rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8", "nvfp4", "k8v4"}

    @property
    def supported_modalities(self) -> set[str]:
        return {"text", "image", "video"}  # 两个注册模型均原生支持图像/视频，无需启动参数

    @staticmethod
    def _resolve_bin() -> str:
        """定位 ninfer-serve：NINFER_BIN -> NINFER_HOME/NINFER_ROOT -> PATH -> 常见检出位置。"""
        explicit = str(os.environ.get("NINFER_BIN") or "").strip()
        if explicit:
            value = normalize_model_path(explicit)
            hit = (shutil.which(value) if os.sep not in value else "") or _pick_bin(Path(value))
            if hit:
                return hit
            raise FileNotFoundError(f"NINFER_BIN={explicit} 未指向可执行的 {_BIN_NAME}（支持二进制路径 / build 目录 / 仓库根目录）")

        homes = [Path(normalize_model_path(raw)) for raw in (str(os.environ.get(key) or "").strip() for key in ("NINFER_HOME", "NINFER_ROOT")) if raw]
        if homes:
            hit = next((found for found in map(_pick_bin, homes) if found), "")
            if hit:
                return hit
            raise FileNotFoundError(f"NINFER_HOME/NINFER_ROOT 下没有可执行的 {_BIN_NAME}: {', '.join(map(str, homes))}")

        hit = shutil.which(_BIN_NAME)
        if hit:
            return hit

        candidates = _search_roots()
        hits = [found for found in map(_pick_bin, candidates) if found]
        if not hits:
            raise FileNotFoundError(f"未找到 {_BIN_NAME}：请设置 NINFER_BIN 指向 build/apps/{_BIN_NAME}，" f"或设置 NINFER_HOME 指向 ninfer 仓库根目录，或把 build/apps 加入 PATH；" f"已自动探测 {len(candidates)} 处（如 {', '.join(map(str, candidates[:4]))} 等）")
        # 多份检出（如 ninfer / ninfer-all）时取最近构建的那份：旧构建可能不认新 artifact 的布局
        best = max(hits, key=lambda found: Path(found).stat().st_mtime)
        if len(hits) > 1:
            warn(f"[ninfer-serve] 发现 {len(hits)} 份 {_BIN_NAME}，取最新构建: {best}（其余: {', '.join(h for h in hits if h != best)}）")
        info(f"[ninfer-serve] 自动定位到 {best}（可用 NINFER_BIN / NINFER_HOME 覆盖）")
        return best

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

        ninfer_bin = self._resolve_bin()

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
            ninfer_bin,
            self._model_path,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--model-id",
            self._MODEL_ID,
            "--max-context",
            str(n_ctx),
        ]
        if kv_dtype != "bf16":  # 默认 bf16，不传
            cmd += ["--kv-dtype", kv_dtype]
        if mtp_n > 0:  # MTP 默认关闭；--lm-head-draft 不再是独立开关：artifact 带 proposal 头时引擎默认就用
            cmd += ["--spec", "mtp", "--draft-tokens", str(mtp_n)]
            if not load_cfg.get("lm_head_draft", True):
                cmd.append("--full-draft-head")

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
            "messages": _stringify_tool_arguments(messages),
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
        if not resp.is_success:
            # ninfer 的报错体写明是哪个字段不合法，必须带出来，否则只剩一句 400 Bad Request
            raise RuntimeError(f"ninfer-serve 返回 {resp.status_code}: {resp.text[:800]}")
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
