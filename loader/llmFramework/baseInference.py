from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import signal
import socket
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import httpx, psutil

from ...utils.common import cache_verdict, info as log_info, error as log_error
from ..chatDataFilter import chatDataFilter


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    prompt_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    cached_tokens: int = -1
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    memo_restored: int = 0
    memo_missed: int = 0
    memo_pruned: int = 0
    kv_remaining: int = 0


@dataclass
class RawOutput:
    text: str
    tokens: int
    prompt_tokens: int
    calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    cached_tokens: int = -1
    reasoning: str = ""


@dataclass
class MemoryUsage:
    gpu_allocated_mb: float = 0
    gpu_reserved_mb: float = 0
    gpu_total_mb: float = 0
    gpu_free_mb: float = 0
    process_rss_mb: float = 0
    system_available_mb: float = 0


class baseInference(ABC):
    """推理框架抽象基类：Server 模式基类"""

    _RAW_MEMO_MAX = 256
    _STARTUP_TIMEOUT = 180.0
    _CACHE_LOG_KEYS = ("selected slot", "purging slot", "forcing full", "checkpoint", "prompt cache", "similarity", "cache_ram", "n_past")

    def __init__(self) -> None:
        self._server_process: subprocess.Popen | None = None
        self._port: int | None = None
        self._host: str = "127.0.0.1"
        self._http_client: httpx.Client | None = None
        self._effective_load: dict[str, Any] = {}
        self._sleeping = False
        self._sleep_capable = False
        self._raw_memo: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._stderr_tail: deque[str] = deque(maxlen=60)
        self._slot_last: dict[int, dict[str, Any]] = {}

    @property
    def base_url(self) -> str:
        if not self._port:
            raise RuntimeError("服务未分配端口")
        return f"http://{self._host}:{self._port}"

    @property
    def is_loaded(self) -> bool:
        return self._server_process is not None and self._server_process.poll() is None

    @property
    def effective_load(self) -> dict[str, Any]:
        return dict(self._effective_load)

    @property
    def supported_modalities(self) -> set[str]:
        return {"text"}

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _spawn(self, cmd: list[str]) -> None:
        self._stderr_tail.clear()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            preexec_fn=os.setsid,
        )
        self._server_process = proc

        def pump() -> None:
            for line in proc.stderr:
                self._stderr_tail.append(line)
                if any(key in line for key in self._CACHE_LOG_KEYS):
                    log_info("[llama-server]", line.strip()[:240])

        threading.Thread(target=pump, daemon=True).start()

    def _startup_error(self) -> str:
        return "".join(self._stderr_tail)

    def _wait_for_server_ready(self, health_url: str, timeout: float | None = None) -> bool:
        timeout = timeout or self._STARTUP_TIMEOUT
        start_time = time.time()
        proc = self._server_process
        while time.time() - start_time < timeout:
            if proc is None or proc.poll() is not None:
                log_error(f"Server 进程已退出，退出码: {proc.returncode if proc else '已卸载'}")
                return False
            try:
                resp = httpx.get(health_url, timeout=2.0)
                if resp.status_code in {200, 404}:
                    return True
            except httpx.RequestError:
                pass
            time.sleep(1.0)
        return False

    @abstractmethod
    def _create(self, model_path: str, load_cfg: dict[str, Any]) -> "baseInference":
        pass

    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "baseInference":
        self._create(model_path, load_cfg)
        log_info("加载完成:", model_path)
        return self

    def _unload_engine(self) -> None:
        if self._http_client:
            self._http_client.close()
            self._http_client = None

        if self._server_process:
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._server_process = None

        self._port = None
        self._mark_unloaded()

    def unload(self) -> None:
        self._unload_engine()

    def sleep_to_ram(self) -> bool:
        if not self.is_loaded or self._sleeping or not self._sleep_capable:
            return self._sleeping
        start = time.perf_counter()
        try:
            self._engine_sleep()
        except Exception as exc:
            log_info("引擎休眠失败", type(self).__name__, exc)
            return False
        self._sleeping = True
        log_info("引擎已休眠", type(self).__name__, f"{round(time.perf_counter() - start, 2)}s")
        return True

    def wake(self) -> None:
        if not self._sleeping:
            return
        start = time.perf_counter()
        try:
            self._engine_wake()
        except Exception as exc:
            log_info("引擎唤醒失败", type(self).__name__, exc)
            raise
        self._sleeping = False
        log_info("引擎已唤醒", type(self).__name__, f"{round(time.perf_counter() - start, 2)}s")

    def _engine_sleep(self) -> None:
        if self._server_process:
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGSTOP)

    def _engine_wake(self) -> None:
        if self._server_process:
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGCONT)

    def _mark_unloaded(self) -> None:
        self._effective_load = {}
        self._sleeping = False
        self._sleep_capable = False
        self._raw_memo.clear()

    def release_cache(self) -> MemoryUsage:
        gc.collect()
        return self.memory_usage()

    def memory_usage(self) -> MemoryUsage:
        usage = MemoryUsage()
        usage.process_rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 1)
        usage.system_available_mb = round(psutil.virtual_memory().available / (1024**2), 1)
        return usage

    def generate(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any] | None = None,
        prune_on_miss: bool = False,
        cancel: threading.Event | None = None,
    ) -> GenerationResult:
        """支持 cancel 事件与回退清理：当请求在中途被客户端 ESC 取消时，绝不污染 _raw_memo 与 _slot_last。"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded / Server not running. Call load() first.")
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Generation cancelled before execution.")

        cfg = dict(gen_cfg or {})
        system_instruction = str(cfg.pop("system_prompt", "") or "")

        cleaned_messages = chatDataFilter.preprocess_messages(
            messages=messages,
            supported_modalities=self.supported_modalities,
            system_instruction=system_instruction,
        )

        live_keys = chatDataFilter.assistant_keys(cleaned_messages)
        cleaned_messages, restored, missed = chatDataFilter.restore_raw_assistant(cleaned_messages, self._raw_memo)

        self._log_slot_diff(cleaned_messages, cfg)

        start = time.perf_counter()
        # 如果子类 _response 支持 cancel 参数则传入，否则保持兼容调用
        if cancel is not None and "cancel" in inspect.signature(self._response).parameters:
            raw = self._response(messages=cleaned_messages, gen_cfg=cfg, cancel=cancel)
        else:
            raw = self._response(messages=cleaned_messages, gen_cfg=cfg)
        elapsed = time.perf_counter() - start

        # 如果在生成期间用户按了 ESC 取消了请求，直接抛错退出，防止把废弃分支的回复写入 _raw_memo
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Generation cancelled during execution.")

        final_text, final_calls, final_reason = chatDataFilter.postprocess_result(raw.text, raw.calls, raw.finish_reason)
        final_reasoning = (raw.reasoning or chatDataFilter.split_raw(raw.text)["reasoning"]).strip()

        key = chatDataFilter.memo_key(
            cleaned_messages[-1] if cleaned_messages else None,
            {"role": "assistant", "content": final_text, "tool_calls": final_calls},
        )
        self._raw_memo[key] = {"reasoning": final_reasoning, "content": final_text}
        self._raw_memo.move_to_end(key)
        while len(self._raw_memo) > self._RAW_MEMO_MAX:
            self._raw_memo.popitem(last=False)

        pruned = 0
        # 当发生未命中或客户端回退重发（missed > 0）时，清理已不在当前分支里的废弃思考指纹
        if prune_on_miss and (missed > 0 or cache_verdict(raw.cached_tokens, raw.prompt_tokens)[0] in {"未命中", "部分命中"}):
            stale = [k for k in list(self._raw_memo) if k not in live_keys and k != key]
            for k in stale:
                self._raw_memo.pop(k, None)
            pruned = len(stale)

        return GenerationResult(
            text=final_text,
            tokens_generated=raw.tokens,
            time_seconds=elapsed,
            tokens_per_second=raw.tokens / elapsed if elapsed > 0 else 0.0,
            prompt_tokens=raw.prompt_tokens,
            tool_calls=final_calls,
            finish_reason=final_reason,
            reasoning=final_reasoning,
            cached_tokens=raw.cached_tokens,
            memo_restored=restored,
            memo_missed=missed,
            memo_pruned=pruned,
            kv_remaining=self.remaining_kvCache(raw.prompt_tokens + raw.tokens, cfg.get("id_slot")),
        )

    def _log_slot_diff(self, messages: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
        slot = cfg.get("id_slot")
        slot = -1 if slot is None else int(slot)
        digest = lambda obj: hashlib.md5(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:6]
        now = {"tools": digest(cfg.get("tools") or []), "msgs": [digest(m) for m in messages]}
        last = self._slot_last.get(slot)
        self._slot_last[slot] = now
        if last is None:
            log_info("[槽位差异]", f"槽={slot}", "该槽首次请求", f"消息={len(messages)}", f"tools#{now['tools']}")
            return

        common = next((i for i, (a, b) in enumerate(zip(last["msgs"], now["msgs"])) if a != b), min(len(last["msgs"]), len(now["msgs"])))
        where = f"首个分歧=第{common}条({messages[common].get('role')})" if common < len(messages) else "本次是上次的前缀/等长"
        log_info(
            "[槽位差异]",
            f"槽={slot}",
            f"上次{len(last['msgs'])}条 本次{len(messages)}条 共同前缀{common}条",
            where,
            f"tools#{last['tools']}->#{now['tools']}" + ("(变了，整段前缀失效)" if last["tools"] != now["tools"] else "(不变)"),
        )

    @abstractmethod
    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> RawOutput:
        pass

    def context_limit(self) -> int:
        return int(self._effective_load.get("context_length", 0)) if self.is_loaded else 0

    def remaining_kvCache(self, used_tokens: int, slot: int | None = None) -> int:
        slots = max(1, int(self._effective_load.get("parallel_slots", 1)))
        return max(0, (self.context_limit() // slots) - used_tokens)