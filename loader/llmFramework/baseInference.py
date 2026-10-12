from __future__ import annotations

import gc
import os
import signal
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import httpx
import psutil

from ...utils.common import info as log_info, error as log_error
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


@dataclass
class RawOutput:
    text: str
    tokens: int
    prompt_tokens: int
    calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    cached_tokens: int = -1


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

    def __init__(self) -> None:
        self._server_process: subprocess.Popen | None = None
        self._port: int | None = None
        self._host: str = "127.0.0.1"
        self._http_client: httpx.Client | None = None
        self._effective_load: dict[str, Any] = {}
        self._sleeping = False
        self._sleep_capable = False
        self._raw_memo: OrderedDict[str, dict[str, str]] = OrderedDict()
        self.last_diag: dict[str, Any] = {}

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

    def _wait_for_server_ready(self, health_url: str, timeout: float = 120.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._server_process and self._server_process.poll() is not None:
                log_error(f"Server 进程意外退出，退出码: {self._server_process.returncode}")
                return False
            try:
                resp = httpx.get(health_url, timeout=2.0)
                if resp.status_code in {200, 404}:  # 404 表明 http server 已成功接管
                    return True
            except httpx.RequestError:
                pass
            time.sleep(1.0)
        return False

    @abstractmethod
    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "baseInference":
        pass

    def _unload_engine(self) -> None:
        if self._http_client:
            self._http_client.close()
            self._http_client = None

        if self._server_process:
            try:
                # 终止子进程组
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
        # 默认使用 SIGSTOP 挂起子进程
        if self._server_process:
            os.killpg(os.getpgid(self._server_process.pid), signal.SIGSTOP)

    def _engine_wake(self) -> None:
        # 默认使用 SIGCONT 恢复子进程
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
        usage.process_rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)
        usage.system_available_mb = round(psutil.virtual_memory().available / (1024 ** 2), 1)
        return usage

    def generate(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded / Server not running. Call load() first.")

        cfg = dict(gen_cfg or {})
        system_instruction = str(cfg.pop("system_prompt", "") or "")

        cleaned_messages = chatDataFilter.preprocess_messages(
            messages=messages,
            supported_modalities=self.supported_modalities,
            system_instruction=system_instruction,
        )

        cleaned_messages, restored, missed = chatDataFilter.restore_raw_assistant(cleaned_messages, self._raw_memo)

        start = time.perf_counter()
        raw = self._response(messages=cleaned_messages, gen_cfg=cfg)
        elapsed = time.perf_counter() - start

        final_text, final_calls, final_reason = chatDataFilter.postprocess_result(
            raw.text, raw.calls, raw.finish_reason
        )

        key = chatDataFilter.memo_key(
            cleaned_messages[-1] if cleaned_messages else None,
            {"role": "assistant", "content": final_text, "tool_calls": final_calls},
        )
        self._raw_memo[key] = chatDataFilter.split_raw(raw.text)
        self._raw_memo.move_to_end(key)
        while len(self._raw_memo) > self._RAW_MEMO_MAX:
            self._raw_memo.popitem(last=False)

        return GenerationResult(
            text=final_text,
            tokens_generated=raw.tokens,
            time_seconds=elapsed,
            tokens_per_second=raw.tokens / elapsed if elapsed > 0 else 0.0,
            prompt_tokens=raw.prompt_tokens,
            tool_calls=final_calls,
            finish_reason=final_reason,
            cached_tokens=raw.cached_tokens,
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

    def remaining_kvCache(self, used_tokens: int) -> int:
        return max(0, self.context_limit() - used_tokens)