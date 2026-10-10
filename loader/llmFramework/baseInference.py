from __future__ import annotations

import gc
import os
import time
from collections import OrderedDict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import psutil
import torch

from ...utils.common import info as log_info
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
    cached_tokens: int = -1  # 引擎上报的 prompt 前缀缓存命中数，-1 表示引擎未提供


@dataclass
class RawOutput:
    """引擎 _response 的原始输出（未经 postprocess）。"""
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
    """推理框架抽象基类：入参规范收敛。"""

    _RAW_MEMO_MAX = 256

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._effective_load: dict[str, Any] = {}
        self._sleeping = False
        self._sleep_capable = False
        # 引擎原始输出记忆库：下一轮请求把清洗后的历史还原成 KV 里的原文，避免前缀分叉
        self._raw_memo: OrderedDict[str, str] = OrderedDict()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def effective_load(self) -> dict[str, Any]:
        return dict(self._effective_load)

    @property
    def supported_modalities(self) -> set[str]:
        return {"text"}

    @abstractmethod
    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "baseInference":
        pass
    @abstractmethod
    def _unload_engine(self) -> None:
        pass

    def unload(self) -> None:
        self._unload_engine()

    def count_tokens(self, text_or_messages: str | list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        """默认用引擎真实 tokenizer 计数；无 tokenizer 的引擎（如 llama）需覆写。"""
        tokenizer = self._require_tokenizer()
        text = text_or_messages if isinstance(text_or_messages, str) else self._build_chat_prompt(text_or_messages, tools)
        return len(tokenizer.encode(text))

    def sleep_to_ram(self) -> bool:
        if self._model is None or self._sleeping or not self._sleep_capable:
            return self._sleeping
        start = time.perf_counter()
        try:
            self._engine_sleep()
        except Exception as exc:
            log_info("引擎休眠失败", type(self).__name__, exc)
            return False
        self._sleeping = True
        self.release_cache()
        log_info("引擎已休眠至 RAM", type(self).__name__, f"{round(time.perf_counter() - start, 2)}s")
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
        raise NotImplementedError

    def _engine_wake(self) -> None:
        raise NotImplementedError

    def _mark_unloaded(self) -> None:
        self._model = None
        self._tokenizer = None
        self._effective_load = {}
        self._sleeping = False
        self._sleep_capable = False
        self._raw_memo.clear()

    def release_cache(self) -> MemoryUsage:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return self.memory_usage()

    def memory_usage(self) -> MemoryUsage:
        usage = MemoryUsage()
        if torch.cuda.is_available():
            usage.gpu_allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
            usage.gpu_reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
            usage.gpu_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
            usage.gpu_free_mb = round(usage.gpu_total_mb - usage.gpu_reserved_mb, 1)

        usage.process_rss_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)
        usage.system_available_mb = round(psutil.virtual_memory().available / (1024 ** 2), 1)
        return usage

    def generate(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """核心模板方法：仅 2 个入参。"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        cfg = dict(gen_cfg or {})
        system_instruction = str(cfg.pop("system_prompt", "") or "")

        cleaned_messages = chatDataFilter.preprocess_messages(
            messages=messages,
            supported_modalities=self.supported_modalities,
            system_instruction=system_instruction,
        )

        cleaned_messages, restored, missed = chatDataFilter.restore_raw_assistant(cleaned_messages, self._raw_memo)
        log_info("[DBG] assistant 原文还原", f"命中 {restored} 条, 未命中 {missed} 条, 记忆库 {len(self._raw_memo)} 条")

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
        self._raw_memo[key] = raw.text
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
        """子类物理调用：仅 2 个入参。cached_tokens 须取引擎真实上报值。"""
        pass
    def context_limit(self) -> int:
        """上下文总窗口；各引擎需在 load() 末尾把实际生效值写入 _effective_load["context_length"]。"""
        return int(self._effective_load.get("context_length", 0)) if self.is_loaded else 0

    def remaining_kvCache(self, used_tokens: int) -> int:
        """剩余 KV Cache = 上下文窗口 - 本轮已用 (prompt + completion)。"""
        return max(0, self.context_limit() - used_tokens)

    def _require_tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("当前引擎未提供 tokenizer")
        return self._tokenizer

    def _build_chat_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        """用模型自带 chat template（含 tools）渲染 prompt；渲染失败直接抛出，不回退拼接。"""
        return self._require_tokenizer().apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True
        )
