from __future__ import annotations

import gc
import os
import time
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

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._effective_load: dict[str, Any] = {}
        self._sleeping = False
        self._sleep_capable = False

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

    @abstractmethod
    def count_tokens(self, text_or_messages: str | list[dict[str, Any]]) -> int:
        pass

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

        start = time.perf_counter()
        raw_text, gen_tokens, prompt_tokens, calls, finish_reason = self._response(
            messages=cleaned_messages,
            gen_cfg=cfg,
        )
        elapsed = time.perf_counter() - start

        final_text, final_calls, final_reason = chatDataFilter.postprocess_result(
            raw_text, calls, finish_reason
        )

        return GenerationResult(
            text=final_text,
            tokens_generated=gen_tokens,
            time_seconds=elapsed,
            tokens_per_second=gen_tokens / elapsed if elapsed > 0 else 0.0,
            prompt_tokens=prompt_tokens,
            tool_calls=final_calls,
            finish_reason=final_reason,
        )

    @abstractmethod
    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> tuple[str, int, int, list[dict[str, Any]], str]:
        """子类物理调用：仅 2 个入参。"""
        pass
    #当前模型上下文总窗口大小
    def context_limit(self) -> int:
        if not self.is_loaded:
            return 0
        # 优先读取已生效的 context_length 配置
        if "context_length" in self._effective_load:
            return int(self._effective_load["context_length"])
        # 回退尝试读取底层引擎属性 (例如 llama.cpp 的 n_ctx)
        if hasattr(self._model, "n_ctx") and callable(self._model.n_ctx):
            return int(self._model.n_ctx())
        return int(getattr(self._model, "n_ctx", 0) or 0)
    #估算下次生成的token数量
    def estimate_genTokens(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        # 1. 尝试获取本轮完整 prompt 的 Token IDs
        tokenizer = getattr(self, "_tokenizer", None)
        model = getattr(self, "_model", None)
        total_tokens: list[int] = []

        try:
            if model and hasattr(model, "tokenize"):
                # llama-cpp 原生方式
                prompt_str = self._build_chat_prompt(messages)
                total_tokens = list(model.tokenize(prompt_str.encode("utf-8")))
            elif tokenizer and hasattr(tokenizer, "encode"):
                prompt_str = self._build_chat_prompt(messages)
                total_tokens = list(tokenizer.encode(prompt_str))
        except Exception:
            total_tokens = []

        total_prompt_len = len(total_tokens) if total_tokens else self.count_tokens(messages)

        # 2. 与底层已缓存的 Token 序列进行最长公共前缀比对
        cached_tokens = getattr(self, "_last_input_tokens", [])
        matched_tokens = 0

        if total_tokens and cached_tokens:
            for t_new, t_old in zip(total_tokens, cached_tokens):
                if t_new == t_old:
                    matched_tokens += 1
                else:
                    break

        # 增量计算：总 Prompt 减去命中的公共前缀
        delta_tokens = max(0, total_prompt_len - matched_tokens)
        hit_rate = (matched_tokens / total_prompt_len * 100) if total_prompt_len > 0 else 0.0

        # 3. 计算本轮最大可用生成空间 (窗口上限 - 本轮总输入 - 安全裕度)
        total_ctx = self.context_limit()
        available_gen = max(0, total_ctx - total_prompt_len - 32) if total_ctx > 0 else 2048

        # 记录本轮 token 序列，供下一次比对
        if total_tokens:
            self._last_input_tokens = total_tokens

        return {
            "total_prompt_tokens": total_prompt_len,
            "cached_tokens": matched_tokens,
            "delta_tokens": delta_tokens,
            "hit_rate_pct": round(hit_rate, 2),
            "available_generation_tokens": available_gen,
        }
    
    #计算剩余的 KV Cache 
    def remaining_kvCache(self, used_tokens: int | None = None) -> int:
        if not self.is_loaded:
            return 0
        total_ctx = self.context_limit()
        if total_ctx <= 0:
            return 0

        # 1. 显式传入了当前轮次已用 tokens (prompt + completion)
        if used_tokens is not None:
            return max(0, total_ctx - used_tokens)

        # 2. 回退尝试读取底层引擎的实时已用 token 计数 (如 llama.cpp 的 n_tokens)
        current_used = getattr(self._model, "n_tokens", None)
        if current_used is not None and isinstance(current_used, int):
            return max(0, total_ctx - current_used)

        return total_ctx

    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        tokenizer = getattr(self, "_tokenizer", None)
        try:
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)