from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from ...utils import RUNTIME_DIR
from ...utils.common import info
from ...utils.hardware import detect_gpu
from .baseInference import RawOutput, baseInference

try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
except ImportError:
    llama_cpp = None
    Llama = None
    Jinja2ChatFormatter = None


def _gpu_offload_supported() -> bool | None:
    for module_name in ("llama_cpp", "llama_cpp.llama_cpp"):
        try:
            fn = getattr(importlib.import_module(module_name), "llama_supports_gpu_offload", None)
            if callable(fn):
                return bool(fn())
        except (ImportError, AttributeError, TypeError):
            continue
    return None


class llama(baseInference):
    """llama-cpp-python GGUF 推理引擎"""

    _FALLBACK_N_CTX = 8192
    _CONTEXT_SAFETY_MARGIN = 32
    _MIN_GENERATION_TOKENS = 16

    def __init__(self) -> None:
        super().__init__()
        self._saved_llm_kwargs: dict[str, Any] | None = None
        self._saved_draft_kwargs: dict[str, Any] | None = None
        self._draft_model = None
        self._formatter = None
        self._dbg_n = 0
        self._last_seq: list[int] = []  # 上一轮结束时引擎 KV 中的完整 token 序列（仅用于分叉诊断）

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

    def load(self, model_path: str, load_cfg: dict[str, Any]) -> "llama":
        if Llama is None:
            raise RuntimeError("GGUF 模型需要安装 llama-cpp-python")

        source = self._resolve_gguf_file(model_path)
        gpu_layers = int(load_cfg.get("gpu_offload_layers", -1))
        if gpu_layers != 0 and detect_gpu() and _gpu_offload_supported() is False:
            raise RuntimeError("当前 llama-cpp-python 未启用 CUDA，请安装 CUDA 构建版本。")

        # 计算并对齐 context 长度
        calculated_n_ctx = load_cfg.get("context", 0)
        if calculated_n_ctx <= llama._FALLBACK_N_CTX:
            calculated_n_ctx = llama._FALLBACK_N_CTX

        print("~~~~模型加载:ctx~~~~~~",calculated_n_ctx)
        llm_kwargs: dict[str, Any] = {
            "model_path": str(source),
            "n_gpu_layers": gpu_layers,
            "n_ctx": calculated_n_ctx,
            "n_batch": int(load_cfg.get("batch_size", 512)),
            "verbose": True,  # [DBG] 临时开启：stderr 输出前缀命中/partial kv removal 日志
            "use_mlock": bool(load_cfg.get("use_mlock", True)),
        }
        if "flash_attention" in load_cfg:
            llm_kwargs["flash_attn"] = bool(load_cfg["flash_attention"])

        lora = load_cfg.get("lora")
        if lora:
            llm_kwargs["lora_path"] = str(lora)

        # draft 处理：[0] 为主草稿模型，[1] 为辅助推测解码
        draft_list = load_cfg.get("draft") or []
        if isinstance(draft_list, str):
            draft_list = [draft_list]
        self._saved_draft_kwargs = None

        if draft_list and len(draft_list) > 0 and draft_list[0]:
            main_draft = self._resolve_gguf_file(draft_list[0])
            self._saved_draft_kwargs = {
                "model_path": str(main_draft),
                "n_gpu_layers": int(load_cfg.get("gpu_offload_layers", 0)),
            }
            self._draft_model = Llama(**self._saved_draft_kwargs)
            llm_kwargs["draft_model"] = self._draft_model

        self._saved_llm_kwargs = {k: v for k, v in llm_kwargs.items() if k != "draft_model"}
        self._model = Llama(**llm_kwargs)
        self._sleep_capable = True
        template = self._model.metadata.get("tokenizer.chat_template", "")
        self._dump("chat_template.jinja", template)
        info("[DBG] 模型信息", f"arch={self._model.metadata.get('general.architecture')}",
             f"chat_template长度={len(template)}(已写入 debug/chat_template.jinja)",
             f"n_ctx={self._model.n_ctx()}", f"n_batch={llm_kwargs['n_batch']}", f"draft={'有' if self._draft_model else '无'}")

        self._effective_load = {
            "engine": "llama",
            "context_length": int(self._model.n_ctx()),
            "gpu_offload_layers": llm_kwargs["n_gpu_layers"],
            "batch_size": llm_kwargs["n_batch"],
            "draft": draft_list,
            "lora": lora,
        }
        return self

    def _chat_formatter(self):
        """按 GGUF 内嵌 chat template 构造渲染器（与 create_chat_completion 实际喂给模型的 prompt 一致），惰性缓存。"""
        if self._formatter is None:
            template = self._model.metadata.get("tokenizer.chat_template")
            if not template:
                raise RuntimeError("GGUF 缺少 tokenizer.chat_template，无法计算消息 token 数")
            token_str = lambda tid: self._model.detokenize([tid]).decode("utf-8", errors="ignore")
            self._formatter = Jinja2ChatFormatter(
                template=template,
                eos_token=token_str(self._model.token_eos()),
                bos_token=token_str(self._model.token_bos()),
                add_generation_prompt=True,
            )
        return self._formatter

    def _prompt_tokens(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
        """消息 → 模板渲染 → token 序列，与 llama-cpp-python 的 chat handler 实际喂给模型的序列一致。"""
        prompt = self._chat_formatter()(messages=messages, tools=tools or None).prompt
        # 模板已含 BOS 时不再重复添加
        return self._model.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    def count_tokens(self, text_or_messages: str | list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        if isinstance(text_or_messages, str):
            return len(self._model.tokenize(text_or_messages.encode("utf-8")))
        return len(self._prompt_tokens(text_or_messages, tools))

    def _dump(self, name: str, text: str) -> None:
        """[DBG] 把调试文本落盘到 assets/runtime/debug/，便于 diff 相邻请求的渲染结果。"""
        try:
            path = RUNTIME_DIR / "debug"
            path.mkdir(parents=True, exist_ok=True)
            (path / name).write_text(text, encoding="utf-8")
        except OSError as exc:
            info("[DBG] 调试文件写入失败", name, exc)

    def _debug_divergence(self, tokens: list[int]) -> None:
        """[DBG] 落盘本轮渲染后的完整 prompt，并对比上一轮 KV 序列，打印分叉位置、公共前缀与前后文本。"""
        self._dbg_n += 1
        text = lambda seq: self._model.detokenize(seq, special=True).decode("utf-8", errors="replace")
        self._dump(f"prompt_{self._dbg_n:03d}.txt", text(tokens))
        old = self._last_seq
        info("[DBG] KV现状", f"请求#{self._dbg_n}", f"引擎n_tokens={self._model.n_tokens}",
             f"上轮记录={len(old)}", f"本轮prompt={len(tokens)}", f"(完整prompt已写入 debug/prompt_{self._dbg_n:03d}.txt)")
        if not old:
            info("[DBG] 分叉诊断: 无上一轮 KV 记录(首轮/刚唤醒)")
            return
        d = next((i for i, (a, b) in enumerate(zip(old, tokens)) if a != b), min(len(old), len(tokens)))
        show = lambda seq: repr(text(seq[max(0, d - 4):d + 6]))
        verdict = "严格延长(应命中)" if d == len(old) else f"在第 {d} 个 token 分叉(需回退 {len(old) - d} 步)"
        info("[DBG] 分叉诊断:", verdict, "| 上轮该处:", show(old), "| 本轮该处:", show(tokens))
        info("[DBG] 公共前缀", f"{d} tokens", "| 开头:", repr(text(tokens[:60])), "| 结尾:", repr(text(tokens[max(0, d - 20):d])))

    def _response(
        self,
        messages: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> RawOutput:
        cfg = dict(gen_cfg)
        tools = cfg.pop("tools", None)
        tool_choice = cfg.pop("tool_choice", None)
        prompt_ids = self._prompt_tokens(messages, tools)
        self._debug_divergence(prompt_ids)
        cfg["max_tokens"] = self._clamp_to_context(len(prompt_ids), int(cfg.pop("max_tokens", 512)))

        # 1. 字段映射对齐（llama-cpp-python 专有键名）
        if "repetition_penalty" in cfg:
            cfg["repeat_penalty"] = cfg.pop("repetition_penalty")
        if "mirostat" in cfg:
            cfg["mirostat_mode"] = cfg.pop("mirostat")

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
        if tool_choice:
            extra["tool_choice"] = tool_choice

        # 2. 过滤底层不支持的参数，避免 TypeError 崩溃
        valid_params = inspect.signature(self._model.create_chat_completion).parameters
        filtered_cfg = {k: v for k, v in cfg.items() if k in valid_params}

        # 3. 生成：perf 计数器的 n_p_eval = 本轮真正被重新计算的 prompt token 数
        llama_cpp.llama_perf_context_reset(self._model.ctx)
        result = self._model.create_chat_completion(messages=messages, **extra, **filtered_cfg)
        self._last_seq = self._model._input_ids.tolist()
        perf = llama_cpp.llama_perf_context(self._model.ctx)
        info("[DBG] 引擎计数器", f"n_p_eval(重算prompt)={perf.n_p_eval}", f"n_eval(生成)={perf.n_eval}",
             f"生成后n_tokens={self._model.n_tokens}", f"tools={len(tools or [])}", f"tool_choice={tool_choice}")
        choice = result["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls") or []
        usage = result.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        # 投机解码会让 n_p_eval 混入草稿验证批次，此时无法得到准确值
        cached = -1 if self._draft_model else max(0, prompt_tokens - llama_cpp.llama_perf_context(self._model.ctx).n_p_eval)

        return RawOutput(
            text=str(message.get("content") or ""),
            tokens=int(usage.get("completion_tokens", 0)),
            prompt_tokens=prompt_tokens,
            calls=calls,
            finish_reason=str(choice.get("finish_reason") or ("tool_calls" if calls else "stop")),
            cached_tokens=cached,
        )

    def _clamp_to_context(self, prompt_tokens: int, max_new_tokens: int) -> int:
        n_ctx = self._model.n_ctx()
        available = n_ctx - prompt_tokens - self._CONTEXT_SAFETY_MARGIN
        if available < self._MIN_GENERATION_TOKENS:
            raise ValueError(f"上下文不足: prompt 约 {prompt_tokens} token, 窗口大小 {n_ctx} token")
        return min(max_new_tokens, available)

    def _engine_sleep(self) -> None:
        self._last_seq = []
        model, self._model = self._model, None
        if hasattr(model, "close"):
            model.close()
        draft, self._draft_model = self._draft_model, None
        if draft and hasattr(draft, "close"):
            draft.close()

    def _engine_wake(self) -> None:
        if self._saved_draft_kwargs:
            self._draft_model = Llama(**self._saved_draft_kwargs)
        if self._saved_llm_kwargs:
            kw = dict(self._saved_llm_kwargs)
            if self._draft_model:
                kw["draft_model"] = self._draft_model
            self._model = Llama(**kw)

    def _unload_engine(self) -> None:
        self._engine_sleep()
        self._mark_unloaded()
        self._formatter = None
        self._saved_llm_kwargs = None
        self._saved_draft_kwargs = None
        self.release_cache()