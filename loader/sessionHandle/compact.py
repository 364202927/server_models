from __future__ import annotations

import asyncio
import copy
import re
from collections import OrderedDict
from typing import Any

from ...utils.common import (
    compute_messages_fingerprint,
    info as log_info,
    skeletonize_code,
)


class compact:
    """压缩上下文"""

    # LRU 缓存：Key = 会话指纹, Value = 压缩后的全局 DSL (S_latest)
    _LRU_MAX_SIZE = 128
    _FINGERPRINT_CACHE: OrderedDict[str, str] = OrderedDict()
    _LOCK = asyncio.Lock()

    DSL_START = "[STATE_MACHINE]"
    DSL_END = "[/STATE_MACHINE]"

    @classmethod
    def get_cached_dsl(cls, fingerprint: str) -> str | None:
        if fingerprint in cls._FINGERPRINT_CACHE:
            cls._FINGERPRINT_CACHE.move_to_end(fingerprint)
            return cls._FINGERPRINT_CACHE[fingerprint]
        return None

    @classmethod
    def set_cached_dsl(cls, fingerprint: str, dsl: str) -> None:
        if fingerprint in cls._FINGERPRINT_CACHE:
            cls._FINGERPRINT_CACHE.move_to_end(fingerprint)
        cls._FINGERPRINT_CACHE[fingerprint] = dsl
        if len(cls._FINGERPRINT_CACHE) > cls._LRU_MAX_SIZE:
            cls._FINGERPRINT_CACHE.popitem(last=False)

    @classmethod
    def extract_dsl_snapshots(cls, messages: list[dict[str, Any]]) -> list[str]:
        """仅提取历史消息中零散的 DSL 纯文本块，严禁塞入代码。"""
        snapshots = []
        pattern = re.compile(r"\[STATE_MACHINE\](.*?)\[/STATE_MACHINE\]", re.DOTALL)
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                matches = pattern.findall(content)
                snapshots.extend([m.strip() for m in matches if m.strip()])
        return snapshots

    @classmethod
    def purge_dsl_blocks(cls, text: str) -> str:
        """剥离文本中的旧 DSL 块。"""
        cleaned = re.sub(r"\[STATE_MACHINE\].*?\[/STATE_MACHINE\]", "", text, flags=re.DOTALL)
        return cleaned.strip()

    @classmethod
    def skeletonize_history_codes(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """第一道防线：沉淀为历史时将 Turn 1 及之前的代码剥离为骨架。"""
        if len(messages) <= 2:
            return messages

        # 保护当轮（倒数前 2 条通常是当轮最新问答），对前面的历史执行骨架化
        history = [copy.deepcopy(m) for m in messages[:-2]]
        recent = messages[-2:]

        code_block_regex = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

        for msg in history:
            content = msg.get("content")
            if isinstance(content, str) and "```" in content:

                def repl(match):
                    lang = match.group(1)
                    code = match.group(2)
                    return f"```{lang}\n{skeletonize_code(code, lang)}```"

                msg["content"] = code_block_regex.sub(repl, content)

        return history + recent

    @classmethod
    def apply_third_defense_sliding_window(
        cls,
        messages: list[dict[str, Any]],
        token_count: int,
        context_limit: int,
        latest_dsl: str = "",
    ) -> list[dict[str, Any]]:
        """第三道防线：达到 85% 水位线触发 FIFO 滑窗成对驱逐，锁定锚点。"""
        if context_limit <= 0 or (token_count / context_limit) < 0.85:
            return messages

        if len(messages) <= 4:
            return messages

        log_info("触发第三道防线：85% 上下文水位滑窗驱逐")

        # 锚点 1：System 消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_rounds = [m for m in messages if m.get("role") != "system"]

        # 锚点 2：保留最新 1~2 轮问答
        protected_count = min(4, len(chat_rounds))
        middle_rounds = chat_rounds[:-protected_count]
        recent_rounds = chat_rounds[-protected_count:]

        # FIFO 轮次淘汰最旧的成对问答
        while len(middle_rounds) >= 2:
            middle_rounds.pop(0)  # User
            middle_rounds.pop(0)  # Assistant
            # 简化截断，若驱逐一对即可达到释放效果则退出
            break

        # 锚点 3：若存在全局状态快照 S_latest，确保挂载在首部
        result = system_msgs + middle_rounds + recent_rounds
        if latest_dsl and result:
            # 附着到第一条非 system 消息或更新到 system 尾部
            target = result[0]
            if target.get("role") == "system":
                if cls.DSL_START not in target.get("content", ""):
                    target["content"] = f"{target.get('content', '')}\n\n[STATE_MACHINE]\n{latest_dsl}\n[/STATE_MACHINE]".strip()

        return result

    @classmethod
    def apply_ingress(
        cls,
        messages: list[dict[str, Any]],
        strategy: str,
        context_limit: int,
        token_count: int,
    ) -> list[dict[str, Any]]:
        """请求入口主治理管道。"""
        if strategy != "code":
            return messages

        # 1. 第一道防线：历史代码骨架化
        processed = cls.skeletonize_history_codes(messages)

        # 2. 第二道防线：计算前序指纹，若命中 S_latest 则就地收敛
        fingerprint = compute_messages_fingerprint(processed[:-1] if len(processed) > 1 else processed)
        cached_dsl = cls.get_cached_dsl(fingerprint)

        if cached_dsl:
            # 清除前面零散旧 DSL
            for m in processed[:-1]:
                if isinstance(m.get("content"), str):
                    m["content"] = cls.purge_dsl_blocks(m["content"])
            # 将最新的 S_latest 注入给最新的一条历史 Assistant，无损衔接
            last_assistant = next((m for m in reversed(processed[:-1]) if m.get("role") == "assistant"), None)
            if last_assistant and isinstance(last_assistant.get("content"), str):
                last_assistant["content"] = f"[STATE_MACHINE]\n{cached_dsl}\n[/STATE_MACHINE]\n{last_assistant['content']}"

        # 3. 第三道防线：85% 水位滑窗驱逐
        processed = cls.apply_third_defense_sliding_window(processed, token_count, context_limit, latest_dsl=cached_dsl or "")

        return processed

    @classmethod
    async def try_background_squash(
        cls,
        model_id: str,
        messages: list[dict[str, Any]],
        current_reply: str,
        models_mgr: Any,
        context_limit: int,
    ) -> None:
        """后台滞后折叠任务：在上一轮回复完成后运行，零 TTFT 损耗。"""
        try:
            full_messages = list(messages) + [{"role": "assistant", "content": current_reply}]
            snapshots = cls.extract_dsl_snapshots(full_messages)

            # 若零散 DSL 数量不足，无需合并
            if len(snapshots) < 2:
                return

            # 评估水位线：DSL 累积较多或接近 75% 时触发
            rough_tokens = sum(len(str(m.get("content", ""))) // 4 for m in full_messages)
            if context_limit > 0 and (rough_tokens / context_limit) < 0.70:
                return

            log_info("启动第二道防线：后台异步状态机 DSL 折叠 (Squash)")

            # 输入隔离：纯文本 DSL 摘要提取，不含大段代码
            squash_prompt = "Combine the following state machine DSL fragments into a single, unified latest state snapshot.\n" "Constraints: only output pure DSL inside [STATE_MACHINE]...[/STATE_MACHINE], tokens <= 128.\n\n" + "\n---\n".join(snapshots)

            squash_messages = [
                {"role": "system", "content": "You are a state compression engine."},
                {"role": "user", "content": squash_prompt},
            ]

            gen_cfg = {
                "max_tokens": 128,
                "temperature": 0.1,
            }

            # 异步跑微型模型折叠
            res = await asyncio.to_thread(models_mgr.generate, model_id, squash_messages, gen_cfg)
            match = re.search(r"\[STATE_MACHINE\](.*?)\[/STATE_MACHINE\]", res.text, re.DOTALL)
            new_dsl = match.group(1).strip() if match else res.text.strip()

            # 将合并后的 S_latest 写入 LRU 缓存，挂载至指纹
            fp = compute_messages_fingerprint(full_messages)
            cls.set_cached_dsl(fp, new_dsl)
            log_info("后台 DSL 折叠完成，已更新会话指纹缓存", fp[:8])
        except Exception as exc:
            log_info("后台 DSL 折叠异常 (已忽略以保证安全)", exc)
