"""异步 Console 监控器：显示缓冲日志并解析管理命令。"""

from __future__ import annotations

import ast
import asyncio
import inspect
import os
import re
import signal
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from .common import (
    get_log_buffer,
    kError,
    kInfo,
    kLog,
    kWarn,
    log,
    set_console_active,
    warn,
)
from .recordBuffer import RecordBuffer


_IS_WIN = sys.platform == "win32"
if _IS_WIN:
    import msvcrt
else:
    import termios
    import tty


kColor = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}
kReset = "\033[0m"
kTagColor = {kError: kColor["red"], kWarn: kColor["yellow"],
             kInfo: kColor["cyan"], kLog: kColor["white"]}
kLogFilter = {kLog, kInfo, kError, kWarn}

CommandHandler = Callable[[int, Any], Awaitable[Any] | Any]


class Console:
    """在单独协程中运行的终端监控器。

    ``command_handler`` 接收解析后的 ``(id, args)``。它可以是同步函数或
    异步函数，因而能够连接模型管理器或服务层，而不依赖旧项目事件总线。
    """

    def __init__(
        self,
        command_handler: CommandHandler | None = None,
        log_buffer: RecordBuffer | None = None,
        log_filter: set[str] | None = None,
    ) -> None:
        self._command_handler = command_handler
        self._log_buffer = log_buffer or get_log_buffer()
        self._log_filter = log_filter or set(kLogFilter)

    @staticmethod
    def _str2Id(src: str) -> dict[str, Any] | None:
        """解析 ``id``、数字列表、``key=value`` 或字典参数。"""
        parts = [part.strip() for part in src.strip().split(",", 1)]
        if not parts or not parts[0].isdigit():
            return None
        command_id = int(parts[0])
        if len(parts) == 1 or not parts[1]:
            return {"id": command_id}
        rest = parts[1].strip()
        if rest.startswith("{"):
            value = ast.literal_eval(rest)
            if not isinstance(value, dict):
                raise ValueError("花括号参数必须是字典")
            return {"id": command_id, "args": value}
        if "=" in rest:
            values: dict[str, Any] = {}
            for item in re.split(r"[,\s]+", rest):
                if not item:
                    continue
                key, raw_value = item.split("=", 1)
                values[key.strip()] = Console._parse_value(raw_value.strip())
            return {"id": command_id, "args": values}
        return {
            "id": command_id,
            "args": [Console._parse_value(value) for value in re.split(r"[,\s]+", rest) if value],
        }

    @staticmethod
    def _parse_value(value: str) -> Any:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    async def _printNew(self, current_input: str) -> None:
        records = self._log_buffer.getNew()
        visible = [record for record in records if record.get("tags") in self._log_filter]
        if not visible:
            return
        lines = "".join(
            f"{kTagColor.get(str(record.get('tags')), kColor['white'])}"
            f"{record.get('data', {}).get('msg', '')}{kReset}\n"
            for record in visible
        )
        self._write(f"\r\033[K{lines}{kColor['magenta']}>>> {current_input}{kReset}")

    @staticmethod
    def _write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    @staticmethod
    def _initInput(
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[str],
    ) -> Callable[[], None]:
        if not sys.stdin.isatty():
            raise RuntimeError("Console 需要交互式终端 TTY")
        enqueue = queue.put_nowait
        if _IS_WIN:
            stop = threading.Event()

            def _reader() -> None:
                while not stop.is_set():
                    if msvcrt.kbhit():
                        char = msvcrt.getwch()
                        if char in ("\x00", "\xe0"):
                            msvcrt.getwch()
                        else:
                            loop.call_soon_threadsafe(enqueue, char)
                    stop.wait(0.02)

            threading.Thread(target=_reader, daemon=True, name="console-input").start()
            return stop.set

        file_descriptor = sys.stdin.fileno()
        old_attributes = termios.tcgetattr(file_descriptor)
        tty.setcbreak(file_descriptor)
        loop.add_reader(file_descriptor, lambda: enqueue(sys.stdin.read(1)))

        def _cleanup() -> None:
            loop.remove_reader(file_descriptor)
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_attributes)

        return _cleanup

    async def run(self) -> None:
        """启动交互式监控；Ctrl+C 交由主进程执行统一停机。"""
        if not sys.stdin.isatty():
            warn("Console 未检测到交互式 TTY，跳过终端监听")
            return
        self._write("\n=== Console Monitor Started ===\n")
        self._write("输入 h 查看帮助，Ctrl+C 退出。\n")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()
        cleanup = self._initInput(loop, queue)
        current_input = ""
        set_console_active(True)
        self._write(f"{kColor['magenta']}>>> {kReset}")
        try:
            while True:
                try:
                    char = await asyncio.wait_for(queue.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    await self._printNew(current_input)
                    continue
                if char == "\x03":
                    os.kill(os.getpid(), signal.SIGINT)
                    return
                if char in ("\n", "\r"):
                    self._write("\n")
                    command = current_input.strip()
                    current_input = ""
                    if command:
                        try:
                            await self._handle_command(command)
                        except Exception as exc:
                            warn("输入异常:", exc)
                    self._write(f"{kColor['magenta']}>>> {kReset}")
                elif char in ("\x7f", "\x08"):
                    if current_input:
                        current_input = current_input[:-1]
                        self._write("\b \b")
                else:
                    current_input += char
                    self._write(char)
                await self._printNew(current_input)
        finally:
            set_console_active(False)
            cleanup()
            self._log_buffer.save2File()

    async def _handle_command(self, command: str) -> Any:
        if command in {"h", "help"}:
            log(
                "可用命令:"
                "\n  <id>                  - 发送无参数命令"
                "\n  <id>,1,2              - 发送参数列表"
                "\n  <id>,x=1,name=test    - 发送键值参数"
                "\n  <id>,{'x': 1}         - 发送字典参数"
                "\n  h/help                - 显示帮助"
            )
            return None
        values = self._str2Id(command)
        if values is None:
            raise ValueError(f"无法识别命令: {command}")
        if self._command_handler is None:
            warn("Console 尚未配置命令处理器，收到命令:", values)
            return values
        result = self._command_handler(values["id"], values.get("args"))
        return await result if inspect.isawaitable(result) else result

    # 新代码可使用 snake_case；保留参考项目方法名。
    parse_command = _str2Id
    print_new = _printNew
    handle_command = _handle_command


console = Console

__all__ = [
    "Console", "console", "CommandHandler", "kColor", "kReset", "kTagColor", "kLogFilter",
]
