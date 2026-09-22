"""项目通用工具与统一日志入口。"""

import gzip
import json
import logging
import os
import pickle
import pprint
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .recordBuffer import RecordBuffer


kLog = "log"
kInfo = "info"
kWarn = "warn"
kError = "error"

# True 时在 main.py 额外启动 FastAPI；Console 始终启动。可用 AI_KAPI=false 覆盖。
kApi = os.getenv("AI_KAPI", "True").strip().lower() not in {"0", "false", "no", "off"}

_LOG_LEVEL_TAGS = {
    logging.DEBUG: kLog,
    logging.INFO: kInfo,
    logging.WARNING: kWarn,
    logging.ERROR: kError,
    logging.CRITICAL: kError,
}
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "assets" / "runtime" / "logs"


def _env_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


_logBuffer = RecordBuffer(
    filePath=os.getenv("AI_LOG_DIR", str(_DEFAULT_LOG_DIR)),
    max_size=_env_positive_int("AI_LOG_BUFFER_SIZE", 4096),
)
_logLock = threading.RLock()
_consoleActive = False
_printLogs = True


def getFileExtension(fileName: str) -> tuple[str, str]:
    name, extension = os.path.splitext(fileName)
    return extension[1:].lower(), name


def joinPath(*parts: str | os.PathLike[str]) -> str:
    """拼接路径并返回字符串，兼容旧工具调用方式。"""
    return os.path.join(*(os.fspath(part) for part in parts))


def str2time(mode: str = "strNow", value: datetime | None = None) -> str:
    """返回本地时间字符串。

    ``strNow`` 用于日志，``date`` 用于按日文件名，其他值返回 ISO 时间。
    """
    current = value or datetime.now().astimezone()
    if mode == "strNow":
        # 日志打印精确到秒；毫秒会让 Console 输出难以阅读且没有业务价值。
        return current.strftime("%Y-%m-%d %H:%M:%S")
    if mode == "date":
        return current.strftime("%Y-%m-%d")
    return current.isoformat(timespec="seconds")


def listFind(lists: Iterable[Any], fnJudge: Callable[[Any], bool]) -> Any | None:
    return next((item for item in lists if fnJudge(item)), None)


def dictFind(d: Mapping[Any, Any], fnJudge: Callable[[Any, Any], bool]) -> tuple[Any, Any] | None:
    return next(((key, value) for key, value in d.items() if fnJudge(key, value)), None)


def aContainB(src: str, strOrTab: Iterable[str]) -> bool:
    return any(value in src for value in strOrTab)


def switch(dice: Mapping[Any, Any], key: Any) -> Any:
    return dice.get(key) or dice.get("default") or False


def switchFn(diceFn: Mapping[str, Callable[..., Any]], key: str, **kwargs: Any) -> Any:
    fn = diceFn.get(key) or diceFn.get("default")
    return fn(**kwargs) if fn else False


def readFile(pathFile: str, model: str = "r") -> Any:
    """按扩展名读取配置和测试文件。"""
    if not os.path.isfile(pathFile):
        return None
    fileType, _ = getFileExtension(pathFile)
    readers: dict[str, Callable[[], Any]] = {
        "json": lambda: _read_json(pathFile, model),
        "jsonl": lambda: _read_jsonl(pathFile, model),
        "pkl": lambda: _read_pickle(pathFile),
        "txt": lambda: _read_text(pathFile, model),
        "gz": lambda: _read_gzip(pathFile),
    }
    return readers[fileType]() if fileType in readers else False


def writeFile(data: Any, pathFile: str, model: str = "w") -> bool:
    """写入项目配置文件；目前 JSON/JSONL/TXT 为主要用途。"""
    directory = os.path.dirname(pathFile)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fileType, _ = getFileExtension(pathFile)
    if fileType == "json":
        with open(pathFile, model, encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return True
    if fileType == "jsonl":
        with open(pathFile, model, encoding="utf-8") as file:
            for item in data:
                file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return True
    if fileType == "txt":
        with open(pathFile, model, encoding="utf-8") as file:
            file.write(str(data))
        return True
    return False


def _read_json(pathFile: str, model: str) -> Any:
    with open(pathFile, model, encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(pathFile: str, model: str) -> list[Any]:
    with open(pathFile, model, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _read_pickle(pathFile: str) -> Any:
    with open(pathFile, "rb") as file:
        return pickle.load(file)


def _read_text(pathFile: str, model: str) -> str:
    with open(pathFile, model, encoding="utf-8") as file:
        return file.read()


def _read_gzip(pathFile: str) -> Any:
    with gzip.open(pathFile, "rb") as file:
        return pickle.load(file)


def get_log_buffer() -> RecordBuffer:
    """返回进程级日志缓冲区，供 Console 和状态接口读取。"""
    return _logBuffer


def set_console_active(active: bool) -> None:
    """Console 接管终端时暂停普通打印，避免破坏输入行。"""
    global _consoleActive
    _consoleActive = active


def configure_logging(
    *,
    level: int | str = logging.INFO,
    print_to_console: bool = True,
) -> None:
    """配置标准 logging 和自定义日志函数的输出级别与终端行为。"""
    global _printLogs
    _printLogs = print_to_console
    logging.getLogger("ai").setLevel(level)


def _caller_location(depth: int) -> str:
    try:
        frame = sys._getframe(depth)
        file_name = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
        return f"[{file_name}.{frame.f_code.co_name}:{frame.f_lineno}]"
    except (ValueError, AttributeError):
        return "[Unknown_Location]"


def _push_log(tag: str, message: str) -> str:
    with _logLock:
        record_id = _logBuffer.push(msg=message, tags=tag)
        if _printLogs and not _consoleActive:
            print(message, flush=True)
        return record_id


def _logBase(tag: str, *args: Any) -> str:
    """写入结构化缓冲区；普通 log 带调用位置，其余级别带时间。"""
    prefix = _caller_location(3) if tag == kLog else f"[{str2time('strNow')}]"
    return _push_log(tag, prefix + "".join(map(str, args)))


def log(*msgs: Any) -> str:
    return _logBase(kLog, *msgs)


def info(*msgs: Any) -> str:
    return _logBase(kInfo, *msgs)


def warn(*msgs: Any) -> str:
    return _logBase(kWarn, *msgs)


def err(*msgs: Any) -> str:
    return _logBase(kError, *msgs)


error = err


def logFormat(value: Any) -> str:
    return _logBase(kLog, "\n", pprint.pformat(value))


def logJson(value: Any) -> str:
    return _logBase(kLog, "\n", json.dumps(value, indent=4, ensure_ascii=False, default=str))


def save_logs() -> bool:
    """将日志缓冲区中尚未保存的记录持久化到按日 JSONL 文件。"""
    return _logBuffer.save2File()


class _RecordBufferHandler(logging.Handler):
    """把标准库 logging 记录转入项目日志缓冲区。"""

    _ai_buffer_handler = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tag = _LOG_LEVEL_TAGS.get(record.levelno, kLog)
            if tag == kLog:
                prefix = f"[{Path(record.pathname).stem}.{record.funcName}:{record.lineno}]"
            else:
                prefix = f"[{str2time('strNow')}]"
            _push_log(tag, f"{prefix}[{record.name}]{record.getMessage()}")
        except Exception:
            self.handleError(record)


def get_logger(name: str = "ai", level: int | str | None = None) -> logging.Logger:
    """返回已接入 RecordBuffer 的标准 ``logging.Logger``。

    同一 logger 只安装一个项目 Handler，因此多次调用不会重复打印。
    """
    logger = logging.getLogger(name)
    if not any(getattr(handler, "_ai_buffer_handler", False) for handler in logger.handlers):
        logger.addHandler(_RecordBufferHandler())
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
