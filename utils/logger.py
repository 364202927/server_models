import json,threading,sys,pprint,os,logging
from pathlib import Path
from datetime import datetime
from typing import Any
from .recordBuffer import RecordBuffer

kLog = "log"
kInfo = "info"
kWarn = "warn"
kError = "error"

_LOG_LEVEL_TAGS = {
    logging.DEBUG: kLog,
    logging.INFO: kInfo,
    logging.WARNING: kWarn,
    logging.ERROR: kError,
    logging.CRITICAL: kError,
}
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "assets" / "runtime" / "logs"

#输出格式[文件名.函数名:行号]
def _format_location(file_stem: str, func_name: str, line_no: int) -> str:
    return f"[{file_stem}.{func_name}:{line_no}]"
def _env_positive_int(name: str, default: int) -> int:
    return max(1, int(os.getenv(name, str(default))))

class _LogState:
    """集中管理日志缓冲区与运行时开关，取代原先分散、靠 global 读写的模块级变量。"""

    def __init__(self, log_dir: str, buffer_size: int) -> None:
        self.buffer = RecordBuffer(filePath=log_dir, max_size=buffer_size)
        self.lock = threading.RLock()
        self.console_active = False
        self.print_to_console = True

    def push(self, tag: str, message: str) -> str:
        """加锁写入缓冲区；若终端未被 Console 接管且允许打印，则同步输出。"""
        with self.lock:
            record_id = self.buffer.push(msg=message, tags=tag)
            if self.print_to_console and not self.console_active:
                print(message, flush=True)
            return record_id

_state = _LogState(
    log_dir=os.getenv("AI_LOG_DIR", str(_DEFAULT_LOG_DIR)),
    buffer_size=_env_positive_int("AI_LOG_BUFFER_SIZE", 4096),
)

def get_log_buffer() -> RecordBuffer:
    """返回进程级日志缓冲区，供 Console 和状态接口读取。"""
    return _state.buffer
def set_console_active(active: bool) -> None:
    """Console 接管终端时暂停普通打印，避免破坏输入行。"""
    _state.console_active = active


def configure_logging(*, level: int | str = logging.INFO, print_to_console: bool = True) -> None:
    """配置标准 logging 和自定义日志函数的输出级别与终端行为。"""
    _state.print_to_console = print_to_console
    logging.getLogger("ai").setLevel(level)


def _logBase(tag: str, *args: Any) -> str:
    if tag == kLog:
        try:
            # 帧 0=本函数 1=log/info/warn/err 2=真正调用者
            frame = sys._getframe(2)
            file_stem = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
            prefix = _format_location(file_stem, frame.f_code.co_name, frame.f_lineno)
        except (ValueError, AttributeError):
            prefix = "[Unknown_Location]"
    else:
        prefix = f"[{str2time('strNow')}]"
    return _state.push(tag, prefix + "".join(map(str, args)))


def log(*msgs: Any) -> str:
    return _logBase(kLog, *msgs)
def info(*msgs: Any) -> str:
    return _logBase(kInfo, *msgs)
def warn(*msgs: Any) -> str:
    return _logBase(kWarn, *msgs)
def err(*msgs: Any) -> str:
    return _logBase(kError, *msgs)

def logFormat(value: Any) -> str:
    return _logBase(kLog, "\n", pprint.pformat(value))
def logJson(value: Any) -> str:
    return _logBase(kLog, "\n", json.dumps(value, indent=4, ensure_ascii=False, default=str))
def save_logs() -> bool:
    return _state.buffer.save2File()

class _RecordBufferHandler(logging.Handler):
    _ai_buffer_handler = True
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tag = _LOG_LEVEL_TAGS.get(record.levelno, kLog)
            if tag == kLog:
                location = _format_location(Path(record.pathname).stem, record.funcName, record.lineno)
            else:
                location = f"[{str2time('strNow')}]"
            _state.push(tag, f"{location}[{record.name}]{record.getMessage()}")
        except Exception:
            self.handleError(record)

#返回已接入 RecordBuffer 的标准 ``logging.Logger
def get_logger(name: str = "ai", level: int | str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    has_buffer_handler = any(getattr(h, "_ai_buffer_handler", False) for h in logger.handlers)
    if not has_buffer_handler:
        logger.addHandler(_RecordBufferHandler())
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


__all__ = [
    "RecordBuffer", "str2time", "get_log_buffer", "set_console_active", "configure_logging",
    "get_logger", "log", "info", "warn", "err", "error", "logFormat", "logJson", "save_logs",
    "kLog", "kInfo", "kWarn", "kError",
]
