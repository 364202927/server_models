"""可复用项目基础设施。"""

from .common import (
    RecordBuffer,
    aContainB,
    configure_logging,
    dictFind,
    err,
    error,
    get_log_buffer,
    get_logger,
    info,
    joinPath,
    kError,
    kInfo,
    kLog,
    kWarn,
    listFind,
    log,
    logFormat,
    logJson,
    readFile,
    save_logs,
    set_console_active,
    str2time,
    switch,
    switchFn,
    warn,
    writeFile,
)
from .console import Console, console
from .paths import ASSETS_DIR, CACHE_DIR, CHAT_HISTORY_DIR, MODELS_FILE, RUNTIME_DIR, ensure_asset_dirs
from .recordBuffer import recordBuffer

__all__ = [
    "aContainB", "dictFind", "listFind", "readFile", "switch", "switchFn", "writeFile",
    "joinPath", "str2time", "ASSETS_DIR", "CACHE_DIR", "CHAT_HISTORY_DIR", "MODELS_FILE",
    "RUNTIME_DIR", "ensure_asset_dirs", "RecordBuffer", "recordBuffer", "get_log_buffer", "save_logs",
    "set_console_active", "configure_logging", "get_logger", "log", "info", "warn", "err",
    "error", "logFormat", "logJson", "kLog", "kInfo", "kWarn", "kError", "Console", "console",
]
