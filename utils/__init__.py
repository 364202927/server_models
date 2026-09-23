"""可复用项目基础设施。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_FILE = ASSETS_DIR / "models.json"
RUNTIME_DIR = ASSETS_DIR / "runtime"
CACHE_DIR = ASSETS_DIR / "cache"
CHAT_HISTORY_DIR = ASSETS_DIR / "chat_history"

from .common import (
    RecordBuffer,
    aContainB,
    configure_logging,
    dictFind,
    error,
    ensure_asset_dirs,
    get_log_buffer,
    get_logger,
    info,
    joinPath,
    kError,
    kApi,
    kInfo,
    kLog,
    kWarn,
    listFind,
    log,
    logFormat,
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
from .recordBuffer import recordBuffer

__all__ = [
    "aContainB", "dictFind", "listFind", "readFile", "switch", "switchFn", "writeFile",
    "joinPath", "str2time", "ASSETS_DIR", "CACHE_DIR", "CHAT_HISTORY_DIR", "MODELS_FILE",
    "RUNTIME_DIR", "ensure_asset_dirs", "RecordBuffer", "recordBuffer", "get_log_buffer", "save_logs",
    "set_console_active", "configure_logging", "get_logger", "log", "info", "warn",
    "error", "logFormat", "kLog", "kInfo", "kWarn", "kError", "kApi", "Console", "console",
]
