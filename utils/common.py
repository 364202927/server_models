import pickle,gzip,json,os,inspect
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from typing import Any
from datetime import datetime
from .recordBuffer import RecordBuffer
from pathlib import Path
from .logger import (configure_logging, error, get_log_buffer, get_logger, info, kError, kInfo,
    kLog, kWarn, log, logFormat, save_logs, set_console_active, str2time, warn,
)

# True 时在 main.py 额外启动 FastAPI；Console 始终启动。可用 AI_KAPI=false 覆盖。
kApi = os.getenv("AI_KAPI", "true").strip().lower() not in {"0", "false", "no", "off"}


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
def switchV(dice, key1, key2):
    return dice.get(key1) or dice.get(key2)

# 文件操作
def readFile(pathFile, model='r'):
    if not os.path.isfile(pathFile):  #检测文件是否存在(isfile对不存在的路径也返回False)
        return None
    fileType, _ = getFileExtension(pathFile)
    def _read_json() -> Any:
        with open(pathFile, model, encoding="utf-8") as file:
            return json.load(file)
    def _read_jsonl() -> list[Any]:
        with open(pathFile, model, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    def _read_pickle(p) -> Any:
        with open(pathFile, "rb") as file:
            return pickle.load(file)
    def _read_text() -> str:
        with open(pathFile, model, encoding="utf-8") as file:
            return file.read()
    def _read_gzip() -> Any:
        with gzip.open(pathFile, "rb") as file:
            return pickle.load(file)
    return switchFn({"json": lambda: _read_json(),
                        "jsonl": lambda: _read_jsonl(),
                        "pkl": lambda: _read_pickle(pathFile),
                        "txt": lambda: _read_text(),
                        "gz": lambda: _read_gzip(),
                    }, key=fileType)
def writeFile(data, pathFile, model='w'):
    if data is None or (isinstance(data, (list, dict)) and len(data) == 0):
        return False
    fileType, _ = getFileExtension(pathFile)
    def _json():
        with open(pathFile, model, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    def _jsonl():
        with open(pathFile, model, encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    def _pkl():
        with open(pathFile, 'wb') as f:
            pickle.dump(data, f)
    def _txt():
        with open(pathFile, model, encoding='utf-8') as f:
            f.write(str(data))
    result = switchFn({'json': lambda: _json(),
                        'jsonl': lambda: _jsonl(),
                        'pkl': lambda: _pkl(),
                        'txt': lambda: _txt(),
                    }, key=fileType)
    # switchFn 找不到对应文件类型时返回 False,否则(即便回调无显式返回值/None)视为成功
    return result is not False

# 搜索路径下的文件
def path2File(path, fileType=''):
    try:
        items = os.listdir(path)
        files = [item for item in items
            if os.path.isfile(os.path.join(path, item))
            and item.endswith(fileType)]
        return files
    except Exception as e:
        print(e)
    return []
# 加载文件
def require(modPath):
    mod = import_module(modPath)
    className = modPath.rsplit('.', 1)[-1]
    obj = getattr(mod, className, None)
    if obj is None:
        print(className, "类创建失败，请检查路径", mod)
    return obj

# 当前文件的工作路径
def curPath():
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    return os.path.dirname(os.path.realpath(caller_file)) + '/'
# 加载路径
def joinPath(path, fileName):
    return path + fileName
def getRootName(cls, rootDir: str) -> str:
    try:
        # 获取传入类所在的文件
        strategy_file = Path(inspect.getfile(cls))
        # 向上查找指定的基目录并返回其下一级目录名
        parent = next((p for p in strategy_file.parents if p.name == rootDir), None)
        return strategy_file.relative_to(parent).parts[0] if parent else ''
    except (TypeError, OSError):
        return ''
def getFileExtension(fileName: str) -> tuple[str, str]:
    name, extension = os.path.splitext(fileName)
    return extension[1:].lower(), name
def joinPath(*parts: str | os.PathLike[str]) -> str:
    return os.path.join(*(os.fspath(part) for part in parts))


def ensure_asset_dirs() -> None:
    """创建项目运行时需要的资源目录。"""
    from . import CACHE_DIR, CHAT_HISTORY_DIR, RUNTIME_DIR

    for path in (RUNTIME_DIR, CACHE_DIR, CHAT_HISTORY_DIR):
        path.mkdir(parents=True, exist_ok=True)