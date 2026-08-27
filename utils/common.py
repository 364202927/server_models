"""项目通用函数，避免依赖外部 ``server.utils`` 包。"""

import gzip
import json
import os
import pickle
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def getFileExtension(fileName: str) -> tuple[str, str]:
    name, extension = os.path.splitext(fileName)
    return extension[1:].lower(), name


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
