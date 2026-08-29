"""线程安全的内存记录缓冲区，支持查询和按日持久化 JSONL。"""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _now_string() -> str:
    # 与 Console 日志保持一致，记录时间精确到秒即可。
    return datetime.now().astimezone().isoformat(timespec="seconds")


class recordBuffer:
    """保存日志等结构化记录的有界缓冲区。

    每条记录包含 ``id``、``time``、``tags`` 和 ``data``。内部序号只用于
    正确追踪新增/未保存记录，即使 ``deque`` 淘汰旧记录也不会错过新日志。
    """

    def __init__(self, filePath: str = "", max_size: int = 1024) -> None:
        if max_size <= 0:
            raise ValueError("max_size 必须大于 0")
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._filePath = Path(filePath).expanduser() if filePath else None
        self._index_map: dict[str, dict[str, Any]] = {}
        self._next_sequence = 1
        self._printed_sequence = 0
        self._saved_sequence = 0
        self._lock = threading.RLock()

    def push(self, **kwargs: Any) -> str:
        """添加一条记录并返回短 ID。"""
        with self._lock:
            log_id = uuid.uuid4().hex[:8]
            record_time = kwargs.pop("time", _now_string())
            tags = kwargs.pop("tags", [str(key) for key in kwargs])
            record = {
                "id": log_id,
                "time": record_time,
                "tags": tags,
                "data": kwargs,
                "_sequence": self._next_sequence,
            }
            self._next_sequence += 1
            evicted = self._buffer[0] if len(self._buffer) == self._buffer.maxlen else None
            self._buffer.append(record)
            if evicted is not None:
                self._index_map.pop(str(evicted.get("id", "")), None)
            self._index_map[log_id] = record
            return log_id

    @staticmethod
    def _matches(record: dict[str, Any], filters: dict[str, Any], match: bool) -> bool:
        results: list[bool] = []
        data = record.get("data", {})
        for key, expected in filters.items():
            actual = record.get(key) if key in {"id", "time", "tags"} else data.get(key)
            if key == "tags" and isinstance(actual, (list, tuple, set)):
                results.append(expected in actual)
            else:
                results.append(actual == expected)
        return all(results) if match else any(results)

    def get(
        self,
        se_time: tuple[str, str] | None = None,
        match: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        """按时间、标签或数据字段查询；传入 ``id`` 时返回单条记录。"""
        with self._lock:
            if kwargs.get("id"):
                return self._public_record(self._index_map.get(str(kwargs["id"])))
            start_time = se_time[0] if se_time else None
            end_time = se_time[1] if se_time and len(se_time) > 1 else None
            result: list[dict[str, Any]] = []
            for record in self._buffer:
                record_time = str(record.get("time", ""))
                if start_time and record_time < start_time:
                    continue
                if end_time and record_time > end_time:
                    continue
                if kwargs and not self._matches(record, kwargs, match):
                    continue
                result.append(self._public_record(record) or {})
            return result

    def getNew(self) -> list[dict[str, Any]]:
        """返回上次调用后新增的记录。"""
        with self._lock:
            records = [record for record in self._buffer
                       if int(record.get("_sequence", 0)) > self._printed_sequence]
            if records:
                self._printed_sequence = int(records[-1]["_sequence"])
            return [self._public_record(record) or {} for record in records]

    def update(self, id: str, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            record = self._index_map.get(id)
            if not record:
                return {}
            if "tags" in kwargs:
                record["tags"] = kwargs.pop("tags")
            if "time" in kwargs:
                record["time"] = kwargs.pop("time")
            record["data"].update(kwargs)
            return self._public_record(record) or {}

    @staticmethod
    def _public_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return {key: value for key, value in record.items() if key != "_sequence"}

    def save2File(self) -> bool:
        """将尚未保存的记录按日期追加到 ``YYYY-MM-DD.jsonl``。"""
        if self._filePath is None:
            return False
        with self._lock:
            unsaved = [record for record in self._buffer
                       if int(record.get("_sequence", 0)) > self._saved_sequence]
            if not unsaved:
                return True
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in unsaved:
                day = str(record.get("time", ""))[:10] or datetime.now().strftime("%Y-%m-%d")
                groups[day].append(self._public_record(record) or {})
            self._filePath.mkdir(parents=True, exist_ok=True)
            for day, records in groups.items():
                with (self._filePath / f"{day}.jsonl").open("a", encoding="utf-8") as file:
                    for record in records:
                        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._saved_sequence = int(unsaved[-1]["_sequence"])
            return True

    def readFile(self, days: int | None = 7) -> bool:
        """读取最近若干天的日志；``days=None`` 时读取目录中全部 JSONL。"""
        if self._filePath is None or not self._filePath.is_dir():
            return False
        if days is not None and days < 0:
            raise ValueError("days 不能小于 0")
        if days is None:
            files = sorted(self._filePath.glob("*.jsonl"))
        else:
            today = datetime.now()
            files = [self._filePath / f"{(today - timedelta(days=offset)).strftime('%Y-%m-%d')}.jsonl"
                     for offset in range(days, -1, -1)]
        loaded = 0
        with self._lock:
            seen = set(self._index_map)
            for filename in files:
                if not filename.is_file():
                    continue
                with filename.open("r", encoding="utf-8") as file:
                    for line in file:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        record_id = str(record.get("id", ""))
                        if not record_id or record_id in seen:
                            continue
                        record["_sequence"] = self._next_sequence
                        self._next_sequence += 1
                        evicted = self._buffer[0] if len(self._buffer) == self._buffer.maxlen else None
                        self._buffer.append(record)
                        if evicted is not None:
                            self._index_map.pop(str(evicted.get("id", "")), None)
                        self._index_map[record_id] = record
                        seen.add(record_id)
                        loaded += 1
            if self._buffer:
                last_sequence = int(self._buffer[-1].get("_sequence", 0))
                self._saved_sequence = last_sequence
                self._printed_sequence = last_sequence
        return loaded > 0

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._index_map.clear()
            self._printed_sequence = self._next_sequence - 1
            self._saved_sequence = self._next_sequence - 1

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def buffer(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_record(record) or {} for record in self._buffer]

    # 新代码可使用 snake_case；保留 camelCase 兼容参考项目调用方式。
    get_new = getNew
    save_to_file = save2File
    read_file = readFile


RecordBuffer = recordBuffer

__all__ = ["recordBuffer", "RecordBuffer"]
