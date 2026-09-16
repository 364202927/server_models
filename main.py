"""服务启动入口。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
if __package__ in (None, ""):
    # 直接执行 ``python main.py`` 时把项目父目录加入导入路径。
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ai.loader import ModelsMgr
    from ai.utils.paths import MODELS_FILE
    from ai.utils.serverApi import serverApi
    from ai.utils.console import Console
    from ai.utils.common import kApi, info, save_logs
    from ai.msgHandler import MsgHandler
else:
    from .loader import ModelsMgr
    from .utils.paths import MODELS_FILE
    from .utils.serverApi import serverApi
    from .utils.console import Console
    from .utils.common import kApi, info, save_logs
    from .msgHandler import MsgHandler


REAP_INTERVAL_SEC = 30

manager = ModelsMgr(str(MODELS_FILE))
handler = MsgHandler(manager)
server = serverApi(manager, handler)
console = Console(command_handler=lambda message_id, args: handler.handle(
    message_id, args, source="console"))
app = server.app


async def _reaper() -> None:
    """空闲模型回收；与入口方式无关，Console-only 模式同样需要。"""
    while True:
        await asyncio.sleep(REAP_INTERVAL_SEC)
        try:
            reaped = await asyncio.to_thread(manager.reap_idle)
        except Exception as exc:
            info("空闲回收异常", type(exc).__name__, exc)
            continue
        if reaped:
            info("空闲回收", reaped)


if __name__ == "__main__":
    async def _run() -> None:
        # 恢复上次运行时驻留显存的模型；失败只记日志，不阻塞启动。
        await asyncio.to_thread(manager.restore_from_snapshots)

        tasks = [asyncio.create_task(_reaper(), name="reaper")]
        if sys.stdin.isatty():
            tasks.append(asyncio.create_task(console.run(), name="console"))
        if kApi:
            tasks.append(asyncio.create_task(server.run(), name="fastapi"))
        if len(tasks) == 1:
            # 只有 reaper 时没有任何对外入口，直接退出。
            tasks[0].cancel()
            return
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled() and task.exception():
                raise task.exception()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C 也要落盘，否则缓冲区里的日志直接丢失。
        save_logs()
