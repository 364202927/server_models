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
    from ai.utils.common import kApi, save_logs
    from ai.msgHandler import MsgHandler
else:
    from .loader import ModelsMgr
    from .utils.paths import MODELS_FILE
    from .utils.serverApi import serverApi
    from .utils.console import Console
    from .utils.common import kApi, save_logs
    from .msgHandler import MsgHandler


manager = ModelsMgr(str(MODELS_FILE))
handler = MsgHandler(manager)
server = serverApi(manager, handler)
console = Console(command_handler=lambda message_id, data: handler.handle(
    message_id, data, source="console"))
app = server.app


if __name__ == "__main__":
    async def _run() -> None:
        tasks = []
        if sys.stdin.isatty():
            tasks.append(asyncio.create_task(console.run(), name="console"))
        if kApi:
            tasks.append(asyncio.create_task(server.run(), name="fastapi"))
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled() and task.exception():
                raise task.exception()
        save_logs()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
