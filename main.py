from __future__ import annotations

import asyncio,sys
from pathlib import Path
if __package__ in (None, ""):
    # 直接执行 ``python main.py`` 时把项目父目录加入导入路径。
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from server_models.loader import modelsMgr
    from server_models.utils import MODELS_FILE
    from server_models.utils.serverApi import serverApi
    from server_models.utils.console import Console
    from server_models.utils.common import kApi, info, save_logs
    from server_models.utils.hardware import hardware_summary
    from server_models.msgHandler import msgHandler
else:
    from .loader import modelsMgr
    from .utils import MODELS_FILE
    from .utils.serverApi import serverApi
    from .utils.console import Console
    from .utils.common import kApi, info, save_logs
    from .utils.hardware import hardware_summary
    from .msgHandler import msgHandler


REAP_INTERVAL_SEC = 30
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
        # 启动顺序:1.初始化并打印硬件 2.按 launch 加载模型
        print("硬件信息\n" + hardware_summary())
        tasks = [asyncio.create_task(_reaper(), name="reaper")]
        if sys.stdin.isatty():
            tasks.append(asyncio.create_task(console.run(), name="console"))
        if kApi:
            tasks.append(asyncio.create_task(server.run(), name="fastapi"))
        if len(tasks) == 1:
            tasks[0].cancel()
            return

        # 加载模型耗时且可能卡住：放后台线程，不能挡住 console/fastapi 启动，也不进 wait 集合
        launch = asyncio.create_task(asyncio.to_thread(manager.autolaunch), name="autolaunch")
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            # 先杀引擎子进程：加载线程的健康检查随之返回，线程池才能正常收尾；否则退出会死等、显存不释放
            await asyncio.shield(asyncio.to_thread(manager.unload_all))
            await asyncio.gather(launch, return_exceptions=True)

        for task in done:
            if not task.cancelled():
                exc = task.exception()
                if exc and not isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                    raise exc
    #
    manager = modelsMgr(str(MODELS_FILE))
    handler = msgHandler(manager)
    server = serverApi(manager, handler)
    console = Console(command_handler=lambda message_id, args: handler.handle(message_id, args, source="console"))
    app = server.app
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        save_logs()#记录log
