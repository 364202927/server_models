from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from .loader import models_mgr
from .msgHandler import msgHandler
from .utils.common import info, kApi, save_logs, set_console_active
from .utils.console import Console
from .utils.gateway import gateway
from .utils.hardware import hardware_summary


REAP_INTERVAL_SEC = 30
async def _reaper() -> None:
    """空闲模型回收；与入口方式无关，Console-only 模式同样需要。"""
    while True:
        await asyncio.sleep(REAP_INTERVAL_SEC)
        try:
            reaped = await asyncio.to_thread(models_mgr.reap_idle)
        except Exception as exc:
            info("空闲回收异常", type(exc).__name__, exc)
            continue
        if reaped:
            info("空闲回收", reaped)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="server_models.main", description="模型服务入口")
    parser.add_argument("--client", action="store_true", help="以客户端辅助模式运行（关闭控制台输入与自动启动）")
    return parser.parse_args(argv)

# 只能使用-m加载
if __name__ == "__main__":
    args = parse_args()
    async def _run() -> None:
        tasks = [asyncio.create_task(_reaper(), name="reaper")]
        # 非客户端模式且在交互终端下，才启动终端交互输入
        if not args.client and sys.stdin.isatty():
            tasks.append(asyncio.create_task(console.run(), name="console"))
        if kApi:
            tasks.append(asyncio.create_task(server.run(), name="fastapi"))
        if not tasks:
            return
        # 客户端模式下不主动拉起模型，等待前端指令触发
        launch = None
        if not args.client:
            launch = asyncio.create_task(asyncio.to_thread(models_mgr.autolaunch), name="autolaunch")
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            await asyncio.shield(asyncio.to_thread(models_mgr.unload_all))
            if launch is not None:
                await asyncio.gather(launch, return_exceptions=True)

        for task in done:
            if not task.cancelled():
                exc = task.exception()
                if exc and not isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                    raise exc
    #
    print("硬件信息\n" + hardware_summary())
    set_console_active(True)
    handler = msgHandler()
    server = gateway(handler)
    console = Console(command_handler=lambda message_id, msg_args: handler.handle(message_id, msg_args, source="console"))
    app = server.app
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        set_console_active(False)
        save_logs()