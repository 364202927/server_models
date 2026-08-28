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
    from ai.utils.server import ModelServer
else:
    from .loader import ModelsMgr
    from .utils.paths import MODELS_FILE
    from .utils.server import ModelServer


manager = ModelsMgr(str(MODELS_FILE))
server = ModelServer(manager)
app = server.app


if __name__ == "__main__":
    asyncio.run(server.run())
