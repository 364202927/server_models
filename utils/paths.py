"""项目资源目录路径。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_FILE = ASSETS_DIR / "models.json"
RUNTIME_DIR = ASSETS_DIR / "runtime"
CACHE_DIR = ASSETS_DIR / "cache"
CHAT_HISTORY_DIR = ASSETS_DIR / "chat_history"


def ensure_asset_dirs() -> None:
    for path in (RUNTIME_DIR, CACHE_DIR, CHAT_HISTORY_DIR):
        path.mkdir(parents=True, exist_ok=True)
