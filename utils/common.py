import pickle,gzip,json,os,inspect,struct,importlib.util
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
kApi = os.getenv("AI_KAPI", "True").strip().lower() not in {"0", "false", "no", "off"}


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

#显存转换成token
def vm2tokens(engine: str, path: str, context: float | int, dtype: str, tensor_parallel: int = 1, align_step: int = 512) -> int:
    if float(context) <= 1000:
        return 2048

    # KV cache 每元素字节数 (llama.cpp 量化类型带 block scale, 不是整数字节)
    kv_bytes = {
        "float32": 4.0, "fp32": 4.0, "f32": 4.0,
        "bfloat16": 2.0, "bf16": 2.0, "float16": 2.0, "fp16": 2.0, "f16": 2.0,
        "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "int8": 1.0,
        "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32,
    }
    # llama.cpp 支持的 KV cache 类型, 其余(如 fp8)按 f16 处理
    llama_types = {"f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0",
                   "float32", "float16", "bfloat16", "fp16", "fp32"}
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def read_meta_py(gguf_path: Path) -> dict:
        meta = {}
        with open(gguf_path, "rb") as f:
            def unpack(code):
                return struct.unpack("<" + code, f.read(struct.calcsize(code)))[0]
            def read_str():
                return f.read(unpack("Q")).decode("utf-8", errors="ignore")
            def skip_items(item_type, count):
                if item_type in fmt:
                    f.seek(struct.calcsize(fmt[item_type]) * count, 1)
                    return
                for _ in range(count):
                    skip(item_type)
            def skip(t):
                if t == 8:
                    f.seek(unpack("Q"), 1)
                elif t == 9:
                    item_type = unpack("I")
                    skip_items(item_type, unpack("Q"))
                else:
                    f.seek(struct.calcsize(fmt[t]), 1)
            def read(t):
                if t == 8:
                    return read_str()
                if t != 9:
                    return unpack(fmt[t])
                item_type = unpack("I")
                count = unpack("Q")
                if item_type in fmt and count <= 4096:  # 逐层数组 (如 head_count_kv)
                    return [unpack(fmt[item_type]) for _ in range(count)]
                skip_items(item_type, count)
                return None
            if f.read(4) != b"GGUF":
                raise ValueError(f"不是合法的 GGUF 文件: {gguf_path}")
            if unpack("I") < 2:
                raise ValueError(f"不支持 GGUF v1: {gguf_path}")
            unpack("Q")  # tensor_count
            for _ in range(unpack("Q")):
                key, t = read_str(), unpack("I")
                if key.startswith("tokenizer."):
                    skip(t)
                    continue
                value = read(t)
                if value is not None:
                    meta[key] = value
        return meta

    def read_meta(gguf_path: Path) -> dict:
        if importlib.util.find_spec("gguf") is None:
            return read_meta_py(gguf_path)
        from gguf import GGUFReader
        fields = GGUFReader(str(gguf_path)).fields
        return {k: v.contents() for k, v in fields.items() if not k.startswith("tokenizer.")}
    def per_layer(value, n: int) -> list[int]:
        if not isinstance(value, (list, tuple)):
            return [int(value)] * n
        values = [int(x) for x in value] or [0]
        return (values + [values[-1]] * n)[:n]
    def spec_gguf(meta: dict) -> tuple[list[int], int, int, int]:
        arch = meta.get("general.architecture")
        if not arch:
            raise ValueError("GGUF 缺少 general.architecture")
        def g(key, default=None):
            v = meta.get(f"{arch}.{key}")
            return default if v is None else v
        n_layer = int(g("block_count") or 0)
        n_head_raw = g("attention.head_count")
        if not n_layer or not n_head_raw:
            raise ValueError(f"GGUF 缺少 block_count / attention.head_count (arch={arch})")
        n_kv_layer = n_layer - int(g("nextn_predict_layers", 0))  # MTP 层不分配 KV
        n_head = per_layer(n_head_raw, n_layer)
        n_head_kv = per_layer(g("attention.head_count_kv", n_head_raw), n_layer)
        interval = int(g("full_attention_interval", 0))          # 混合架构: 每 N 层一个全注意力层
        kv_heads = [
            n_head_kv[i] for i in range(n_kv_layer)
            if n_head_kv[i] > 0 and (interval <= 0 or (i + 1) % interval == 0)
        ]
        k_dim = int(g("attention.key_length") or int(g("embedding_length")) // max(n_head))
        v_dim = int(g("attention.value_length") or k_dim)

        # 线性注意力/循环层的状态: 按 llama.cpp 的 n_embd_r + n_embd_s 估算, f32, 单序列
        fixed = 0
        d_inner, d_state, d_conv = g("ssm.inner_size"), g("ssm.state_size"), g("ssm.conv_kernel")
        if d_inner and d_state and d_conv:
            n_group = int(g("ssm.group_count", 1))
            state = (d_conv - 1) * (d_inner + 2 * n_group * d_state) + d_state * d_inner
            fixed = (n_kv_layer - len(kv_heads)) * int(state) * 4
        return kv_heads, k_dim, v_dim, fixed

    def spec_hf(cfg: dict) -> tuple[list[int], int, int, int]:
        cfg = cfg.get("text_config") or cfg  # 多模态模型的文本部分
        n_layer, n_head, hidden = cfg.get("num_hidden_layers"), cfg.get("num_attention_heads"), cfg.get("hidden_size")
        if not (n_layer and n_head):
            raise ValueError("config.json 缺少 num_hidden_layers / num_attention_heads")
        layer_types, interval = cfg.get("layer_types"), cfg.get("full_attention_interval")
        if layer_types:  # 排除 linear_attention
            n_attn = sum(1 for t in layer_types if "attention" in t and "linear" not in t)
        elif interval:
            n_attn = n_layer // interval
        else:
            n_attn = n_layer
        head_dim = cfg.get("head_dim") or hidden // n_head
        return [cfg.get("num_key_value_heads") or n_head] * n_attn, head_dim, head_dim, 0

    # ---- 解析模型结构 ----
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"模型路径不存在: {path}")
    if p.is_file() and p.suffix.lower() == ".gguf":
        ggufs = [p]
    else:
        ggufs = sorted(p.glob("*.gguf")) if p.is_dir() else []  # 分片 GGUF 的首片含完整元数据

    if ggufs:
        kv_heads, k_dim, v_dim, fixed = spec_gguf(read_meta(ggufs[0]))
    else:
        cfg_file = (p if p.is_dir() else p.parent) / "config.json"
        if not cfg_file.exists():
            raise ValueError(f"无法解析该模型架构参数: {path}")
        kv_heads, k_dim, v_dim, fixed = spec_hf(json.loads(cfg_file.read_text(encoding="utf-8")))

    # ---- 显存 -> token ----
    tp = max(1, int(tensor_parallel))
    dtype_key = str(dtype).lower().strip()
    if str(engine).lower().startswith("llama") and dtype_key not in llama_types:
        dtype_key = "f16"  # llama.cpp 默认 f16 KV

    heads_per_gpu = sum(max(1, -(-h // tp)) for h in kv_heads)  # kv_heads < tp 时每卡复制
    bytes_per_token = heads_per_gpu * (k_dim + v_dim) * kv_bytes.get(dtype_key, 2.0)
    if bytes_per_token <= 0:
        raise ValueError(f"模型没有带 KV cache 的注意力层: {path}")

    budget = float(context) * 1024 * 1024 - fixed
    tokens = max(0, int(budget // bytes_per_token)) // align_step * align_step
    print(f"vm2tokens engine={engine} dtype={dtype_key} tp={tp} attn_layers={len(kv_heads)} "
      f"kv_heads={sorted(set(kv_heads))} k/v_dim={k_dim}/{v_dim} "
      f"B/token={bytes_per_token:.0f} fixed={fixed / 1024 / 1024:.1f}MB -> {tokens}")
    return max(align_step, tokens)