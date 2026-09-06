"""
CLI入口 - AI模型量化交易能力评估工具

使用方式:
  python -m ai.cli evaluate --model /path/to/model --engine vllm
  python -m ai.cli hardware
"""

import argparse
import sys
from pathlib import Path

from .config import BenchmarkConfig
from .hardware import print_hardware_info
from .loader import HFLoader, VLLMLoader, ModelLoader
from .benchmark import BenchmarkRunner, ReportGenerator


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="AI Model Quantitative Trading Capability Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用vLLM评估模型
  python -m ai.cli evaluate --model /path/to/model --engine vllm

  # 使用Transformers评估（适合测试）
  python -m ai.cli evaluate --model /path/to/model --engine hf

  # 仅显示硬件信息
  python -m ai.cli hardware

  # 指定量化方式
  python -m ai.cli evaluate --model /path/to/model --engine vllm --quantization w8a8
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # evaluate 命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument(
        "--model", "-m",
        required=True,
        help="模型路径或HuggingFace模型ID"
    )
    eval_parser.add_argument(
        "--engine", "-e",
        choices=["vllm", "hf"],
        default="hf",
        help="推理引擎: vllm (高性能) 或 hf (Transformers)"
    )
    eval_parser.add_argument(
        "--quantization", "-q",
        choices=["w8a8", "awq", "gptq", "4bit", "8bit", None],
        default=None,
        help="量化方式"
    )
    eval_parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="数据类型"
    )
    eval_parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="最大上下文长度"
    )
    eval_parser.add_argument(
        "--tensor-parallel", "-tp",
        type=int,
        default=1,
        help="张量并行数（多GPU）"
    )
    eval_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU显存利用率 (0-1)"
    )
    eval_parser.add_argument(
        "--test-cases",
        type=Path,
        default=None,
        help="自定义测试用例JSON文件路径"
    )
    eval_parser.add_argument(
        "--no-color",
        action="store_true",
        help="禁用彩色输出"
    )
    eval_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )

    # hardware 命令
    hw_parser = subparsers.add_parser("hardware", help="显示硬件信息")

    return parser


def run_evaluate(args) -> int:
    """执行模型评估流程"""
    # 根据引擎类型选择加载器
    loader: ModelLoader = VLLMLoader() if args.engine == "vllm" else HFLoader()

    # 构建加载参数
    load_kwargs = {
        "quantization": args.quantization,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel,
    }
    if args.max_model_len:
        load_kwargs["max_model_len"] = args.max_model_len
    if args.engine == "vllm":
        load_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    # 输出加载信息
    print(f"\n正在加载模型: {args.model}")
    print(f"推理引擎: {args.engine.upper()}")
    if args.quantization:
        print(f"量化方式: {args.quantization}")
    print()

    # 加载模型
    try:
        loader.load(args.model, **load_kwargs)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return 1

    print(f"模型加载成功: {loader.model_info.name}")
    print(f"参数量: {loader.model_info.parameters}\n")

    # 执行评估
    config = BenchmarkConfig(verbose=args.verbose)
    runner = BenchmarkRunner(config, args.test_cases)
    reporter = ReportGenerator(use_color=not args.no_color)
    runner.set_progress_callback(reporter.print_progress)

    print("开始评估...\n")

    try:
        result = runner.run(loader)
        print(reporter.generate(result))
    except Exception as e:
        print(f"评估过程出错: {e}")
        return 1
    finally:
        loader.unload()

    return 0 if result.passed else 1


def run_hardware() -> int:
    """显示硬件信息"""
    print_hardware_info()
    return 0


def main() -> int:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    match args.command:
        case "evaluate":
            return run_evaluate(args)
        case "hardware":
            return run_hardware()
        case _:
            parser.print_help()
            return 0


if __name__ == "__main__":
    sys.exit(main())
