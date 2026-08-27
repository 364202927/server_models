"""
CLI报告生成器 - 生成美观的命令行评估报告
"""

from ..config import DIMENSION_NAMES
from .runner import BenchmarkResult
from .scorer import Scorer


class ReportGenerator:
    """CLI报告生成器"""

    # ANSI颜色代码
    COLORS = {
        "reset": "\033[0m", "bold": "\033[1m",
        "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
        "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m",
    }

    def __init__(self, use_color: bool = True):
        self.use_color = use_color
        self.scorer = Scorer()

    def _c(self, text: str, color: str) -> str:
        """应用颜色"""
        if not self.use_color:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _progress_bar(self, score: float, width: int = 10) -> str:
        """生成彩色进度条"""
        filled = int(score / 100 * width)
        bar_color = "green" if score >= 80 else ("yellow" if score >= 60 else "red")
        return self._c("█" * filled, bar_color) + "░" * (width - filled)

    def generate(self, result: BenchmarkResult) -> str:
        """生成完整报告"""
        lines = []

        # 标题
        lines.append("")
        lines.append(self._c("╔" + "═" * 60 + "╗", "cyan"))
        lines.append(self._c("║", "cyan") + self._c("  AI Model Quant-Trading Capability Assessment", "bold").center(68) + self._c("║", "cyan"))
        lines.append(self._c("║", "cyan") + "  AI模型量化交易能力评估".center(52) + self._c("║", "cyan"))
        lines.append(self._c("╠" + "═" * 60 + "╣", "cyan"))

        # 硬件信息
        hw = result.hardware_info
        if hw and hw.gpus:
            gpu = hw.gpus[0]
            gpu_info = f"{gpu.name} ({gpu.memory_total_mb}MB)"
        else:
            gpu_info = "No GPU detected"

        lines.append(self._c("║", "cyan") + f"  Hardware: {gpu_info[:45]}".ljust(60) + self._c("║", "cyan"))

        if hw:
            lines.append(self._c("║", "cyan") + f"  CPU: {hw.cpu.name[:48]}".ljust(60) + self._c("║", "cyan"))
            lines.append(self._c("║", "cyan") + f"  Memory: {hw.memory.total_gb}GB | Platform: {hw.platform[:20]}".ljust(60) + self._c("║", "cyan"))

        # 模型信息
        lines.append(self._c("╠" + "═" * 60 + "╣", "cyan"))
        mi = result.model_info
        lines.append(self._c("║", "cyan") + f"  Model: {mi.name[:50]}".ljust(60) + self._c("║", "cyan"))
        lines.append(self._c("║", "cyan") + f"  Type: {mi.model_type} | Params: {mi.parameters} | Quant: {mi.quantization or 'None'}".ljust(60) + self._c("║", "cyan"))

        # 测试结果
        lines.append(self._c("╠" + "═" * 60 + "╣", "cyan"))
        lines.append(self._c("║", "cyan") + self._c("  TEST RESULTS", "bold").ljust(68) + self._c("║", "cyan"))
        lines.append(self._c("║", "cyan") + "  " + "─" * 56 + "  " + self._c("║", "cyan"))

        breakdowns = self.scorer.get_breakdown(result)
        for i, bd in enumerate(breakdowns, 1):
            bar = self._progress_bar(bd.raw_score)
            score_str = f"{bd.raw_score:5.1f}/100"

            if bd.raw_score >= 80:
                score_color = "green"
            elif bd.raw_score >= 60:
                score_color = "yellow"
            else:
                score_color = "red"

            # 维度名称（左对齐，固定宽度）
            dim_name = f"{i}. {bd.dimension_name}"
            line = f"  {dim_name:<20} {bar}  {self._c(score_str, score_color)}"
            lines.append(self._c("║", "cyan") + line.ljust(68 if not self.use_color else 77) + self._c("║", "cyan"))

        # 总分
        lines.append(self._c("║", "cyan") + "  " + "─" * 56 + "  " + self._c("║", "cyan"))

        total_bar = self._progress_bar(result.total_score)
        grade = self.scorer.calculate_grade(result.total_score)

        if result.total_score >= 80:
            total_color = "green"
        elif result.total_score >= 60:
            total_color = "yellow"
        else:
            total_color = "red"

        total_str = f"{result.total_score:5.1f}/100"
        total_line = f"  {'WEIGHTED TOTAL':<20} {total_bar}  {self._c(total_str, total_color)}  [{grade}]"
        lines.append(self._c("║", "cyan") + total_line.ljust(68 if not self.use_color else 82) + self._c("║", "cyan"))

        # 结论
        lines.append(self._c("╠" + "═" * 60 + "╣", "cyan"))

        if result.passed:
            verdict_icon = self._c("✓", "green")
            verdict_text = "SUITABLE for quantitative trading"
        else:
            verdict_icon = self._c("✗", "red")
            verdict_text = "NOT RECOMMENDED for quantitative trading"

        lines.append(self._c("║", "cyan") + f"  VERDICT: {verdict_icon} {verdict_text}".ljust(68 if not self.use_color else 77) + self._c("║", "cyan"))
        lines.append(self._c("║", "cyan") + f"  {result.verdict[:56]}".ljust(60) + self._c("║", "cyan"))

        lines.append(self._c("╚" + "═" * 60 + "╝", "cyan"))

        # 建议
        recommendation = self.scorer.get_recommendation(result)
        lines.append("")
        lines.append(self._c("建议: ", "bold") + recommendation)

        # 优劣势
        strengths, weaknesses = self.scorer.get_strengths_weaknesses(result)
        if strengths:
            lines.append(self._c("\n优势: ", "green") + ", ".join(strengths))
        if weaknesses:
            lines.append(self._c("待改进: ", "yellow") + ", ".join(weaknesses))

        lines.append("")

        return "\n".join(lines)

    def generate_simple(self, result: BenchmarkResult) -> str:
        """生成简单报告（无颜色）"""
        old_color = self.use_color
        self.use_color = False
        report = self.generate(result)
        self.use_color = old_color
        return report

    def print_progress(self, dimension: str, current: int, total: int) -> None:
        """打印进度"""
        dim_name = DIMENSION_NAMES.get(dimension, dimension)
        bar_width = 20
        filled = int(current / total * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"\r[{bar}] {current}/{total} - 正在测试: {dim_name}...", end="", flush=True)

        if current == total:
            print()  # 换行
