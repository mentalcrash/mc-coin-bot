"""ChartGenerator — headless matplotlib 차트 생성기.

matplotlib Agg 백엔드로 equity curve, drawdown, 월간 수익률 히트맵,
거래 PnL 분포 차트를 BytesIO(PNG bytes)로 생성합니다.

Rules Applied:
    - matplotlib.use("agg") 최상단 호출 (headless)
    - plt.close(fig) 필수 (메모리 누수 방지)
    - 디스크 I/O 없음 (BytesIO 출력)
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.models.backtest import PerformanceMetrics, TradeRecord

# 차트 스타일 상수
_FIGSIZE_WIDE = (12, 5)
_FIGSIZE_SQUARE = (10, 6)
_DPI = 150
_COLOR_EQUITY = "#2196F3"
_COLOR_DD_FILL = "#EF5350"
_COLOR_DD_LINE = "#C62828"
_COLOR_PNL_WIN = "#4CAF50"
_COLOR_PNL_LOSS = "#F44336"
_COLOR_MEAN_LINE = "#FF9800"
_HEATMAP_CMAP = "RdYlGn"
_PNG_FORMAT = "png"
_PNG_MAGIC = b"\x89PNG"
_MIN_HEATMAP_POINTS = 2


def _fig_to_bytes(fig: Any) -> bytes:
    """Figure → PNG bytes, 자원 해제."""
    buf = io.BytesIO()
    fig.savefig(buf, format=_PNG_FORMAT, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


class ChartGenerator:
    """Headless matplotlib 차트 생성기 (Agg backend)."""

    def generate_equity_curve(self, equity_series: pd.Series) -> bytes:
        """Equity curve PNG → bytes.

        Args:
            equity_series: DatetimeIndex equity 시리즈

        Returns:
            PNG bytes (빈 데이터면 빈 bytes)
        """
        if len(equity_series) == 0:
            return b""

        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
        values = equity_series.to_numpy()
        ax.plot(equity_series.index, values, color=_COLOR_EQUITY, linewidth=1.5)
        ax.fill_between(
            equity_series.index,
            values,
            values[0],
            alpha=0.15,
            color=_COLOR_EQUITY,
        )
        ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (USD)")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        return _fig_to_bytes(fig)

    def generate_drawdown(self, equity_series: pd.Series) -> bytes:
        """Drawdown curve PNG → bytes.

        Args:
            equity_series: DatetimeIndex equity 시리즈

        Returns:
            PNG bytes (빈 데이터면 빈 bytes)
        """
        if len(equity_series) == 0:
            return b""

        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100

        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
        dd_values = drawdown.to_numpy()
        ax.fill_between(drawdown.index, dd_values, 0, color=_COLOR_DD_FILL, alpha=0.4)
        ax.plot(drawdown.index, dd_values, color=_COLOR_DD_LINE, linewidth=1.0)
        ax.set_title("Drawdown", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        return _fig_to_bytes(fig)

    def generate_monthly_heatmap(self, equity_series: pd.Series) -> bytes:
        """월간 수익률 히트맵 PNG → bytes.

        Args:
            equity_series: DatetimeIndex equity 시리즈

        Returns:
            PNG bytes (빈 데이터면 빈 bytes)
        """
        import pandas as pd

        if len(equity_series) < _MIN_HEATMAP_POINTS:
            return b""

        # 월간 수익률 계산
        monthly = equity_series.resample("ME").last().pct_change().dropna() * 100

        if len(monthly) == 0:
            return b""

        # year x month pivot
        years = monthly.index.year
        months = monthly.index.month
        df = pd.DataFrame({"year": years, "month": months, "return": monthly.to_numpy()})
        pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
        pivot.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][: len(pivot.columns)]

        fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
        data_arr = pivot.to_numpy(dtype=float)
        abs_max = max(abs(np.nanmin(data_arr)), abs(np.nanmax(data_arr)), 1.0)
        im = ax.imshow(data_arr, cmap=_HEATMAP_CMAP, aspect="auto", vmin=-abs_max, vmax=abs_max)

        # 축 레이블
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # 셀 값 표시
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = data_arr[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, label="Return (%)")
        ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
        return _fig_to_bytes(fig)

    def generate_trade_pnl_distribution(self, trades: list[TradeRecord]) -> bytes:
        """거래 PnL 분포 히스토그램 PNG → bytes.

        Args:
            trades: 종결된 거래 목록

        Returns:
            PNG bytes (빈 데이터면 빈 bytes)
        """
        pnls = [float(t.pnl) for t in trades if t.pnl is not None]
        if not pnls:
            return b""

        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

        n_bins = min(30, max(10, len(pnls) // 3))
        ax.hist(pnls, bins=n_bins, color=_COLOR_EQUITY, alpha=0.7, edgecolor="white")

        mean_pnl = float(np.mean(pnls))
        ax.axvline(
            mean_pnl,
            color=_COLOR_MEAN_LINE,
            linestyle="--",
            linewidth=2,
            label=f"Mean: ${mean_pnl:,.2f}",
        )
        ax.axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        ax.set_title("Trade PnL Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("PnL (USD)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
        return _fig_to_bytes(fig)

    def generate_daily_report(
        self,
        equity_series: pd.Series,
        trades: list[TradeRecord],
        metrics: PerformanceMetrics,
    ) -> list[tuple[str, bytes]]:
        """일일 리포트 차트 세트.

        Args:
            equity_series: equity curve
            trades: 거래 목록
            metrics: 성과 지표

        Returns:
            [(filename, bytes), ...]
        """
        result: list[tuple[str, bytes]] = []

        equity_png = self.generate_equity_curve(equity_series)
        if equity_png:
            result.append(("equity_curve.png", equity_png))

        dd_png = self.generate_drawdown(equity_series)
        if dd_png:
            result.append(("drawdown.png", dd_png))

        heatmap_png = self.generate_monthly_heatmap(equity_series)
        if heatmap_png:
            result.append(("monthly_heatmap.png", heatmap_png))

        pnl_png = self.generate_trade_pnl_distribution(trades)
        if pnl_png:
            result.append(("trade_pnl_dist.png", pnl_png))

        return result

    def generate_weekly_report(
        self,
        equity_series: pd.Series,
        trades: list[TradeRecord],
        metrics: PerformanceMetrics,
    ) -> list[tuple[str, bytes]]:
        """주간 리포트 차트 세트.

        Args:
            equity_series: equity curve
            trades: 거래 목록
            metrics: 성과 지표

        Returns:
            [(filename, bytes), ...]
        """
        result: list[tuple[str, bytes]] = []

        equity_png = self.generate_equity_curve(equity_series)
        if equity_png:
            result.append(("equity_curve.png", equity_png))

        dd_png = self.generate_drawdown(equity_series)
        if dd_png:
            result.append(("drawdown.png", dd_png))

        heatmap_png = self.generate_monthly_heatmap(equity_series)
        if heatmap_png:
            result.append(("monthly_heatmap.png", heatmap_png))

        pnl_png = self.generate_trade_pnl_distribution(trades)
        if pnl_png:
            result.append(("trade_pnl_dist.png", pnl_png))

        return result
