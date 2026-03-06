"""Tri-Channel Multi-Scale Trend — 20개 주요 에셋 VBT 백테스트.

Usage:
    uv run python scripts/tri_channel_20assets.py
"""

from datetime import UTC, datetime
from decimal import Decimal

from rich.console import Console
from rich.table import Table

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.config.settings import get_settings
from src.core.exceptions import DataNotFoundError
from src.core.logger import setup_logger
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy

# ── Config ────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "POL/USDT",
    "ATOM/USDT",
    "UNI/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "FTM/USDT",
    "FIL/USDT",
    "LTC/USDT",
]

TIMEFRAME = "12H"
START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2026, 3, 1, tzinfo=UTC)
CAPITAL = Decimal(100000)

STRATEGY_NAME = "tri-channel-trend"
STRATEGY_PARAMS = {
    "scale_short": 20,
    "scale_mid": 60,
    "scale_long": 150,
    "bb_std_dev": 2.0,
    "keltner_multiplier": 1.5,
    "entry_threshold": 0.22,
    "vol_window": 30,
    "vol_target": 0.35,
    "min_volatility": 0.05,
    "annualization_factor": 730.0,
    "short_mode": 0,  # DISABLED (Long-only)
    "hedge_threshold": -0.07,
    "hedge_strength_ratio": 0.8,
}

PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=2.0,
    rebalance_threshold=0.10,
    system_stop_loss=0.10,
    use_trailing_stop=True,
    trailing_stop_atr_multiplier=3.0,
    use_intrabar_trailing_stop=False,
)


def main() -> None:
    """20개 에셋 Tri-Channel 백테스트 실행."""
    setup_logger(console_level="WARNING")
    console = Console()

    console.print(
        f"\n[bold cyan]Tri-Channel Multi-Scale Trend — 20 Assets VBT Backtest[/bold cyan]"
        f"\n  Timeframe: {TIMEFRAME} | Period: {START.date()} ~ {END.date()}"
        f"\n  Capital: ${CAPITAL:,.0f} | Short Mode: DISABLED\n"
    )

    # Strategy
    strategy_cls = get_strategy(STRATEGY_NAME)
    strategy = strategy_cls.from_params(**STRATEGY_PARAMS)

    # Warmup
    warmup_bars = 0
    if strategy.config and hasattr(strategy.config, "warmup_periods"):
        warmup_bars = strategy.config.warmup_periods()

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()

    results: list[dict] = []

    for i, symbol in enumerate(SYMBOLS, 1):
        console.print(f"  [{i:2d}/{len(SYMBOLS)}] {symbol:12s} ... ", end="")
        try:
            # Data load (with warmup buffer)
            from datetime import timedelta

            from src.eda.candle_aggregator import timeframe_to_seconds

            tf_seconds = timeframe_to_seconds(TIMEFRAME)
            data_start = (
                START - timedelta(seconds=tf_seconds * warmup_bars) if warmup_bars > 0 else START
            )

            data = data_service.get(
                MarketDataRequest(
                    symbol=symbol,
                    timeframe=TIMEFRAME,
                    start=data_start,
                    end=END,
                )
            )

            portfolio = Portfolio.create(initial_capital=CAPITAL, config=PM_CONFIG)

            request = BacktestRequest(
                data=data,
                strategy=strategy,
                portfolio=portfolio,
                warmup_bars=warmup_bars,
            )

            result = engine.run(request)
            m = result.metrics

            results.append(
                {
                    "symbol": symbol,
                    "return": m.total_return,
                    "cagr": m.cagr,
                    "sharpe": m.sharpe_ratio,
                    "sortino": m.sortino_ratio,
                    "max_dd": m.max_drawdown,
                    "calmar": m.calmar_ratio,
                    "win_rate": m.win_rate,
                    "profit_factor": m.profit_factor,
                    "trades": m.total_trades,
                    "status": "OK",
                }
            )
            console.print(
                f"[green]Sharpe {m.sharpe_ratio:.2f} | Return {m.total_return:+.1f}%[/green]"
            )

        except (DataNotFoundError, Exception) as e:
            results.append(
                {
                    "symbol": symbol,
                    "return": 0.0,
                    "cagr": 0.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "max_dd": 0.0,
                    "calmar": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "trades": 0,
                    "status": f"FAIL: {e!s:.50s}",
                }
            )
            console.print(f"[red]FAIL: {e!s:.80s}[/red]")

    # ── Results Table ───────────────────────────────────────────────
    console.print("\n")
    table = Table(title="Tri-Channel Trend — 20 Assets (12H, 2024-01 ~ 2026-03)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Return %", justify="right")
    table.add_column("CAGR %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Max DD %", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Status", justify="center")

    for i, r in enumerate(sorted(results, key=lambda x: x["sharpe"], reverse=True), 1):
        if r["status"] != "OK":
            table.add_row(
                str(i),
                r["symbol"],
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                f"[red]{r['status']}[/red]",
            )
            continue

        # Color coding
        sharpe_color = "green" if r["sharpe"] > 1.0 else "yellow" if r["sharpe"] > 0.5 else "red"
        ret_color = "green" if r["return"] > 0 else "red"
        dd_color = "green" if r["max_dd"] > -20 else "yellow" if r["max_dd"] > -30 else "red"

        table.add_row(
            str(i),
            r["symbol"],
            f"[{ret_color}]{r['return']:+.1f}[/{ret_color}]",
            f"{r['cagr']:+.1f}",
            f"[{sharpe_color}]{r['sharpe']:.2f}[/{sharpe_color}]",
            f"{r['sortino']:.2f}" if r["sortino"] else "N/A",
            f"[{dd_color}]{r['max_dd']:.1f}[/{dd_color}]",
            f"{r['calmar']:.2f}" if r["calmar"] else "N/A",
            f"{r['win_rate']:.1f}",
            f"{r['profit_factor']:.2f}" if r["profit_factor"] else "N/A",
            str(r["trades"]),
            "[green]OK[/green]",
        )

    console.print(table)

    # ── Summary Stats ─────────────────────────────────────────────
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        avg_sharpe = sum(r["sharpe"] for r in ok_results) / len(ok_results)
        avg_return = sum(r["return"] for r in ok_results) / len(ok_results)
        positive = sum(1 for r in ok_results if r["return"] > 0)
        best = max(ok_results, key=lambda x: x["sharpe"])
        worst = min(ok_results, key=lambda x: x["sharpe"])

        console.print(f"\n[bold]Summary ({len(ok_results)}/{len(results)} assets):[/bold]")
        console.print(f"  Avg Sharpe: {avg_sharpe:.2f} | Avg Return: {avg_return:+.1f}%")
        console.print(
            f"  Positive: {positive}/{len(ok_results)} ({positive / len(ok_results) * 100:.0f}%)"
        )
        console.print(
            f"  Best:  {best['symbol']} (Sharpe {best['sharpe']:.2f}, {best['return']:+.1f}%)"
        )
        console.print(
            f"  Worst: {worst['symbol']} (Sharpe {worst['sharpe']:.2f}, {worst['return']:+.1f}%)"
        )


if __name__ == "__main__":
    main()
