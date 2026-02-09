"""Leverage Cap / System Stop Loss 파라미터 스윕 — EDA 백테스트.

max_leverage_cap × system_stop_loss 조합을 테스트하여
Gate 7 (Paper Trading) 진입 전 최종 리스크 파라미터를 확정한다.

고정값: trailing_stop_atr_multiplier=4.0, rebalance_threshold=0.15 (이전 스윕에서 최적화 완료)
"""

from __future__ import annotations

import asyncio
import itertools
from datetime import UTC, datetime

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config.config_loader import build_strategy, load_config
from src.config.settings import get_settings
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.eda.runner import EDARunner

console = Console()

# --- 스윕 파라미터 ---
LEVERAGE_CAPS = [1.0, 1.5, 2.0, 2.5, 3.0]
STOP_LOSSES = [0.05, 0.07, 0.10, 0.15, 0.20]

# 고정값 (이전 스윕에서 최적화 완료)
FIXED_TS_MULT = 4.0
FIXED_REBAL = 0.15

# 자산별 설정
ASSETS = [
    ("SOL/USDT", "config/tsmom_sol.yaml"),
    ("DOGE/USDT", "config/tsmom_doge.yaml"),
]

# VBT 기준선 (Gate 5 결과)
VBT_BASELINES: dict[str, dict[str, float]] = {
    "SOL/USDT": {"sharpe": 1.33, "cagr": 49.99, "mdd": 26.00, "trades": 181},
    "DOGE/USDT": {"sharpe": 1.25, "cagr": 62.06, "mdd": 34.60, "trades": 223},
}


async def run_single(
    config_path: str,
    lev: float,
    sl: float,
) -> dict[str, object]:
    """단일 파라미터 조합 EDA 백테스트."""
    cfg = load_config(config_path)

    # Pydantic frozen model → model_copy로 오버라이드
    portfolio = cfg.portfolio.model_copy(
        update={
            "max_leverage_cap": lev,
            "system_stop_loss": sl,
            "trailing_stop_atr_multiplier": FIXED_TS_MULT,
            "rebalance_threshold": FIXED_REBAL,
        }
    )
    cfg = cfg.model_copy(update={"portfolio": portfolio})

    settings = get_settings()
    service = MarketDataService(settings)

    start_dt = datetime.strptime(cfg.backtest.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(cfg.backtest.end, "%Y-%m-%d").replace(tzinfo=UTC)

    request = MarketDataRequest(
        symbol=cfg.backtest.symbols[0], timeframe="1m", start=start_dt, end=end_dt
    )
    data = service.get(request)
    strategy = build_strategy(cfg)

    runner = EDARunner.backtest(
        strategy=strategy,
        data=data,
        target_timeframe="1D",
        config=cfg.portfolio,
        initial_capital=cfg.backtest.capital,
    )

    metrics = await runner.run()

    return {
        "lev": lev,
        "sl": sl,
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe": metrics.sharpe_ratio,
        "mdd": metrics.max_drawdown,
        "trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "pf": metrics.profit_factor,
        "volatility": metrics.volatility,
    }


def print_results(
    symbol: str,
    results: list[dict[str, object]],
) -> None:
    """Rich 테이블로 결과 출력."""
    table = Table(title=f"Leverage/SL Sweep Results — {symbol} (TSMOM, 2020-2025)")
    table.add_column("Lev Cap", style="cyan", justify="center")
    table.add_column("SL (%)", style="cyan", justify="center")
    table.add_column("Sharpe", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Vol", justify="right")

    # Sharpe 기준 정렬
    results.sort(key=lambda r: float(str(r["sharpe"])), reverse=True)

    best_sharpe = float(str(results[0]["sharpe"]))
    for r in results:
        sharpe = float(str(r["sharpe"]))
        style = "bold green" if sharpe == best_sharpe else ""
        pf_val = f"{r['pf']:.2f}" if r["pf"] is not None else "—"
        table.add_row(
            f"{r['lev']}x",
            f"{float(str(r['sl'])) * 100:.0f}%",
            f"{sharpe:.4f}",
            f"{r['cagr']:.2f}%",
            f"-{r['mdd']:.2f}%",
            str(r["trades"]),
            f"{r['win_rate']:.1f}%",
            pf_val,
            f"{r['volatility']:.1f}%",
            style=style,
        )

    console.print(table)

    # VBT 기준선
    baseline = VBT_BASELINES.get(symbol)
    if baseline:
        console.print(
            f"\n[dim]VBT baseline ({symbol}): Sharpe={baseline['sharpe']:.2f}, "
            f"CAGR=+{baseline['cagr']:.2f}%, MDD=-{baseline['mdd']:.2f}%, "
            f"Trades={baseline['trades']:.0f}[/dim]"
        )

    # 최적 조합 요약
    best = results[0]
    console.print(
        f"\n[bold yellow]Best ({symbol}): "
        f"Lev={best['lev']}x, SL={float(str(best['sl'])) * 100:.0f}% → "
        f"Sharpe={float(str(best['sharpe'])):.4f}, CAGR={best['cagr']:.2f}%, "
        f"MDD=-{best['mdd']:.2f}%[/bold yellow]\n"
    )


async def sweep_asset(symbol: str, config_path: str) -> list[dict[str, object]]:
    """단일 자산에 대해 모든 조합 스윕."""
    combos = list(itertools.product(LEVERAGE_CAPS, STOP_LOSSES))
    console.print(f"\n[cyan]{'=' * 60}[/cyan]")
    console.print(f"[cyan]{symbol}: Running {len(combos)} combinations...[/cyan]")
    console.print(f"[cyan]Fixed: TS={FIXED_TS_MULT}x, Rebal={FIXED_REBAL:.0%}[/cyan]")
    console.print(f"[cyan]{'=' * 60}[/cyan]\n")

    results: list[dict[str, object]] = []
    for i, (lev, sl) in enumerate(combos, 1):
        console.print(f"  [{i}/{len(combos)}] Lev={lev:.1f}x, SL={sl:.0%}...", end=" ")
        result = await run_single(config_path, lev, sl)
        console.print(
            f"Sharpe={result['sharpe']:.2f}, CAGR={result['cagr']:.1f}%, "
            f"MDD={result['mdd']:.1f}%, Trades={result['trades']}"
        )
        results.append(result)

    print_results(symbol, results)
    return results


async def main() -> None:
    """모든 자산 × 모든 조합 실행."""
    logger.disable("src")  # 로그 억제

    all_results: dict[str, list[dict[str, object]]] = {}

    for symbol, config_path in ASSETS:
        all_results[symbol] = await sweep_asset(symbol, config_path)

    # 전체 요약
    console.print("\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold cyan]SUMMARY — Optimal Parameters per Asset[/bold cyan]")
    console.print("[bold cyan]{'='*60}[/bold cyan]\n")

    summary_table = Table(title="Optimal Leverage/SL per Asset")
    summary_table.add_column("Asset", style="bold")
    summary_table.add_column("Lev Cap", justify="center")
    summary_table.add_column("SL", justify="center")
    summary_table.add_column("Sharpe", justify="right")
    summary_table.add_column("CAGR", justify="right")
    summary_table.add_column("MDD", justify="right")
    summary_table.add_column("VBT Sharpe", justify="right", style="dim")

    for symbol, results in all_results.items():
        best = results[0]  # 이미 Sharpe 정렬됨
        baseline = VBT_BASELINES.get(symbol, {})
        summary_table.add_row(
            symbol,
            f"{best['lev']}x",
            f"{float(str(best['sl'])) * 100:.0f}%",
            f"{float(str(best['sharpe'])):.4f}",
            f"{best['cagr']:.2f}%",
            f"-{best['mdd']:.2f}%",
            f"{baseline.get('sharpe', 0):.2f}",
        )

    console.print(summary_table)


if __name__ == "__main__":
    asyncio.run(main())
