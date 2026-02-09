"""SL/TS 파라미터 스윕 — EDA 백테스트.

Trailing Stop ATR multiplier × System Stop Loss 조합을 테스트하여
거래 빈도와 수익성의 최적 균형점을 찾는다.
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

# --- 스윕 파라미터 --- (최적점 주변 fine-grained)
TS_MULTIPLIERS = [3.5, 4.0, 4.5, 5.0]
STOP_LOSSES = [0.10]
REBALANCE_THRESHOLDS = [0.12, 0.15, 0.18]
BASE_CONFIG = "config/tsmom_sol.yaml"


async def run_single(
    ts_mult: float,
    sl: float,
    rebal: float = 0.10,
) -> dict[str, object]:
    """단일 파라미터 조합 EDA 백테스트."""
    cfg = load_config(BASE_CONFIG)

    # Pydantic frozen model → model_copy로 오버라이드
    portfolio = cfg.portfolio.model_copy(
        update={
            "trailing_stop_atr_multiplier": ts_mult,
            "system_stop_loss": sl,
            "rebalance_threshold": rebal,
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
        "ts_mult": ts_mult,
        "sl": sl,
        "rebal": rebal,
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe": metrics.sharpe_ratio,
        "mdd": metrics.max_drawdown,
        "trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "pf": metrics.profit_factor,
        "volatility": metrics.volatility,
    }


async def main() -> None:
    """모든 조합 실행 및 결과 출력."""
    logger.disable("src")  # 로그 억제

    combos = list(itertools.product(TS_MULTIPLIERS, STOP_LOSSES, REBALANCE_THRESHOLDS))
    console.print(f"[cyan]Running {len(combos)} parameter combinations...[/cyan]\n")

    results: list[dict[str, object]] = []
    for i, (ts, sl, rebal) in enumerate(combos, 1):
        console.print(
            f"  [{i}/{len(combos)}] TS={ts:.1f}x, SL={sl:.0%}, Rebal={rebal:.0%}...", end=" "
        )
        result = await run_single(ts, sl, rebal)
        console.print(
            f"Sharpe={result['sharpe']:.2f}, CAGR={result['cagr']:.1f}%, "
            f"MDD={result['mdd']:.1f}%, Trades={result['trades']}"
        )
        results.append(result)

    # 결과 테이블
    console.print()
    table = Table(title="SL/TS Parameter Sweep Results (TSMOM SOL/USDT, 2020-2025)")
    table.add_column("TS (ATR)", style="cyan", justify="center")
    table.add_column("SL (%)", style="cyan", justify="center")
    table.add_column("Rebal", style="cyan", justify="center")
    table.add_column("Sharpe", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("PF", justify="right")

    # Sharpe 기준 정렬
    results.sort(key=lambda r: float(str(r["sharpe"])), reverse=True)

    best_sharpe = float(str(results[0]["sharpe"]))
    for r in results:
        sharpe = float(str(r["sharpe"]))
        style = "bold green" if sharpe == best_sharpe else ""
        pf_val = f"{r['pf']:.2f}" if r["pf"] is not None else "—"
        table.add_row(
            f"{r['ts_mult']}x",
            f"{float(str(r['sl'])) * 100:.0f}%",
            f"{float(str(r['rebal'])) * 100:.0f}%",
            f"{sharpe:.4f}",
            f"{r['cagr']:.2f}%",
            f"-{r['mdd']:.2f}%",
            str(r["trades"]),
            f"{r['win_rate']:.1f}%",
            pf_val,
            style=style,
        )

    console.print(table)

    # VBT 기준선
    console.print("\n[dim]VBT baseline: Sharpe=1.33, CAGR=+49.99%, MDD=-26.00%, Trades=181[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
