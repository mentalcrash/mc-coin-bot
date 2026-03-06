"""SuperTrend v1.1 FULL Short Mode — 12H × 20에셋 백테스트.

Long-only vs Long/Short 비교.
"""

from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.config.settings import get_settings
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio.portfolio import Portfolio
from src.strategy import get_strategy
from src.strategy.supertrend.config import ShortMode

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────
TIMEFRAME = "12h"

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "MATIC/USDT",
    "NEAR/USDT",
    "ARB/USDT",
    "OP/USDT",
    "ATOM/USDT",
    "DOT/USDT",
    "LTC/USDT",
    "UNI/USDT",
    "FTM/USDT",
    "ICP/USDT",
    "FIL/USDT",
]

BASE_PARAMS = {
    "atr_period": 10,
    "multiplier": 3.0,
    "adx_period": 14,
    "adx_threshold": 25,
}

START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 6, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)


def run_backtest(
    engine: BacktestEngine,
    data_service: MarketDataService,
    strategy_instance: object,
    symbol: str,
    mode_name: str,
) -> dict | None:
    """단일 심볼 백테스트."""
    try:
        config = strategy_instance.config  # type: ignore[attr-defined]
        warmup = config.warmup_periods()

        data = data_service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe=TIMEFRAME,
                start=START_DATE,
                end=END_DATE,
            )
        )
        if data.periods < warmup + 50:
            return None

        portfolio = Portfolio.create(initial_capital=INITIAL_CAPITAL)
        result = engine.run(
            BacktestRequest(
                data=data,
                strategy=strategy_instance,  # type: ignore[arg-type]
                portfolio=portfolio,
                warmup_bars=warmup,
            )
        )
        m = result.metrics
        bm = result.benchmark
        return {
            "symbol": symbol,
            "mode": mode_name,
            "total_return": round(m.total_return, 2),
            "cagr": round(m.cagr, 2),
            "sharpe": round(m.sharpe_ratio, 2),
            "sortino": round(m.sortino_ratio, 2) if m.sortino_ratio else None,
            "max_drawdown": round(m.max_drawdown, 2),
            "total_trades": m.total_trades,
            "win_rate": round(m.win_rate, 1),
            "profit_factor": round(m.profit_factor, 2) if m.profit_factor else None,
            "beta": round(bm.beta, 2) if bm and bm.beta else None,
            "bh_return": round(bm.benchmark_return, 2) if bm else None,
        }
    except Exception as e:
        logger.warning(f"SKIP {symbol} {mode_name}: {e}")
        return None


def main() -> None:
    """Long-only vs FULL Short 비교 백테스트."""
    logger.info("SuperTrend v1.1 — Long-only vs FULL Short (12H × 20 Assets)")

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()

    # 두 모드 전략 생성
    st_long = get_strategy("supertrend").from_params(
        **BASE_PARAMS,
        short_mode=ShortMode.DISABLED,
    )
    st_full = get_strategy("supertrend").from_params(
        **BASE_PARAMS,
        short_mode=ShortMode.FULL,
    )

    all_results: list[dict] = []

    for symbol in SYMBOLS:
        r_long = run_backtest(engine, data_service, st_long, symbol, "Long-Only")
        r_full = run_backtest(engine, data_service, st_full, symbol, "Long/Short")

        if r_long and r_full:
            all_results.extend([r_long, r_full])
            delta_sharpe = r_full["sharpe"] - r_long["sharpe"]
            delta_mdd = r_full["max_drawdown"] - r_long["max_drawdown"]
            logger.info(
                f"  {symbol:12s} | "
                f"LO Sharpe {r_long['sharpe']:+.2f} → LS {r_full['sharpe']:+.2f} "
                f"(Δ{delta_sharpe:+.2f}) | "
                f"LO MDD {r_long['max_drawdown']:.1f}% → LS {r_full['max_drawdown']:.1f}% "
                f"(Δ{delta_mdd:+.1f}%)"
            )

    if not all_results:
        logger.error("No results")
        return

    df = pd.DataFrame(all_results)

    # 결과 저장
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "st_v11_full_short_12h.csv"
    df.to_csv(csv_path, index=False)

    # 비교 테이블
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARISON: Long-Only vs Long/Short (12H)")
    logger.info(f"{'=' * 80}")

    df_long = df[df["mode"] == "Long-Only"].set_index("symbol")
    df_full = df[df["mode"] == "Long/Short"].set_index("symbol")

    comparison = pd.DataFrame(
        {
            "LO_sharpe": df_long["sharpe"],
            "LS_sharpe": df_full["sharpe"],
            "Δ_sharpe": df_full["sharpe"] - df_long["sharpe"],
            "LO_return": df_long["total_return"],
            "LS_return": df_full["total_return"],
            "LO_mdd": df_long["max_drawdown"],
            "LS_mdd": df_full["max_drawdown"],
            "Δ_mdd": df_full["max_drawdown"] - df_long["max_drawdown"],
            "LO_trades": df_long["total_trades"],
            "LS_trades": df_full["total_trades"],
        }
    )
    print("\n" + comparison.to_string())

    # 요약 통계
    improved = (comparison["Δ_sharpe"] > 0).sum()
    degraded = (comparison["Δ_sharpe"] < 0).sum()
    mdd_improved = (comparison["Δ_mdd"] > 0).sum()  # MDD가 양수로 변하면 개선 (덜 음수)

    logger.info("\n--- Summary ---")
    logger.info(f"Sharpe 개선: {improved}/20, 악화: {degraded}/20")
    logger.info(f"MDD 개선: {mdd_improved}/20")
    logger.info(f"LO Avg Sharpe: {df_long['sharpe'].mean():.3f}")
    logger.info(f"LS Avg Sharpe: {df_full['sharpe'].mean():.3f}")
    logger.info(f"LO Avg MDD: {df_long['max_drawdown'].mean():.1f}%")
    logger.info(f"LS Avg MDD: {df_full['max_drawdown'].mean():.1f}%")

    # Sharpe >= 1.0 비교
    lo_ge1 = (df_long["sharpe"] >= 1.0).sum()
    ls_ge1 = (df_full["sharpe"] >= 1.0).sum()
    logger.info(f"LO Sharpe>=1.0: {lo_ge1}/20")
    logger.info(f"LS Sharpe>=1.0: {ls_ge1}/20")

    logger.info(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
