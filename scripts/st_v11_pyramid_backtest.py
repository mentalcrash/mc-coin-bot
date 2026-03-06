"""SuperTrend v1.1 분할 진입(Pyramiding) 백테스트 — 12H × 20에셋.

전체 진입 vs 분할 진입(40/35/25) 비교. PM 없음.
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
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.portfolio import Portfolio
from src.strategy import get_strategy

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

# PM 없음 (SuperTrend 내장 손절 사용)
PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=1.0,
    rebalance_threshold=0.05,
    system_stop_loss=None,
    use_trailing_stop=False,
)

START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 6, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)


def run_backtest(
    engine: BacktestEngine,
    data_service: MarketDataService,
    strategy_instance: object,
    symbol: str,
    label: str,
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

        portfolio = Portfolio.create(
            initial_capital=INITIAL_CAPITAL,
            config=PM_CONFIG,
        )
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
            "mode": label,
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
        logger.warning(f"SKIP {symbol} {label}: {e}")
        return None


def main() -> None:
    """전체 진입 vs 분할 진입 비교 백테스트."""
    logger.info("SuperTrend v1.1 — Full Entry vs Pyramid Entry (12H × 20 Assets)")
    logger.info("Pyramid: 40% → +35%(new high) → +25%(ADX>=30)")

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()

    # 전체 진입 (기존)
    st_full = get_strategy("supertrend").from_params(**BASE_PARAMS)
    # 분할 진입
    st_pyramid = get_strategy("supertrend").from_params(
        **BASE_PARAMS,
        use_pyramiding=True,
        pyramid_stage1_pct=0.40,
        pyramid_stage2_pct=0.35,
        pyramid_stage3_pct=0.25,
        pyramid_high_period=20,
        pyramid_adx_strong=30.0,
    )

    all_results: list[dict] = []

    for symbol in SYMBOLS:
        r_full = run_backtest(engine, data_service, st_full, symbol, "Full Entry")
        r_pyr = run_backtest(engine, data_service, st_pyramid, symbol, "Pyramid")

        if r_full and r_pyr:
            all_results.extend([r_full, r_pyr])
            delta_sharpe = r_pyr["sharpe"] - r_full["sharpe"]
            delta_mdd = r_pyr["max_drawdown"] - r_full["max_drawdown"]
            logger.info(
                f"  {symbol:12s} | "
                f"Full {r_full['sharpe']:+.2f} → Pyr {r_pyr['sharpe']:+.2f} "
                f"(Δ{delta_sharpe:+.2f}) | "
                f"MDD {r_full['max_drawdown']:.1f}% → {r_pyr['max_drawdown']:.1f}% "
                f"(Δ{delta_mdd:+.1f}%)"
            )

    if not all_results:
        logger.error("No results")
        return

    df = pd.DataFrame(all_results)

    # 저장
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "st_v11_pyramid_12h.csv"
    df.to_csv(csv_path, index=False)

    # 비교
    df_f = df[df["mode"] == "Full Entry"].set_index("symbol")
    df_p = df[df["mode"] == "Pyramid"].set_index("symbol")

    comparison = pd.DataFrame(
        {
            "Full_sharpe": df_f["sharpe"],
            "Pyr_sharpe": df_p["sharpe"],
            "Δ_sharpe": df_p["sharpe"] - df_f["sharpe"],
            "Full_return": df_f["total_return"],
            "Pyr_return": df_p["total_return"],
            "Full_mdd": df_f["max_drawdown"],
            "Pyr_mdd": df_p["max_drawdown"],
            "Δ_mdd": df_p["max_drawdown"] - df_f["max_drawdown"],
            "Pyr_trades": df_p["total_trades"],
            "Pyr_winrate": df_p["win_rate"],
            "Pyr_pf": df_p["profit_factor"],
            "Pyr_sortino": df_p["sortino"],
        }
    )
    print("\n" + comparison.to_string())

    # 요약
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")

    sharpe_improved = (comparison["Δ_sharpe"] > 0).sum()
    mdd_improved = (comparison["Δ_mdd"] < 0).sum()

    logger.info(f"Full  Avg Sharpe: {df_f['sharpe'].mean():.3f}")
    logger.info(f"Pyr   Avg Sharpe: {df_p['sharpe'].mean():.3f}")
    logger.info(f"Full  Avg MDD:    {df_f['max_drawdown'].mean():.1f}%")
    logger.info(f"Pyr   Avg MDD:    {df_p['max_drawdown'].mean():.1f}%")
    logger.info(f"Sharpe 개선: {sharpe_improved}/20")
    logger.info(f"MDD 개선:    {mdd_improved}/20")

    f_ge1 = (df_f["sharpe"] >= 1.0).sum()
    p_ge1 = (df_p["sharpe"] >= 1.0).sum()
    logger.info(f"Full Sharpe>=1.0: {f_ge1}/20")
    logger.info(f"Pyr  Sharpe>=1.0: {p_ge1}/20")

    # Tier 1 상세
    tier1 = ["SOL/USDT", "AVAX/USDT", "BTC/USDT", "XRP/USDT", "FTM/USDT", "ETH/USDT"]
    logger.info("\n--- Tier 1 Comparison ---")
    for sym in tier1:
        if sym in df_f.index and sym in df_p.index:
            f = df_f.loc[sym]
            p = df_p.loc[sym]
            delta = p["sharpe"] - f["sharpe"]
            logger.info(
                f"  {sym:12s} | Full {f['sharpe']:+.2f} → Pyr {p['sharpe']:+.2f} "
                f"(Δ{delta:+.2f}) | MDD {f['max_drawdown']:.1f}% → {p['max_drawdown']:.1f}%"
            )

    tier1_f = df_f.loc[[s for s in tier1 if s in df_f.index]]
    tier1_p = df_p.loc[[s for s in tier1 if s in df_p.index]]
    logger.info(
        f"\n  Tier 1 Full Avg: Sharpe {tier1_f['sharpe'].mean():.3f}, MDD {tier1_f['max_drawdown'].mean():.1f}%"
    )
    logger.info(
        f"  Tier 1 Pyr  Avg: Sharpe {tier1_p['sharpe'].mean():.3f}, MDD {tier1_p['max_drawdown'].mean():.1f}%"
    )

    logger.info(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
