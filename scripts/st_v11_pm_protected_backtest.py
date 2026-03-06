"""SuperTrend v1.1 PM 방어 적용 백테스트 — 12H × 20에셋.

PM 없음 vs PM 방어(TS 3.0x ATR + SL 10%) 비교.
ACTIVE 전략과 동일 조건으로 검증.
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

STRATEGY_PARAMS = {
    "atr_period": 10,
    "multiplier": 3.0,
    "adx_period": 14,
    "adx_threshold": 25,
}

START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 6, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)

# PM 방어 설정 (ACTIVE 전략과 동일)
PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=1.0,
    rebalance_threshold=0.10,
    system_stop_loss=0.10,  # 10% SL
    use_trailing_stop=True,  # Trailing Stop 활성화
    trailing_stop_atr_multiplier=3.0,  # 3.0x ATR
    use_intrabar_trailing_stop=False,  # TF bar에서만 TS 체크 (EDA parity 핵심)
)

# PM 없음 설정 (기존 테스트와 동일)
NO_PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=1.0,
    rebalance_threshold=0.05,
    system_stop_loss=None,
    use_trailing_stop=False,
)


def run_backtest(
    engine: BacktestEngine,
    data_service: MarketDataService,
    strategy_instance: object,
    symbol: str,
    pm_config: PortfolioManagerConfig,
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
            config=pm_config,
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
    """PM 없음 vs PM 방어 비교 백테스트."""
    logger.info("SuperTrend v1.1 — No PM vs PM Protected (12H × 20 Assets)")
    logger.info("PM: SL=10%, TS=3.0x ATR, intrabar_ts=False")

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()
    strategy = get_strategy("supertrend").from_params(**STRATEGY_PARAMS)

    all_results: list[dict] = []

    for symbol in SYMBOLS:
        r_no = run_backtest(engine, data_service, strategy, symbol, NO_PM_CONFIG, "No PM")
        r_pm = run_backtest(engine, data_service, strategy, symbol, PM_CONFIG, "PM Protected")

        if r_no and r_pm:
            all_results.extend([r_no, r_pm])
            delta_sharpe = r_pm["sharpe"] - r_no["sharpe"]
            delta_mdd = r_pm["max_drawdown"] - r_no["max_drawdown"]
            logger.info(
                f"  {symbol:12s} | "
                f"NoPM {r_no['sharpe']:+.2f} → PM {r_pm['sharpe']:+.2f} "
                f"(Δ{delta_sharpe:+.2f}) | "
                f"MDD {r_no['max_drawdown']:.1f}% → {r_pm['max_drawdown']:.1f}% "
                f"(Δ{delta_mdd:+.1f}%)"
            )

    if not all_results:
        logger.error("No results")
        return

    df = pd.DataFrame(all_results)

    # 저장
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "st_v11_pm_protected_12h.csv"
    df.to_csv(csv_path, index=False)

    # 비교
    df_no = df[df["mode"] == "No PM"].set_index("symbol")
    df_pm = df[df["mode"] == "PM Protected"].set_index("symbol")

    comparison = pd.DataFrame(
        {
            "NoPM_sharpe": df_no["sharpe"],
            "PM_sharpe": df_pm["sharpe"],
            "Δ_sharpe": df_pm["sharpe"] - df_no["sharpe"],
            "NoPM_return": df_no["total_return"],
            "PM_return": df_pm["total_return"],
            "NoPM_mdd": df_no["max_drawdown"],
            "PM_mdd": df_pm["max_drawdown"],
            "Δ_mdd": df_pm["max_drawdown"] - df_no["max_drawdown"],
            "NoPM_trades": df_no["total_trades"],
            "PM_trades": df_pm["total_trades"],
            "PM_winrate": df_pm["win_rate"],
            "PM_pf": df_pm["profit_factor"],
        }
    )
    print("\n" + comparison.to_string())

    # 요약
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")

    sharpe_improved = (comparison["Δ_sharpe"] > 0).sum()
    mdd_improved = (comparison["Δ_mdd"] > 0).sum()  # less negative = improvement

    logger.info(f"NoPM Avg Sharpe: {df_no['sharpe'].mean():.3f}")
    logger.info(f"PM   Avg Sharpe: {df_pm['sharpe'].mean():.3f}")
    logger.info(f"NoPM Avg MDD:    {df_no['max_drawdown'].mean():.1f}%")
    logger.info(f"PM   Avg MDD:    {df_pm['max_drawdown'].mean():.1f}%")

    no_ge1 = (df_no["sharpe"] >= 1.0).sum()
    pm_ge1 = (df_pm["sharpe"] >= 1.0).sum()
    logger.info(f"NoPM Sharpe>=1.0: {no_ge1}/20")
    logger.info(f"PM   Sharpe>=1.0: {pm_ge1}/20")

    # Tier 1 에셋 상세
    tier1 = ["SOL/USDT", "AVAX/USDT", "BTC/USDT", "XRP/USDT", "FTM/USDT", "ETH/USDT"]
    logger.info("\n--- Tier 1 Assets (PM Protected) ---")
    for sym in tier1:
        if sym in df_pm.index:
            r = df_pm.loc[sym]
            logger.info(
                f"  {sym:12s} | Sharpe {r['sharpe']:+.2f} | "
                f"Return {r['total_return']:+.1f}% | "
                f"MDD {r['max_drawdown']:.1f}% | "
                f"Trades {r['total_trades']} | "
                f"WR {r['win_rate']:.1f}% | "
                f"PF {r['profit_factor']}"
            )

    # Tier 1 평균
    tier1_pm = df_pm.loc[[s for s in tier1 if s in df_pm.index]]
    logger.info(f"\n  Tier 1 Avg Sharpe: {tier1_pm['sharpe'].mean():.3f}")
    logger.info(f"  Tier 1 Avg MDD:    {tier1_pm['max_drawdown'].mean():.1f}%")
    logger.info(f"  Tier 1 Avg Return: {tier1_pm['total_return'].mean():.1f}%")

    logger.info(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
