"""SuperTrend + ADX — BTC 12H 백테스트.

심플 추세추종: SuperTrend 방향 + ADX 필터.
청산: 반대 신호 + Trailing Stop (3.0x ATR).
포지션: 100% vs 80% 비교.
벤치마크: Buy & Hold.
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
from src.strategy.supertrend.config import ShortMode

warnings.filterwarnings("ignore")

# -- Configuration ----------------------------------------------------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "12h"
START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 6, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)

# SuperTrend + ADX 파라미터
PARAMS = {
    "atr_period": 10,
    "multiplier": 3.0,
    "adx_period": 14,
    "adx_threshold": 25,  # ADX >= 25 일 때만 진입
}

# 테스트 시나리오
SCENARIOS: list[dict] = [
    {
        "name": "Long-Only (100%)",
        "short_mode": ShortMode.DISABLED,
        "max_leverage_cap": 1.0,
        "use_trailing_stop": True,
        "trailing_stop_atr_multiplier": 3.0,
    },
    {
        "name": "Long-Only (80%)",
        "short_mode": ShortMode.DISABLED,
        "max_leverage_cap": 0.8,
        "use_trailing_stop": True,
        "trailing_stop_atr_multiplier": 3.0,
    },
    {
        "name": "Long/Short (100%)",
        "short_mode": ShortMode.FULL,
        "max_leverage_cap": 1.0,
        "use_trailing_stop": True,
        "trailing_stop_atr_multiplier": 3.0,
    },
    {
        "name": "Long/Short (80%)",
        "short_mode": ShortMode.FULL,
        "max_leverage_cap": 0.8,
        "use_trailing_stop": True,
        "trailing_stop_atr_multiplier": 3.0,
    },
]


def run_scenario(
    engine: BacktestEngine,
    data_service: MarketDataService,
    scenario: dict,
) -> dict | None:
    """단일 시나리오 백테스트."""
    strategy = get_strategy("supertrend").from_params(
        **PARAMS,
        short_mode=scenario["short_mode"],
    )
    config = strategy.config  # type: ignore[union-attr]
    warmup = config.warmup_periods()

    data = data_service.get(
        MarketDataRequest(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start=START_DATE,
            end=END_DATE,
        )
    )
    if data.periods < warmup + 50:
        logger.error(f"Insufficient data: {data.periods} bars")
        return None

    pm_config = PortfolioManagerConfig(
        max_leverage_cap=scenario["max_leverage_cap"],
        use_trailing_stop=scenario["use_trailing_stop"],
        trailing_stop_atr_multiplier=scenario["trailing_stop_atr_multiplier"],
        rebalance_threshold=0.05,
    )
    portfolio = Portfolio.create(
        initial_capital=INITIAL_CAPITAL,
        config=pm_config,
    )

    result = engine.run(
        BacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
            warmup_bars=warmup,
        )
    )

    m = result.metrics
    bm = result.benchmark
    return {
        "scenario": scenario["name"],
        "total_return": round(m.total_return, 2),
        "cagr": round(m.cagr, 2),
        "sharpe": round(m.sharpe_ratio, 2),
        "sortino": round(m.sortino_ratio, 2) if m.sortino_ratio else None,
        "max_drawdown": round(m.max_drawdown, 2),
        "calmar": round(m.cagr / abs(m.max_drawdown) * 100, 2) if m.max_drawdown else None,
        "total_trades": m.total_trades,
        "win_rate": round(m.win_rate, 1),
        "profit_factor": round(m.profit_factor, 2) if m.profit_factor else None,
        "avg_trade_duration": str(m.avg_trade_duration) if m.avg_trade_duration else None,
        "bh_return": round(bm.benchmark_return, 2) if bm else None,
        "beta": round(bm.beta, 2) if bm and bm.beta else None,
        "alpha": round(bm.alpha, 2) if bm and bm.alpha else None,
    }


def main() -> None:
    """BTC 12H SuperTrend + ADX 백테스트."""
    logger.info("=" * 70)
    logger.info("SuperTrend + ADX  |  BTC/USDT 12H  |  2020-01 ~ 2026-03")
    logger.info(
        f"Params: ATR={PARAMS['atr_period']}, Mult={PARAMS['multiplier']}, "
        f"ADX_period={PARAMS['adx_period']}, ADX_thresh={PARAMS['adx_threshold']}"
    )
    logger.info("=" * 70)

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()

    results: list[dict] = []
    for scenario in SCENARIOS:
        logger.info(f"\n--- {scenario['name']} ---")
        r = run_scenario(engine, data_service, scenario)
        if r:
            results.append(r)
            logger.info(
                f"  Return: {r['total_return']:+.1f}%  |  "
                f"CAGR: {r['cagr']:+.1f}%  |  "
                f"Sharpe: {r['sharpe']:.2f}  |  "
                f"MDD: {r['max_drawdown']:.1f}%  |  "
                f"Trades: {r['total_trades']}  |  "
                f"WinRate: {r['win_rate']:.0f}%  |  "
                f"B&H: {r['bh_return']:+.1f}%"
            )

    if not results:
        logger.error("No results")
        return

    # -- Summary Table --
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SUMMARY: SuperTrend + ADX | BTC/USDT 12H | 2020-2026")
    print("=" * 70)
    print(df.to_string(index=False))

    # -- Save --
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "st_adx_btc_12h.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
