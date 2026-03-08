"""SuperTrend v1.1 Universe Scan — 시가총액 Top 30 에셋 스크리닝.

현재 운용 파라미터(ATR 7, mult 2.5, ADX 14/25)로 전체 에셋 백테스트 후 랭킹.
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
START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 7, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)

# 현재 운용 파라미터 (spot_supertrend.yaml 동일)
STRATEGY_PARAMS = {
    "atr_period": 7,
    "multiplier": 2.5,
    "adx_period": 14,
    "adx_threshold": 25,
}

PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=1.0,
    rebalance_threshold=0.05,
    system_stop_loss=0.10,
    use_trailing_stop=True,
    trailing_stop_atr_multiplier=3.0,
    use_intrabar_trailing_stop=False,
)

COST_MODEL = {
    "maker_fee": 0.001,
    "taker_fee": 0.001,
    "slippage": 0.0005,
    "funding_rate_8h": 0.0,
    "market_impact": 0.0,
}

# Silver 데이터가 있는 모든 USDT 에셋
ALL_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "TRX/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT", "XLM/USDT",
    "HBAR/USDT", "LTC/USDT", "AVAX/USDT", "SUI/USDT", "TON/USDT",
    "SHIB/USDT", "DOT/USDT", "UNI/USDT", "NEAR/USDT", "APT/USDT",
    "RENDER/USDT", "FIL/USDT", "BCH/USDT", "ZEC/USDT", "TAO/USDT",
    "AAVE/USDT",
    # 기존 보유 데이터 (시총 30위 밖이지만 참고용)
    "ARB/USDT", "OP/USDT", "ATOM/USDT", "ICP/USDT",
    "INJ/USDT", "ALGO/USDT", "SEI/USDT", "PEPE/USDT",
    "S/USDT",  # Sonic (구 FTM)
]


def run_single(
    engine: BacktestEngine,
    data_service: MarketDataService,
    strategy: object,
    symbol: str,
) -> dict | None:
    """단일 심볼 백테스트."""
    try:
        config = strategy.config  # type: ignore[attr-defined]
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
            logger.warning(f"SKIP {symbol}: insufficient data ({data.periods} bars)")
            return None

        portfolio = Portfolio.create(
            initial_capital=INITIAL_CAPITAL,
            config=PM_CONFIG,
        )
        result = engine.run(
            BacktestRequest(
                data=data,
                strategy=strategy,  # type: ignore[arg-type]
                portfolio=portfolio,
                warmup_bars=warmup,
            )
        )
        m = result.metrics
        bm = result.benchmark
        return {
            "symbol": symbol.replace("/USDT", ""),
            "sharpe": round(m.sharpe_ratio, 3),
            "total_return": round(m.total_return, 1),
            "cagr": round(m.cagr, 1),
            "max_drawdown": round(m.max_drawdown, 1),
            "sortino": round(m.sortino_ratio, 2) if m.sortino_ratio else None,
            "calmar": round(m.calmar_ratio, 2) if m.calmar_ratio else None,
            "total_trades": m.total_trades,
            "win_rate": round(m.win_rate, 1),
            "profit_factor": round(m.profit_factor, 2) if m.profit_factor else None,
            "bh_return": round(bm.benchmark_return, 1) if bm else None,
            "beta": round(bm.beta, 2) if bm and bm.beta else None,
            "bars": data.periods,
        }
    except Exception as e:
        logger.warning(f"SKIP {symbol}: {e}")
        return None


def main() -> None:
    """전체 에셋 스크리닝."""
    logger.info("=" * 70)
    logger.info("SuperTrend v1.1 Universe Scan")
    logger.info(f"Params: ATR={STRATEGY_PARAMS['atr_period']}, "
                f"Mult={STRATEGY_PARAMS['multiplier']}, "
                f"ADX={STRATEGY_PARAMS['adx_period']}/{STRATEGY_PARAMS['adx_threshold']}")
    logger.info(f"TF={TIMEFRAME}, PM: TS=3.0x ATR, SL=10%, intrabar_ts=False")
    logger.info(f"Period: {START_DATE.date()} ~ {END_DATE.date()}")
    logger.info("=" * 70)

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()
    strategy = get_strategy("supertrend").from_params(**STRATEGY_PARAMS)

    results: list[dict] = []
    for i, symbol in enumerate(ALL_SYMBOLS, 1):
        logger.info(f"[{i}/{len(ALL_SYMBOLS)}] {symbol}...")
        r = run_single(engine, data_service, strategy, symbol)
        if r:
            results.append(r)
            logger.info(
                f"  → Sharpe={r['sharpe']:.3f}  "
                f"Return={r['total_return']:+.1f}%  "
                f"MDD={r['max_drawdown']:.1f}%  "
                f"Trades={r['total_trades']}  "
                f"WR={r['win_rate']:.1f}%"
            )

    if not results:
        logger.error("No results!")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based ranking

    # ── 출력 ──
    print("\n" + "=" * 90)
    print("SuperTrend v1.1 — Asset Ranking (sorted by Sharpe)")
    print("=" * 90)
    print(df.to_string())

    # Top 10 하이라이트
    top10 = df.head(10)
    print("\n" + "=" * 90)
    print("TOP 10 Assets")
    print("=" * 90)
    print(top10[["symbol", "sharpe", "total_return", "cagr", "max_drawdown",
                 "win_rate", "profit_factor", "total_trades"]].to_string())

    avg_sharpe = top10["sharpe"].mean()
    print(f"\nTop 10 Avg Sharpe: {avg_sharpe:.3f}")

    # 저장
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "st_v11_universe_scan.csv"
    df.to_csv(output_path, index_label="rank")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
