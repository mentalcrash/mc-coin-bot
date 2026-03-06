"""SuperTrend v1.1 (ADX) 멀티 타임프레임 × 20에셋 백테스트.

4개 TF (1D, 12H, 8H, 4H) × 20개 주요 에셋에 대한 성과 비교.
"""

from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from loguru import logger

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.config.settings import get_settings
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio.portfolio import Portfolio
from src.strategy import get_strategy

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────
TIMEFRAMES = ["1D", "12h", "8h", "4h"]

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

# ST v1.1 파라미터 (ADX filter 활성)
STRATEGY_PARAMS = {
    "atr_period": 10,
    "multiplier": 3.0,
    "adx_period": 14,
    "adx_threshold": 25,
}

START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 6, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)


def run_single_backtest(
    engine: BacktestEngine,
    data_service: MarketDataService,
    strategy_instance: object,
    symbol: str,
    timeframe: str,
) -> dict | None:
    """단일 심볼+TF 백테스트 실행."""
    try:
        # warmup 계산
        config = strategy_instance.config  # type: ignore[attr-defined]
        warmup = config.warmup_periods()

        req = MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start=START_DATE,
            end=END_DATE,
        )
        data = data_service.get(req)

        if data.periods < warmup + 50:
            return None

        portfolio = Portfolio.create(initial_capital=INITIAL_CAPITAL)
        bt_req = BacktestRequest(
            data=data,
            strategy=strategy_instance,  # type: ignore[arg-type]
            portfolio=portfolio,
            warmup_bars=warmup,
        )
        result = engine.run(bt_req)
        m = result.metrics
        bm = result.benchmark

        return {
            "symbol": symbol,
            "timeframe": timeframe,
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
        logger.warning(f"SKIP {symbol} {timeframe}: {e}")
        return None


def main() -> None:
    """메인 실행."""
    logger.info("SuperTrend v1.1 Multi-TF × 20 Assets Backtest")
    logger.info(f"Period: {START_DATE.date()} ~ {END_DATE.date()}")
    logger.info(f"Params: {STRATEGY_PARAMS}")

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()
    strategy = get_strategy("supertrend").from_params(**STRATEGY_PARAMS)

    all_results: list[dict] = []

    for tf in TIMEFRAMES:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Timeframe: {tf}")
        logger.info(f"{'=' * 60}")

        tf_results: list[dict] = []
        for symbol in SYMBOLS:
            result = run_single_backtest(engine, data_service, strategy, symbol, tf)
            if result:
                tf_results.append(result)
                all_results.append(result)
                logger.info(
                    f"  {symbol:12s} | "
                    f"Sharpe {result['sharpe']:+.2f} | "
                    f"Return {result['total_return']:+.1f}% | "
                    f"MDD {result['max_drawdown']:.1f}% | "
                    f"Trades {result['total_trades']}"
                )

        if tf_results:
            df_tf = pd.DataFrame(tf_results)
            sharpe_avg = df_tf["sharpe"].mean()
            sharpe_ge1 = (df_tf["sharpe"] >= 1.0).sum()
            logger.info(
                f"\n  {tf} Summary: "
                f"Avg Sharpe={sharpe_avg:.2f}, "
                f"Sharpe>=1.0: {sharpe_ge1}/{len(df_tf)}"
            )

    # 최종 결과 저장
    if all_results:
        df = pd.DataFrame(all_results)
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        csv_path = output_dir / "st_v11_multi_tf_20assets.csv"
        df.to_csv(csv_path, index=False)

        # TF별 요약 테이블
        logger.info(f"\n{'=' * 80}")
        logger.info("FINAL SUMMARY BY TIMEFRAME")
        logger.info(f"{'=' * 80}")

        summary = (
            df.groupby("timeframe")
            .agg(
                count=("sharpe", "count"),
                avg_sharpe=("sharpe", "mean"),
                med_sharpe=("sharpe", "median"),
                sharpe_ge1=("sharpe", lambda x: (x >= 1.0).sum()),
                avg_return=("total_return", "mean"),
                avg_mdd=("max_drawdown", "mean"),
                avg_trades=("total_trades", "mean"),
            )
            .reindex(TIMEFRAMES)
        )
        print("\n" + summary.to_string())

        # Sharpe >= 1.0 에셋 목록 (TF별)
        logger.info(f"\n{'=' * 80}")
        logger.info("SHARPE >= 1.0 ASSETS BY TIMEFRAME")
        logger.info(f"{'=' * 80}")
        for tf in TIMEFRAMES:
            tf_data = df[(df["timeframe"] == tf) & (df["sharpe"] >= 1.0)]
            if not tf_data.empty:
                assets = ", ".join(
                    f"{r['symbol']}({r['sharpe']:.2f})"
                    for _, r in tf_data.sort_values("sharpe", ascending=False).iterrows()
                )
                logger.info(f"  {tf}: {assets}")
            else:
                logger.info(f"  {tf}: (none)")

        logger.info(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
