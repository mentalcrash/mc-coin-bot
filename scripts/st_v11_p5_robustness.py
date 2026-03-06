"""SuperTrend v1.1 P5 파라미터 로버스트니스 — 12H × Tier 1.

파라미터 그리드:
  ATR period: 7~14 (8값)
  Multiplier: 2.0~4.0 (5값)
  ADX threshold: 20~30 (5값)
  → 200 조합 × 6 에셋 = 1,200 백테스트

고원(Plateau) 검증: Sharpe >= 0.8인 조합 비율 → 50% 이상이면 PASS.
"""

from __future__ import annotations

import itertools
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

# Tier 1 에셋
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "AVAX/USDT",
    "FTM/USDT",
]

# 파라미터 그리드
ATR_PERIODS = [7, 8, 9, 10, 11, 12, 13, 14]
MULTIPLIERS = [2.0, 2.5, 3.0, 3.5, 4.0]
ADX_THRESHOLDS = [20, 22, 25, 27, 30]

# PM 없음 (SuperTrend 내장 손절)
PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=1.0,
    rebalance_threshold=0.05,
    system_stop_loss=None,
    use_trailing_stop=False,
)

START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 6, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)

# 고원 판정 기준
PLATEAU_SHARPE_THRESHOLD = 0.8  # Sharpe >= 0.8인 조합 비율
PLATEAU_PASS_RATIO = 0.50  # 50% 이상이면 PASS


def run_single(
    engine: BacktestEngine,
    data_service: MarketDataService,
    symbol: str,
    atr_period: int,
    multiplier: float,
    adx_threshold: float,
) -> dict | None:
    """단일 파라미터 조합 백테스트."""
    try:
        strategy = get_strategy("supertrend").from_params(
            atr_period=atr_period,
            multiplier=multiplier,
            adx_period=14,  # ADX period 고정 (민감도 낮음)
            adx_threshold=adx_threshold,
        )
        config = strategy.config
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
                strategy=strategy,
                portfolio=portfolio,
                warmup_bars=warmup,
            )
        )
        m = result.metrics
        return {
            "symbol": symbol,
            "atr_period": atr_period,
            "multiplier": multiplier,
            "adx_threshold": adx_threshold,
            "sharpe": round(m.sharpe_ratio, 3),
            "sortino": round(m.sortino_ratio, 3) if m.sortino_ratio else None,
            "total_return": round(m.total_return, 2),
            "cagr": round(m.cagr, 2),
            "max_drawdown": round(m.max_drawdown, 2),
            "total_trades": m.total_trades,
            "win_rate": round(m.win_rate, 1),
            "profit_factor": round(m.profit_factor, 3) if m.profit_factor else None,
        }
    except Exception as e:
        logger.warning(f"SKIP {symbol} atr={atr_period} mult={multiplier} adx={adx_threshold}: {e}")
        return None


def analyze_plateau(df: pd.DataFrame, symbol: str) -> dict:
    """에셋별 고원 분석."""
    sym_df = df[df["symbol"] == symbol].copy()
    total = len(sym_df)
    if total == 0:
        return {"symbol": symbol, "total": 0}

    ge_08 = (sym_df["sharpe"] >= 0.8).sum()
    ge_10 = (sym_df["sharpe"] >= 1.0).sum()
    ge_05 = (sym_df["sharpe"] >= 0.5).sum()

    return {
        "symbol": symbol,
        "total": total,
        "sharpe_mean": round(sym_df["sharpe"].mean(), 3),
        "sharpe_std": round(sym_df["sharpe"].std(), 3),
        "sharpe_min": round(sym_df["sharpe"].min(), 3),
        "sharpe_max": round(sym_df["sharpe"].max(), 3),
        "sharpe_median": round(sym_df["sharpe"].median(), 3),
        "pct_ge_05": round(ge_05 / total * 100, 1),
        "pct_ge_08": round(ge_08 / total * 100, 1),
        "pct_ge_10": round(ge_10 / total * 100, 1),
        "mdd_mean": round(sym_df["max_drawdown"].mean(), 1),
        "mdd_std": round(sym_df["max_drawdown"].std(), 1),
        "best_params": _best_params(sym_df),
    }


def _best_params(sym_df: pd.DataFrame) -> str:
    """Best Sharpe 파라미터."""
    best = sym_df.loc[sym_df["sharpe"].idxmax()]
    return f"ATR={int(best['atr_period'])},M={best['multiplier']},ADX={int(best['adx_threshold'])}"


def analyze_param_sensitivity(df: pd.DataFrame) -> None:
    """파라미터별 민감도 분석."""
    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 60)

    # ATR Period 민감도
    logger.info("\n--- ATR Period Sensitivity ---")
    for atr in ATR_PERIODS:
        sub = df[df["atr_period"] == atr]
        avg_sharpe = sub["sharpe"].mean()
        avg_mdd = sub["max_drawdown"].mean()
        ge10 = (sub["sharpe"] >= 1.0).sum()
        logger.info(
            f"  ATR={atr:2d} | Avg Sharpe: {avg_sharpe:+.3f} | "
            f"Avg MDD: {avg_mdd:.1f}% | Sharpe>=1.0: {ge10}"
        )

    # Multiplier 민감도
    logger.info("\n--- Multiplier Sensitivity ---")
    for mult in MULTIPLIERS:
        sub = df[df["multiplier"] == mult]
        avg_sharpe = sub["sharpe"].mean()
        avg_mdd = sub["max_drawdown"].mean()
        ge10 = (sub["sharpe"] >= 1.0).sum()
        logger.info(
            f"  Mult={mult:.1f} | Avg Sharpe: {avg_sharpe:+.3f} | "
            f"Avg MDD: {avg_mdd:.1f}% | Sharpe>=1.0: {ge10}"
        )

    # ADX Threshold 민감도
    logger.info("\n--- ADX Threshold Sensitivity ---")
    for adx in ADX_THRESHOLDS:
        sub = df[df["adx_threshold"] == adx]
        avg_sharpe = sub["sharpe"].mean()
        avg_mdd = sub["max_drawdown"].mean()
        ge10 = (sub["sharpe"] >= 1.0).sum()
        logger.info(
            f"  ADX={adx:2d} | Avg Sharpe: {avg_sharpe:+.3f} | "
            f"Avg MDD: {avg_mdd:.1f}% | Sharpe>=1.0: {ge10}"
        )


def analyze_heatmap(df: pd.DataFrame, symbol: str) -> None:
    """ATR × Multiplier 히트맵 (ADX=25 고정)."""
    sub = df[(df["symbol"] == symbol) & (df["adx_threshold"] == 25)]
    if sub.empty:
        return

    pivot = sub.pivot_table(
        values="sharpe",
        index="atr_period",
        columns="multiplier",
        aggfunc="mean",
    )
    logger.info(f"\n--- {symbol} Heatmap (ATR × Mult, ADX=25) ---")
    logger.info(pivot.to_string(float_format=lambda x: f"{x:+.2f}"))


def main() -> None:
    """P5 파라미터 로버스트니스 테스트."""
    param_grid = list(itertools.product(ATR_PERIODS, MULTIPLIERS, ADX_THRESHOLDS))
    total_combos = len(param_grid)
    total_runs = total_combos * len(SYMBOLS)

    logger.info("SuperTrend v1.1 P5 Parameter Robustness — 12H × Tier 1")
    logger.info(
        f"Grid: ATR({ATR_PERIODS[0]}~{ATR_PERIODS[-1]}) × "
        f"Mult({MULTIPLIERS[0]}~{MULTIPLIERS[-1]}) × "
        f"ADX({ADX_THRESHOLDS[0]}~{ADX_THRESHOLDS[-1]})"
    )
    logger.info(f"Combos: {total_combos} × {len(SYMBOLS)} assets = {total_runs} backtests")

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()

    # 데이터 사전 로드 (캐싱)
    logger.info("Loading data...")
    data_cache: dict[str, object] = {}
    for symbol in SYMBOLS:
        data_cache[symbol] = data_service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe=TIMEFRAME,
                start=START_DATE,
                end=END_DATE,
            )
        )
    logger.info(f"Data loaded for {len(data_cache)} symbols")

    all_results: list[dict] = []
    done = 0

    for symbol in SYMBOLS:
        symbol_results = 0
        for atr, mult, adx in param_grid:
            r = run_single(engine, data_service, symbol, atr, mult, adx)
            if r:
                all_results.append(r)
                symbol_results += 1
            done += 1
            if done % 100 == 0:
                logger.info(f"  Progress: {done}/{total_runs} ({done / total_runs * 100:.0f}%)")
        logger.info(f"  {symbol}: {symbol_results}/{total_combos} completed")

    if not all_results:
        logger.error("No results")
        return

    df = pd.DataFrame(all_results)

    # 저장
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "st_v11_p5_robustness_12h.csv"
    df.to_csv(csv_path, index=False)

    # ── 에셋별 고원 분석 ──────────────────────────────────────────
    logger.info(f"\n{'=' * 70}")
    logger.info("P5 PARAMETER PLATEAU ANALYSIS")
    logger.info(f"{'=' * 70}")
    logger.info(
        f"Plateau criterion: Sharpe >= {PLATEAU_SHARPE_THRESHOLD} in >= {PLATEAU_PASS_RATIO * 100:.0f}% of combos"
    )

    plateau_results = []
    for symbol in SYMBOLS:
        p = analyze_plateau(df, symbol)
        plateau_results.append(p)
        if p["total"] == 0:
            continue
        status = "PASS" if p["pct_ge_08"] >= PLATEAU_PASS_RATIO * 100 else "FAIL"
        logger.info(
            f"\n  {symbol:12s} [{status}]"
            f"\n    Sharpe: {p['sharpe_mean']:+.3f} ± {p['sharpe_std']:.3f} "
            f"(min {p['sharpe_min']:+.3f}, max {p['sharpe_max']:+.3f}, med {p['sharpe_median']:+.3f})"
            f"\n    MDD:    {p['mdd_mean']:.1f}% ± {p['mdd_std']:.1f}%"
            f"\n    Sharpe >= 0.5: {p['pct_ge_05']:.1f}% | "
            f">= 0.8: {p['pct_ge_08']:.1f}% | >= 1.0: {p['pct_ge_10']:.1f}%"
            f"\n    Best:   {p['best_params']}"
        )

    # ── 파라미터 민감도 ──────────────────────────────────────────
    analyze_param_sensitivity(df)

    # ── 주요 에셋 히트맵 ──────────────────────────────────────────
    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        analyze_heatmap(df, symbol)

    # ── 종합 판정 ──────────────────────────────────────────────────
    logger.info(f"\n{'=' * 70}")
    logger.info("P5 SUMMARY")
    logger.info(f"{'=' * 70}")

    all_sharpe = df["sharpe"]
    total_combos_all = len(all_sharpe)
    ge_08_all = (all_sharpe >= 0.8).sum()
    ge_10_all = (all_sharpe >= 1.0).sum()
    ge_05_all = (all_sharpe >= 0.5).sum()

    logger.info(f"Total backtests: {total_combos_all}")
    logger.info(f"Global Avg Sharpe: {all_sharpe.mean():+.3f} ± {all_sharpe.std():.3f}")
    logger.info(
        f"Sharpe >= 0.5: {ge_05_all}/{total_combos_all} ({ge_05_all / total_combos_all * 100:.1f}%)"
    )
    logger.info(
        f"Sharpe >= 0.8: {ge_08_all}/{total_combos_all} ({ge_08_all / total_combos_all * 100:.1f}%)"
    )
    logger.info(
        f"Sharpe >= 1.0: {ge_10_all}/{total_combos_all} ({ge_10_all / total_combos_all * 100:.1f}%)"
    )

    # 기본값(ATR=10, Mult=3.0, ADX=25) vs 전체 평균 비교
    default_df = df[
        (df["atr_period"] == 10) & (df["multiplier"] == 3.0) & (df["adx_threshold"] == 25)
    ]
    if not default_df.empty:
        logger.info("\nDefault (ATR=10, M=3.0, ADX=25):")
        for _, row in default_df.iterrows():
            rank = (df[df["symbol"] == row["symbol"]]["sharpe"] >= row["sharpe"]).sum()
            total_sym = len(df[df["symbol"] == row["symbol"]])
            pct_rank = rank / total_sym * 100
            logger.info(
                f"  {row['symbol']:12s} Sharpe {row['sharpe']:+.3f} "
                f"(rank {rank}/{total_sym}, top {pct_rank:.0f}%)"
            )

    # Pass/Fail 판정
    pass_count = sum(
        1 for p in plateau_results if p["total"] > 0 and p["pct_ge_08"] >= PLATEAU_PASS_RATIO * 100
    )
    total_assets = sum(1 for p in plateau_results if p["total"] > 0)

    overall = "PASS" if pass_count >= total_assets * 0.5 else "FAIL"
    logger.info(f"\nPlateau PASS: {pass_count}/{total_assets} assets")
    logger.info(f"Overall P5: {overall}")

    # 최적 파라미터 범위 (상위 10% Sharpe 기준)
    top_10_pct = all_sharpe.quantile(0.90)
    top_df = df[df["sharpe"] >= top_10_pct]
    logger.info(f"\n--- Top 10% Parameter Ranges (Sharpe >= {top_10_pct:.3f}) ---")
    logger.info(
        f"  ATR:  {int(top_df['atr_period'].min())} ~ {int(top_df['atr_period'].max())} "
        f"(mode: {int(top_df['atr_period'].mode().iloc[0])})"
    )
    logger.info(
        f"  Mult: {top_df['multiplier'].min():.1f} ~ {top_df['multiplier'].max():.1f} "
        f"(mode: {top_df['multiplier'].mode().iloc[0]:.1f})"
    )
    logger.info(
        f"  ADX:  {int(top_df['adx_threshold'].min())} ~ {int(top_df['adx_threshold'].max())} "
        f"(mode: {int(top_df['adx_threshold'].mode().iloc[0])})"
    )

    logger.info(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
