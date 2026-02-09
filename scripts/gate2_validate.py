#!/usr/bin/env python3
"""Gate 2: 단일에셋 IS/OOS 70/30 검증 스크립트.

18개 활성 전략에 대해 Best Asset 기준 단일에셋 IS/OOS 검증을 수행합니다.
평가 표준: OOS Sharpe >= 0.3, Sharpe Decay < 50%, OOS Return > 0%
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.validation import TieredValidator, ValidationLevel
from src.config.settings import get_settings
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.portfolio import Portfolio
from src.strategy.registry import get_strategy

# Gate 2 평가 대상: (전략 CLI 이름, Best Asset)
STRATEGIES: list[tuple[str, str]] = [
    # G1 PASS 전략 (7개)
    ("vol-regime", "ETH/USDT"),
    ("tsmom", "SOL/USDT"),
    ("enhanced-tsmom", "BTC/USDT"),
    ("vol-structure", "SOL/USDT"),
    ("kama", "DOGE/USDT"),
    ("vol-adaptive", "SOL/USDT"),
    ("donchian", "SOL/USDT"),
    # G1 WATCH 전략 (11개)
    ("donchian-ensemble", "ETH/USDT"),
    ("adx-regime", "SOL/USDT"),
    ("ttm-squeeze", "BTC/USDT"),
    ("stoch-mom", "SOL/USDT"),
    ("max-min", "DOGE/USDT"),
    ("gk-breakout", "DOGE/USDT"),
    ("mtf-macd", "SOL/USDT"),
    ("hmm-regime", "BTC/USDT"),
    ("adaptive-breakout", "SOL/USDT"),
    ("bb-rsi", "SOL/USDT"),
    ("mom-mr-blend", "ETH/USDT"),
]

# Gate 2 통과 기준 (전략 평가 표준 기준)
G2_MIN_OOS_SHARPE = 0.3
G2_MAX_SHARPE_DECAY = 0.50  # 50%
G2_MIN_OOS_RETURN = 0.0  # 양수 수익


def judge_gate2(
    oos_sharpe: float,
    sharpe_decay: float,
    oos_return: float,
) -> str:
    """Gate 2 판정 (전략 평가 표준 기준)."""
    if (
        oos_sharpe >= G2_MIN_OOS_SHARPE
        and sharpe_decay < G2_MAX_SHARPE_DECAY
        and oos_return > G2_MIN_OOS_RETURN
    ):
        return "PASS"
    return "FAIL"


def main() -> None:
    settings = get_settings()
    data_service = MarketDataService(settings)
    validator = TieredValidator()

    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)

    results: list[dict[str, object]] = []
    errors: list[tuple[str, str, str]] = []

    print("=" * 90)
    print("Gate 2: Single-Asset IS/OOS 70/30 Validation")
    print(f"Period: {start.date()} ~ {end.date()}")
    print(
        f"Criteria: OOS Sharpe >= {G2_MIN_OOS_SHARPE}, Decay < {G2_MAX_SHARPE_DECAY:.0%}, OOS Return > 0%"
    )
    print("=" * 90)

    for strategy_name, best_asset in STRATEGIES:
        t0 = time.time()
        try:
            # 데이터 로드
            data = data_service.get(
                MarketDataRequest(
                    symbol=best_asset,
                    timeframe="1D",
                    start=start,
                    end=end,
                )
            )

            # 전략 생성
            strategy_cls = get_strategy(strategy_name)
            strategy = strategy_cls()
            config_kwargs = strategy_cls.recommended_config()
            portfolio = Portfolio.create(
                initial_capital=Decimal(100000),
                config=PortfolioManagerConfig(**config_kwargs),
            )

            # IS/OOS 검증
            result = validator.validate(
                level=ValidationLevel.QUICK,
                data=data,
                strategy=strategy,
                portfolio=portfolio,
                split_ratio=0.7,
            )

            fold = result.fold_results[0]
            oos_sharpe = fold.test_sharpe
            is_sharpe = fold.train_sharpe
            decay = fold.sharpe_decay
            oos_return = fold.test_return
            is_return = fold.train_return
            oos_mdd = fold.test_max_drawdown
            oos_trades = fold.test_trades

            gate2_verdict = judge_gate2(oos_sharpe, decay, oos_return)
            elapsed = time.time() - t0

            results.append(
                {
                    "strategy": strategy_name,
                    "asset": best_asset,
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": oos_sharpe,
                    "decay": decay,
                    "is_return": is_return,
                    "oos_return": oos_return,
                    "oos_mdd": oos_mdd,
                    "oos_trades": oos_trades,
                    "code_verdict": result.verdict,
                    "gate2_verdict": gate2_verdict,
                    "elapsed": elapsed,
                }
            )

            status = "✓" if gate2_verdict == "PASS" else "✗"
            print(
                f"  {status} {strategy_name:<22} {best_asset:<12} "
                f"IS={is_sharpe:>6.2f}  OOS={oos_sharpe:>6.2f}  "
                f"Decay={decay:>6.1%}  OOS_Ret={oos_return:>7.1f}%  "
                f"G2={gate2_verdict}  ({elapsed:.1f}s)"
            )

        except Exception as e:
            elapsed = time.time() - t0
            errors.append((strategy_name, best_asset, str(e)))
            print(f"  ✗ {strategy_name:<22} {best_asset:<12} ERROR: {e!s:.60s}  ({elapsed:.1f}s)")

    # 요약
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    passed = [r for r in results if r["gate2_verdict"] == "PASS"]
    failed = [r for r in results if r["gate2_verdict"] == "FAIL"]

    print(f"\nPASS: {len(passed)}/{len(results)}")
    for r in passed:
        print(
            f"  ✓ {r['strategy']:<22} OOS Sharpe={r['oos_sharpe']:>5.2f}  Decay={r['decay']:>5.1%}"
        )

    print(f"\nFAIL: {len(failed)}/{len(results)}")
    for r in failed:
        reasons = []
        if r["oos_sharpe"] < G2_MIN_OOS_SHARPE:
            reasons.append(f"OOS Sharpe {r['oos_sharpe']:.2f} < {G2_MIN_OOS_SHARPE}")
        if r["decay"] >= G2_MAX_SHARPE_DECAY:
            reasons.append(f"Decay {r['decay']:.1%} >= {G2_MAX_SHARPE_DECAY:.0%}")
        if r["oos_return"] <= G2_MIN_OOS_RETURN:
            reasons.append(f"OOS Return {r['oos_return']:.1f}% <= 0%")
        print(f"  ✗ {r['strategy']:<22} {', '.join(reasons)}")

    if errors:
        print(f"\nERRORS: {len(errors)}")
        for name, asset, err in errors:
            print(f"  ! {name:<22} {asset:<12} {err}")

    # 상세 테이블
    print("\n" + "-" * 90)
    print(
        f"{'Strategy':<22} {'Asset':<12} {'IS_Sh':>6} {'OOS_Sh':>7} {'Decay':>7} {'OOS_Ret':>8} {'OOS_MDD':>8} {'Trades':>7} {'G2':>5}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['strategy']:<22} {r['asset']:<12} "
            f"{r['is_sharpe']:>6.2f} {r['oos_sharpe']:>7.2f} "
            f"{r['decay']:>6.1%} {r['oos_return']:>7.1f}% "
            f"{r['oos_mdd']:>7.1f}% {r['oos_trades']:>7d} "
            f"{r['gate2_verdict']:>5}"
        )
    print("-" * 90)


if __name__ == "__main__":
    main()
