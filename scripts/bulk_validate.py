#!/usr/bin/env python3
"""Bulk Gate 2 Validation: Gate 1 PASS 전략 × Multi-Asset IS/OOS Quick Validation.

Gate 1을 통과한 7개 전략에 대해 8-coin Equal Weight 멀티에셋 포트폴리오로
IS/OOS 70/30 Quick Validation을 일괄 실행하고, 결과를 JSON으로 저장한다.

Usage:
    uv run python scripts/bulk_validate.py

Output:
    results/gate2_validation_results.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.validation import TieredValidator, ValidationLevel
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy

# =============================================================================
# Constants
# =============================================================================

# Gate 1 PASS 전략 (Sharpe > 1.0)
GATE1_PASS_STRATEGIES = [
    "vol-regime",
    "tsmom",
    "enhanced-tsmom",
    "vol-structure",
    "kama",
    "vol-adaptive",
    "donchian",
]

SYMBOLS_8 = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]

SYMBOLS_5 = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
]

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)

INITIAL_CAPITAL = Decimal(100000)
RESULTS_DIR = ROOT / "results"


# =============================================================================
# Helpers
# =============================================================================


def create_portfolio(strategy_name: str) -> Portfolio:
    """전략별 권장 설정으로 Portfolio 생성."""
    strategy_cls = get_strategy(strategy_name)
    rec = strategy_cls.recommended_config()

    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    return Portfolio.create(initial_capital=INITIAL_CAPITAL, config=pm_config)


def fold_to_dict(fold: Any) -> dict[str, Any]:
    """FoldResult → JSON-serializable dict."""
    return {
        "fold_id": fold.fold_id,
        "train_sharpe": fold.train_sharpe,
        "test_sharpe": fold.test_sharpe,
        "train_return": fold.train_return,
        "test_return": fold.test_return,
        "train_max_drawdown": fold.train_max_drawdown,
        "test_max_drawdown": fold.test_max_drawdown,
        "train_trades": fold.train_trades,
        "test_trades": fold.test_trades,
        "sharpe_decay": fold.sharpe_decay,
    }


def validate_strategy(
    validator: TieredValidator,
    service: MarketDataService,
    strategy_name: str,
    symbols: list[str],
) -> dict[str, Any]:
    """단일 전략 Gate 2 검증."""
    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls()
    portfolio = create_portfolio(strategy_name)

    multi_data = service.get_multi(
        symbols=symbols,
        timeframe="1D",
        start=START,
        end=END,
    )

    result = validator.validate_multi(
        level=ValidationLevel.QUICK,
        data=multi_data,
        strategy=strategy,
        portfolio=portfolio,
    )

    return {
        "strategy": strategy_name,
        "verdict": result.verdict,
        "passed": result.passed,
        "avg_train_sharpe": round(result.avg_train_sharpe, 4),
        "avg_test_sharpe": round(result.avg_test_sharpe, 4),
        "sharpe_decay": round(result.avg_sharpe_decay, 4),
        "consistency": round(result.consistency_ratio, 4),
        "overfit_probability": round(result.overfit_probability, 4),
        "failure_reasons": list(result.failure_reasons),
        "fold_results": [fold_to_dict(f) for f in result.fold_results],
        "computation_time_seconds": round(result.computation_time_seconds, 2),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # 8-symbol 데이터 사용 시도, 실패 시 5-symbol fallback
    service = MarketDataService()
    symbols = SYMBOLS_8

    try:
        service.get_multi(
            symbols=SYMBOLS_8,
            timeframe="1D",
            start=START,
            end=END,
        )
        logger.info("8-symbol data available")
    except Exception:
        logger.warning("8-symbol data not fully available, falling back to 5 symbols")
        symbols = SYMBOLS_5

    total = len(GATE1_PASS_STRATEGIES)
    logger.info(f"Gate 2 Validation: {total} strategies × {len(symbols)}-coin EW portfolio")
    logger.info(f"Period: {START.date()} ~ {END.date()}, Level: QUICK (IS/OOS 70/30)")

    validator = TieredValidator()
    results: dict[str, Any] = {}
    errors: list[str] = []
    t0 = time.perf_counter()

    for i, strategy_name in enumerate(GATE1_PASS_STRATEGIES, 1):
        logger.info(f"[{i}/{total}] Validating {strategy_name}...")
        try:
            entry = validate_strategy(validator, service, strategy_name, symbols)
            results[strategy_name] = entry
            logger.info(
                f"  → {entry['verdict']} | IS Sharpe: {entry['avg_train_sharpe']:.2f}, "
                f"OOS Sharpe: {entry['avg_test_sharpe']:.2f}, "
                f"Decay: {entry['sharpe_decay']:.1%}"
            )
        except Exception:
            logger.exception(f"FAIL: {strategy_name}")
            errors.append(strategy_name)

    elapsed = time.perf_counter() - t0

    # Pass criteria description
    pass_criteria = {
        "level": "QUICK",
        "min_oos_sharpe": 0.5,
        "max_sharpe_decay": 0.30,
        "mode": "multi-asset",
    }

    output = {
        "meta": {
            "run_date": datetime.now(tz=UTC).isoformat(),
            "validation_level": "quick",
            "symbols": symbols,
            "period": f"{START.date()} ~ {END.date()}",
            "initial_capital": str(INITIAL_CAPITAL),
            "pass_criteria": pass_criteria,
            "total_strategies": total,
            "successful": total - len(errors),
            "failed": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "errors": errors,
        },
        "results": results,
    }

    output_path = RESULTS_DIR / "gate2_validation_results.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    # Summary
    logger.info(f"Done in {elapsed:.1f}s — {total - len(errors)}/{total} succeeded")
    logger.info(f"Results saved to {output_path}")

    pass_count = sum(1 for r in results.values() if r["passed"])
    logger.info(f"Results: {pass_count} PASS, {len(results) - pass_count} FAIL/WARN")

    if errors:
        logger.warning(f"Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
