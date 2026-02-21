"""P4A 일괄 백테스트 실행 스크립트.

3개 전략 x 5개 코인 = 15개 백테스트를 순차 실행하고 결과를 수집합니다.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.config.settings import get_settings
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.strategy import get_strategy

STRATEGIES = ["accel-vol-trend", "complexity-trend", "vol-surface-mom"]
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
CAPITAL = 100_000.0


def build_portfolio() -> Portfolio:
    """표준 P4A 포트폴리오 설정."""
    cost_model = CostModel(
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage=0.0005,
        funding_rate_8h=0.0001,
        market_impact=0.0002,
    )
    config = PortfolioManagerConfig(
        max_leverage_cap=2.0,
        rebalance_threshold=0.10,
        system_stop_loss=0.10,
        use_trailing_stop=True,
        trailing_stop_atr_multiplier=3.0,
        cost_model=cost_model,
    )
    return Portfolio.create(initial_capital=CAPITAL, config=config)


def run_single(
    strategy_name: str,
    symbol: str,
    data_service: MarketDataService,
    engine: BacktestEngine,
    portfolio: Portfolio,
) -> dict[str, Any]:
    """단일 백테스트 실행 및 결과 반환."""
    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls()

    data_request = MarketDataRequest(
        symbol=symbol,
        timeframe="1D",
        start=START,
        end=END,
    )

    try:
        data = data_service.get(data_request)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    request = BacktestRequest(
        data=data,
        strategy=strategy,
        portfolio=portfolio,
    )

    try:
        result = engine.run(request)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    m = result.metrics
    b = result.benchmark
    return {
        "symbol": symbol,
        "sharpe": round(m.sharpe_ratio, 4),
        "sortino": round(m.sortino_ratio, 4),
        "calmar": round(m.calmar_ratio, 4),
        "cagr": round(m.cagr, 2),
        "mdd": round(m.max_drawdown, 2),
        "total_return": round(m.total_return, 2),
        "trades": m.total_trades,
        "win_rate": round(m.win_rate, 1),
        "profit_factor": round(m.profit_factor, 2),
        "alpha": round(b.alpha, 2) if b else None,
        "beta": round(b.beta, 4) if b else None,
    }


def main() -> None:
    """메인 실행."""
    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()
    portfolio = build_portfolio()

    all_results: dict[str, list[dict[str, Any]]] = {}

    for strat in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strat}")
        print(f"{'='*60}")

        results = []
        for symbol in SYMBOLS:
            print(f"  Running {strat} on {symbol}...", end=" ", flush=True)
            r = run_single(strat, symbol, data_service, engine, portfolio)
            if "error" in r:
                print(f"ERROR: {r['error']}")
            else:
                print(
                    f"Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']:+.1f}%, "
                    f"MDD={r['mdd']:.1f}%, Trades={r['trades']}"
                )
            results.append(r)

        all_results[strat] = results

    # 결과 출력
    print("\n\n" + "=" * 80)
    print("  P4A BULK BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    for strat, results in all_results.items():
        print(f"\n--- {strat} ---")
        print(
            f"{'Symbol':<12} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} "
            f"{'Trades':>8} {'WinRate':>8} {'PF':>8} {'Alpha':>8} {'Beta':>8}"
        )
        print("-" * 80)
        for r in results:
            if "error" in r:
                print(f"{r['symbol']:<12} ERROR: {r['error']}")
            else:
                print(
                    f"{r['symbol']:<12} {r['sharpe']:>8.2f} {r['cagr']:>+7.1f}% "
                    f"{r['mdd']:>7.1f}% {r['trades']:>8} {r['win_rate']:>7.1f}% "
                    f"{r['profit_factor']:>7.2f} {r['alpha']:>+7.1f}% {r['beta']:>8.4f}"
                )

        # Best asset
        valid = [r for r in results if "error" not in r]
        if valid:
            best = max(valid, key=lambda x: x["sharpe"])
            print(f"\n  Best Asset: {best['symbol']} (Sharpe={best['sharpe']:.2f})")

    # JSON 저장
    output_path = Path("reports/p4a_bulk_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
