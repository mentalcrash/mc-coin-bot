"""EDA vs VBT 8-Asset TSMOM 백테스트 비교.

최적 설정으로 VBT(벡터 백테스트)와 EDA(이벤트 기반) 결과를 비교합니다.
"""

import asyncio
from datetime import UTC, datetime

from loguru import logger

from src.backtest.engine import BacktestEngine
from src.backtest.request import MultiAssetBacktestRequest
from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.market_data import MarketDataRequest, MultiSymbolData
from src.data.service import MarketDataService
from src.eda.runner import EDARunner
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.portfolio.portfolio import Portfolio
from src.strategy.tsmom import TSMOMConfig, TSMOMStrategy

# =========================================================================
# 최적 설정 (문서 기반)
# =========================================================================
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)
TIMEFRAME = "1d"
INITIAL_CAPITAL = 100_000.0

# TSMOM 전략 설정
TSMOM_CONFIG = TSMOMConfig(
    lookback=30,
    vol_window=30,
    vol_target=0.35,
)

# PM 설정 (최적)
PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=2.0,
    rebalance_threshold=0.10,
    system_stop_loss=0.10,
    use_trailing_stop=True,
    trailing_stop_atr_multiplier=3.0,
    use_intrabar_stop=True,
    cost_model=CostModel.binance_futures(),
)


def load_multi_data() -> MultiSymbolData:
    """8개 심볼 Silver 1D 데이터 로드."""
    settings = get_settings()
    service = MarketDataService(settings)
    ohlcv_dict: dict[str, object] = {}
    loaded_symbols: list[str] = []

    for symbol in SYMBOLS:
        request = MarketDataRequest(
            symbol=symbol,
            timeframe=TIMEFRAME,
            start=START,
            end=END,
        )
        data = service.get(request)
        ohlcv_dict[symbol] = data.ohlcv
        loaded_symbols.append(symbol)
        logger.info("Loaded {} bars for {}", len(data.ohlcv), symbol)

    return MultiSymbolData(
        symbols=loaded_symbols,
        timeframe=TIMEFRAME,
        start=START,
        end=END,
        ohlcv=ohlcv_dict,  # type: ignore[arg-type]
    )


def run_vbt_backtest(data: MultiSymbolData) -> dict[str, float]:
    """VBT 멀티에셋 백테스트 실행."""
    strategy = TSMOMStrategy(TSMOM_CONFIG)
    portfolio = Portfolio.create(initial_capital=INITIAL_CAPITAL, config=PM_CONFIG)

    request = MultiAssetBacktestRequest(
        data=data,
        strategy=strategy,
        portfolio=portfolio,
        weights=None,  # Equal Weight
    )

    engine = BacktestEngine()
    result = engine.run_multi(request)
    m = result.portfolio_metrics

    return {
        "total_return": m.total_return,
        "cagr": m.cagr,
        "sharpe": m.sharpe_ratio,
        "max_drawdown": m.max_drawdown,
        "win_rate": m.win_rate,
        "total_trades": float(m.total_trades),
        "winning_trades": float(m.winning_trades),
        "losing_trades": float(m.losing_trades),
        "volatility": m.volatility or 0.0,
        "profit_factor": m.profit_factor or 0.0,
    }


async def run_eda_backtest(data: MultiSymbolData) -> dict[str, float]:
    """EDA 멀티에셋 백테스트 실행."""
    strategy = TSMOMStrategy(TSMOM_CONFIG)
    weights = {s: 1.0 / len(SYMBOLS) for s in SYMBOLS}

    runner = EDARunner(
        strategy=strategy,
        data=data,
        config=PM_CONFIG,
        initial_capital=INITIAL_CAPITAL,
        asset_weights=weights,
    )

    m = await runner.run()

    return {
        "total_return": m.total_return,
        "cagr": m.cagr,
        "sharpe": m.sharpe_ratio,
        "max_drawdown": m.max_drawdown,
        "win_rate": m.win_rate,
        "total_trades": float(m.total_trades),
        "winning_trades": float(m.winning_trades),
        "losing_trades": float(m.losing_trades),
        "volatility": m.volatility or 0.0,
        "profit_factor": m.profit_factor or 0.0,
    }


def print_comparison(vbt: dict[str, float], eda: dict[str, float]) -> None:
    """결과 비교 테이블 출력."""
    print("\n" + "=" * 70)
    print("  EDA vs VBT  |  8-Asset TSMOM (vol_target=0.35, EW)")
    print("=" * 70)
    print(f"{'Metric':<20} {'VBT':>15} {'EDA':>15} {'Delta':>15}")
    print("-" * 70)

    metrics = [
        ("Total Return %", "total_return", ".2f"),
        ("CAGR %", "cagr", ".2f"),
        ("Sharpe Ratio", "sharpe", ".4f"),
        ("Max Drawdown %", "max_drawdown", ".2f"),
        ("Win Rate %", "win_rate", ".1f"),
        ("Total Trades", "total_trades", ".0f"),
        ("Winning Trades", "winning_trades", ".0f"),
        ("Losing Trades", "losing_trades", ".0f"),
        ("Volatility %", "volatility", ".2f"),
        ("Profit Factor", "profit_factor", ".4f"),
    ]

    for label, key, fmt in metrics:
        v = vbt[key]
        e = eda[key]
        delta = e - v
        sign = "+" if delta > 0 else ""
        print(f"{label:<20} {v:>15{fmt}} {e:>15{fmt}} {sign}{delta:>14{fmt}}")

    print("=" * 70)

    # 방향 일치 체크
    vbt_positive = vbt["total_return"] > 0
    eda_positive = eda["total_return"] > 0
    direction_match = vbt_positive == eda_positive
    print(f"\nReturn direction match: {'PASS' if direction_match else 'FAIL'}")
    print(f"  VBT: {'positive' if vbt_positive else 'negative'}")
    print(f"  EDA: {'positive' if eda_positive else 'negative'}")

    # Sharpe 비교
    sharpe_diff = abs(vbt["sharpe"] - eda["sharpe"])
    print(f"\nSharpe difference: {sharpe_diff:.4f}")
    if sharpe_diff < 0.5:
        print("  -> Acceptable (within 0.5 range)")
    else:
        print("  -> WARNING: Large divergence")

    # Trade count 비교
    trade_ratio = eda["total_trades"] / max(vbt["total_trades"], 1)
    print(f"\nTrade count ratio (EDA/VBT): {trade_ratio:.2f}x")


def main() -> None:
    """메인 실행."""
    setup_logger()

    print("Loading 8-asset Silver 1D data...")
    data = load_multi_data()

    print("\n--- Running VBT Backtest ---")
    vbt_result = run_vbt_backtest(data)

    print("\n--- Running EDA Backtest ---")
    eda_result = asyncio.run(run_eda_backtest(data))

    print_comparison(vbt_result, eda_result)


if __name__ == "__main__":
    main()
