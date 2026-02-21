"""Phase 4 / Phase 5 stability runner logic for CLI integration.

Phase 4: 5-coin x 6-year single-asset backtest (심볼 간 병렬 지원)
Phase 5 stability: Parameter stability validation (plateau + +/-20%)
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.table import Table

if TYPE_CHECKING:
    from rich.console import Console

    from src.backtest.engine import BacktestEngine
    from src.data.service import MarketDataService

# =============================================================================
# Constants
# =============================================================================

_DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
_DEFAULT_START = datetime(2020, 1, 1, tzinfo=UTC)
_DEFAULT_END = datetime(2025, 12, 31, tzinfo=UTC)
_DEFAULT_CAPITAL = Decimal(100_000)
_RESULTS_DIR = Path("results")

# Phase 4 PASS thresholds
_P4_MIN_SHARPE = 1.0
_P4_MIN_CAGR = 20.0
_P4_MAX_MDD = 40.0
_P4_MIN_TRADES = 50

# Phase 5 plateau detection
_PLATEAU_MIN_COUNT = 3
_PLATEAU_THRESHOLD_RATIO = 0.8

# Phase 5 전략별 파라미터 설정 (P4 PASS strategies)
P5_STRATEGIES: dict[str, dict[str, Any]] = {
    "kama": {
        "best_asset": "DOGE/USDT",
        "baseline": {
            "er_lookback": 10,
            "fast_period": 2,
            "slow_period": 30,
            "atr_multiplier": 1.5,
            "vol_target": 0.30,
            "short_mode": 1,
        },
        "sweeps": {
            "er_lookback": [5, 7, 8, 9, 10, 11, 12, 14, 16, 20],
            "slow_period": [20, 24, 26, 28, 30, 32, 34, 36, 40, 50],
            "vol_target": [0.15, 0.20, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40, 0.50],
        },
    },
    "donchian-ensemble": {
        "best_asset": "ETH/USDT",
        "baseline": {
            "vol_target": 0.40,
            "atr_period": 14,
            "short_mode": 0,
        },
        "sweeps": {
            "vol_target": [0.20, 0.25, 0.30, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.55],
            "atr_period": [8, 10, 11, 12, 14, 16, 17, 18, 20, 25],
        },
    },
    "ttm-squeeze": {
        "best_asset": "BTC/USDT",
        "baseline": {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_mult": 1.5,
            "mom_period": 20,
            "exit_sma_period": 21,
            "vol_target": 0.40,
        },
        "sweeps": {
            "bb_period": [12, 14, 16, 18, 20, 22, 24, 26, 30],
            "kc_mult": [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0],
            "vol_target": [0.20, 0.25, 0.30, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.55],
        },
    },
    "max-min": {
        "best_asset": "DOGE/USDT",
        "baseline": {
            "lookback": 10,
            "max_weight": 0.5,
            "min_weight": 0.5,
            "vol_target": 0.30,
        },
        "sweeps": {
            "lookback": [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20],
            "vol_target": [0.15, 0.20, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40, 0.50],
            "max_weight": [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7],
        },
    },
    "bb-rsi": {
        "best_asset": "SOL/USDT",
        "baseline": {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "bb_weight": 0.6,
            "rsi_weight": 0.4,
            "vol_target": 0.20,
        },
        "sweeps": {
            "bb_period": [12, 14, 16, 18, 20, 22, 24, 26, 30],
            "rsi_period": [8, 10, 11, 12, 14, 16, 17, 18, 20],
            "vol_target": [0.10, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25, 0.30],
            "bb_weight": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
    },
    "ctrend": {
        "best_asset": "SOL/USDT",
        "baseline": {
            "training_window": 252,
            "prediction_horizon": 5,
            "alpha": 0.5,
            "vol_window": 30,
            "vol_target": 0.35,
            "short_mode": 2,
        },
        "sweeps": {
            "training_window": [126, 150, 175, 200, 225, 252, 280, 315, 350, 400],
            "prediction_horizon": [1, 2, 3, 4, 5, 6, 7, 10, 14, 21],
            "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "vol_target": [0.15, 0.20, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.50, 0.60],
        },
    },
    "anchor-mom": {
        "best_asset": "DOGE/USDT",
        "timeframe": "12h",
        "baseline": {
            "nearness_lookback": 60,
            "mom_lookback": 30,
            "strong_nearness": 0.95,
            "vol_target": 0.35,
            "short_mode": 1,
        },
        "sweeps": {
            "nearness_lookback": [20, 30, 40, 48, 54, 60, 66, 72, 90, 120],
            "mom_lookback": [10, 18, 24, 27, 30, 33, 36, 45, 60, 80],
            "strong_nearness": [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
            "vol_target": [0.15, 0.20, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.50, 0.60],
        },
    },
}

# 가중치 쌍 (한쪽을 스윕하면 다른 쪽은 1 - value)
P5_WEIGHT_PAIRS: dict[str, dict[str, str]] = {
    "max-min": {"max_weight": "min_weight"},
    "bb-rsi": {"bb_weight": "rsi_weight"},
}

# =============================================================================
# Shared helpers
# =============================================================================

_TF_MAP: dict[str, str] = {
    "1D": "1D",
    "4H": "4h",
    "3H": "3h",
    "2H": "2h",
    "1H": "1h",
    "12H": "12h",
    "6H": "6h",
}


def resolve_timeframe(strategy_name: str) -> str:
    """전략 YAML의 meta.timeframe → MarketDataRequest용 TF 문자열.

    Examples:
        "1D" → "1D"
        "4H (annualization_factor=2190)" → "4h"
        "12H" → "12h"
    """
    from src.pipeline.store import StrategyStore

    store = StrategyStore()
    if not store.exists(strategy_name):
        return "1D"

    record = store.load(strategy_name)
    raw_tf = record.meta.timeframe.split("(")[0].strip().upper()
    return _TF_MAP.get(raw_tf, "1D")


def _create_portfolio(strategy_name: str, capital: Decimal) -> Any:
    """전략별 권장 설정으로 Portfolio 생성."""
    from src.portfolio import Portfolio, PortfolioManagerConfig
    from src.strategy import get_strategy

    strategy_cls = get_strategy(strategy_name)
    rec = strategy_cls.recommended_config()
    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    return Portfolio.create(initial_capital=capital, config=pm_config)


# =============================================================================
# Phase 4 Logic
# =============================================================================

_MAX_WORKERS = 4


def _phase4_worker(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    capital: Decimal,
) -> dict[str, Any] | None:
    """프로세스 풀 워커 — 단일 전략+심볼 백테스트 (pickling 가능).

    프로세스별 새로운 Engine/Service 인스턴스 생성.
    """
    from src.backtest.engine import BacktestEngine
    from src.data.service import MarketDataService

    engine = BacktestEngine()
    service = MarketDataService()
    return _run_phase4_single(
        engine, service, strategy_name, symbol, timeframe, start, end, capital
    )


def _run_phase4_single(
    engine: BacktestEngine,
    service: MarketDataService,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    capital: Decimal,
) -> dict[str, Any] | None:
    """단일 전략+심볼 백테스트 실행."""
    from src.backtest.request import BacktestRequest
    from src.data.market_data import MarketDataRequest
    from src.strategy import get_strategy

    try:
        data = service.get(
            MarketDataRequest(symbol=symbol, timeframe=timeframe, start=start, end=end),
        )
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
        portfolio = _create_portfolio(strategy_name, capital)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        result = engine.run(request)

        m = result.metrics
        entry: dict[str, Any] = {
            "strategy": strategy_name,
            "symbol": symbol,
            "sharpe_ratio": m.sharpe_ratio,
            "cagr": m.cagr,
            "max_drawdown": m.max_drawdown,
            "total_trades": m.total_trades,
            "total_return": m.total_return,
            "profit_factor": m.profit_factor,
            "win_rate": m.win_rate,
            "sortino_ratio": m.sortino_ratio,
            "calmar_ratio": m.calmar_ratio,
            "volatility": m.volatility,
        }
        if result.benchmark is not None:
            entry["alpha"] = result.benchmark.alpha
            entry["beta"] = result.benchmark.beta
    except Exception:
        logger.exception(f"FAIL: {strategy_name} / {symbol}")
        return None
    else:
        return entry


def _update_yaml_p4(strategy_name: str, results: list[dict[str, Any]]) -> None:
    """Phase 4 결과를 전략 YAML에 기록."""
    from src.pipeline.models import AssetMetrics, PhaseId, PhaseVerdict, StrategyStatus
    from src.pipeline.store import StrategyStore

    store = StrategyStore()
    if not store.exists(strategy_name):
        return

    sorted_results = sorted(results, key=lambda x: x.get("sharpe_ratio") or 0, reverse=True)
    best = sorted_results[0]
    best_sharpe = best.get("sharpe_ratio") or 0
    best_cagr = (best.get("cagr") or 0) * 100
    best_mdd = abs(best.get("max_drawdown") or 0)
    best_trades = best.get("total_trades") or 0

    verdict = (
        PhaseVerdict.PASS
        if (
            best_sharpe > _P4_MIN_SHARPE
            and best_cagr > _P4_MIN_CAGR
            and best_mdd < _P4_MAX_MDD
            and best_trades > _P4_MIN_TRADES
        )
        else PhaseVerdict.FAIL
    )

    details = {
        "best_asset": best["symbol"],
        "sharpe": round(best_sharpe, 2),
        "cagr": round(best_cagr, 1),
        "mdd": round(best_mdd, 1),
        "trades": best_trades,
    }
    rationale = (
        f"{best['symbol']} Sharpe {best_sharpe:.2f}, CAGR {best_cagr:+.1f}%, MDD -{best_mdd:.1f}%"
    )

    store.record_phase(strategy_name, PhaseId.P4, verdict, details=details, rationale=rationale)

    metrics = [
        AssetMetrics(
            symbol=r["symbol"],
            sharpe=round(r.get("sharpe_ratio") or 0, 2),
            cagr=round((r.get("cagr") or 0) * 100, 1),
            mdd=round(abs(r.get("max_drawdown") or 0), 1),
            trades=r.get("total_trades") or 0,
            profit_factor=round(r["profit_factor"], 2) if r.get("profit_factor") else None,
            win_rate=round(r["win_rate"], 1) if r.get("win_rate") else None,
            sortino=round(r["sortino_ratio"], 2) if r.get("sortino_ratio") else None,
            calmar=round(r["calmar_ratio"], 2) if r.get("calmar_ratio") else None,
            alpha=round(r["alpha"], 1) if r.get("alpha") else None,
            beta=round(r["beta"], 2) if r.get("beta") else None,
        )
        for r in results
    ]
    store.set_asset_performance(strategy_name, metrics)

    if verdict == PhaseVerdict.FAIL:
        store.update_status(strategy_name, StrategyStatus.RETIRED)

    logger.info(f"  YAML updated: {strategy_name} P4 {verdict}")


def _run_symbols_parallel(
    sname: str,
    symbols: list[str],
    tf: str,
    start: datetime,
    end: datetime,
    capital_dec: Decimal,
    console: Console,
) -> list[dict[str, Any]]:
    """심볼 간 병렬 실행 (ProcessPoolExecutor)."""
    results: list[dict[str, Any]] = []
    n_workers = min(_MAX_WORKERS, len(symbols))
    console.print(f"  [dim]Parallel mode: {n_workers} workers[/dim]")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_phase4_worker, sname, sym, tf, start, end, capital_dec): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                entry = future.result()
            except Exception:
                logger.exception(f"  Worker failed: {sname} / {sym}")
                continue
            if entry:
                results.append(entry)
                msg = f"    {sym}: Sharpe={entry['sharpe_ratio']:.2f} CAGR={entry['cagr']:.1f}% MDD={entry['max_drawdown']:.1f}% Trades={entry['total_trades']}"
                logger.info(msg)
    return results


def _run_symbols_sequential(
    sname: str,
    symbols: list[str],
    tf: str,
    start: datetime,
    end: datetime,
    capital_dec: Decimal,
) -> list[dict[str, Any]]:
    """심볼 간 순차 실행 (fallback)."""
    from src.backtest.engine import BacktestEngine
    from src.data.service import MarketDataService

    results: list[dict[str, Any]] = []
    engine = BacktestEngine()
    service = MarketDataService()
    for sym in symbols:
        logger.info(f"  {sname} / {sym}")
        entry = _run_phase4_single(engine, service, sname, sym, tf, start, end, capital_dec)
        if entry:
            results.append(entry)
            msg = f"    Sharpe={entry['sharpe_ratio']:.2f} CAGR={entry['cagr']:.1f}% MDD={entry['max_drawdown']:.1f}% Trades={entry['total_trades']}"
            logger.info(msg)
    return results


def run_phase4(
    strategies: list[str],
    symbols: list[str],
    start: datetime,
    end: datetime,
    capital: int,
    save_json: bool,
    console: Console,
    *,
    parallel: bool = True,
) -> None:
    """Phase 4 전체 실행: 전략별 x 심볼별 백테스트 + Rich 출력 + YAML 업데이트.

    Args:
        parallel: True이면 심볼 간 ProcessPoolExecutor 병렬 실행.
    """
    _RESULTS_DIR.mkdir(exist_ok=True)

    capital_dec = Decimal(capital)
    t0 = time.perf_counter()
    all_results: dict[str, list[dict[str, Any]]] = {}

    for sname in strategies:
        tf = resolve_timeframe(sname)
        console.rule(f"[bold]{sname}[/] (TF={tf})")

        if parallel and len(symbols) > 1:
            results = _run_symbols_parallel(sname, symbols, tf, start, end, capital_dec, console)
        else:
            results = _run_symbols_sequential(sname, symbols, tf, start, end, capital_dec)

        all_results[sname] = results

    elapsed = time.perf_counter() - t0

    # Save JSON
    if save_json:
        output_path = _RESULTS_DIR / "phase4_pipeline_results.json"
        output = {
            "meta": {
                "run_date": datetime.now(tz=UTC).isoformat(),
                "period": f"{start.date()} ~ {end.date()}",
                "initial_capital": str(capital_dec),
                "symbols": symbols,
                "elapsed_seconds": round(elapsed, 1),
            },
            "results": all_results,
        }
        output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]Results saved: {output_path}[/dim]")

    # Update YAML
    for sname, results in all_results.items():
        if results:
            _update_yaml_p4(sname, results)

    # Rich Table output
    for sname, results in all_results.items():
        table = Table(title=f"Phase 4: {sname}")
        table.add_column("Symbol", style="cyan")
        table.add_column("Sharpe", justify="right")
        table.add_column("CAGR", justify="right")
        table.add_column("MDD", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("PF", justify="right")
        table.add_column("Alpha", justify="right")
        table.add_column("Beta", justify="right")

        for r in sorted(results, key=lambda x: x.get("sharpe_ratio") or 0, reverse=True):
            alpha_val = r.get("alpha")
            beta_val = r.get("beta")
            table.add_row(
                r["symbol"],
                f"{r['sharpe_ratio']:.2f}",
                f"{r['cagr']:.1f}%",
                f"-{r['max_drawdown']:.1f}%",
                str(r["total_trades"]),
                f"{r['profit_factor']:.2f}" if r.get("profit_factor") else "N/A",
                f"{alpha_val:.1f}%" if alpha_val is not None else "N/A",
                f"{beta_val:.2f}" if beta_val is not None else "N/A",
            )

        console.print(table)

    console.print(f"\n[dim]Elapsed: {elapsed:.1f}s[/dim]")


# =============================================================================
# Phase 5 Logic
# =============================================================================


def analyze_sweep(
    param_name: str,
    baseline_value: Any,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """스윕 결과 분석: 고원 + ±20% 안정성."""
    valid = [r for r in results if not math.isnan(r["sharpe_ratio"])]
    if not valid:
        return {
            "param": param_name,
            "baseline_value": baseline_value,
            "plateau_exists": False,
            "pm20_stable": False,
            "verdict": "FAIL",
            "reason": "No valid results",
        }

    sharpes = [r["sharpe_ratio"] for r in valid]
    values = [r["value"] for r in valid]
    best_sharpe = max(sharpes)

    # Plateau Detection: best Sharpe의 80% 이상인 값이 3개 이상
    plateau_threshold = best_sharpe * _PLATEAU_THRESHOLD_RATIO if best_sharpe > 0 else 0
    plateau_count = sum(1 for s in sharpes if s >= plateau_threshold)
    plateau_exists = plateau_count >= _PLATEAU_MIN_COUNT

    plateau_values = [v for v, s in zip(values, sharpes, strict=True) if s >= plateau_threshold]
    plateau_range = (min(plateau_values), max(plateau_values)) if plateau_values else (None, None)

    # ±20% Stability
    if isinstance(baseline_value, int | float):
        low = baseline_value * 0.8
        high = baseline_value * 1.2
        pm20_results = [r for r in valid if low <= r["value"] <= high]
        if pm20_results:
            pm20_sharpes = [r["sharpe_ratio"] for r in pm20_results]
            pm20_stable = all(s > 0 for s in pm20_sharpes)
            pm20_min_sharpe = min(pm20_sharpes)
            pm20_max_sharpe = max(pm20_sharpes)
        else:
            pm20_stable = False
            pm20_min_sharpe = None
            pm20_max_sharpe = None
    else:
        pm20_stable = True
        pm20_min_sharpe = None
        pm20_max_sharpe = None

    baseline_result = next((r for r in valid if r["value"] == baseline_value), None)
    baseline_sharpe = baseline_result["sharpe_ratio"] if baseline_result else None

    verdict = "PASS" if (plateau_exists and pm20_stable) else "FAIL"

    return {
        "param": param_name,
        "baseline_value": baseline_value,
        "baseline_sharpe": baseline_sharpe,
        "best_sharpe": best_sharpe,
        "plateau_exists": plateau_exists,
        "plateau_count": plateau_count,
        "plateau_threshold": round(plateau_threshold, 4),
        "plateau_range": plateau_range,
        "pm20_stable": pm20_stable,
        "pm20_min_sharpe": pm20_min_sharpe,
        "pm20_max_sharpe": pm20_max_sharpe,
        "verdict": verdict,
        "all_sharpes": list(zip(values, sharpes, strict=True)),
    }


def _run_one_at_a_time_sweep(
    engine: BacktestEngine,
    strategy_name: str,
    baseline: dict[str, Any],
    param_name: str,
    param_values: list[Any],
    data: Any,
    capital: Decimal,
) -> list[dict[str, Any]]:
    """한 파라미터만 변경하며 스윕 (나머지 고정)."""
    from src.backtest.request import BacktestRequest
    from src.strategy import get_strategy

    results: list[dict[str, Any]] = []
    weight_pair = P5_WEIGHT_PAIRS.get(strategy_name, {})

    for value in param_values:
        params = dict(baseline)
        params[param_name] = value

        if param_name in weight_pair:
            complement_name = weight_pair[param_name]
            params[complement_name] = round(1.0 - value, 6)

        try:
            strategy_cls = get_strategy(strategy_name)
            strategy = strategy_cls.from_params(**params)
            portfolio = _create_portfolio(strategy_name, capital)
            request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
            result = engine.run(request)
            results.append(
                {
                    "value": value,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "total_return": result.metrics.total_return,
                    "max_drawdown": result.metrics.max_drawdown,
                    "cagr": result.metrics.cagr,
                    "total_trades": result.metrics.total_trades,
                    "error": None,
                }
            )
        except Exception as e:
            logger.warning(f"  {param_name}={value}: {e}")
            results.append(
                {
                    "value": value,
                    "sharpe_ratio": float("nan"),
                    "total_return": float("nan"),
                    "max_drawdown": float("nan"),
                    "cagr": float("nan"),
                    "total_trades": 0,
                    "error": str(e),
                }
            )

    return results


def _phase5_sweep_worker(
    strategy_name: str,
    baseline: dict[str, Any],
    param_name: str,
    value: Any,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    capital: Decimal,
) -> dict[str, Any]:
    """프로세스 풀 워커 — 단일 sweep point 백테스트 (pickling 가능).

    프로세스별 새로운 Engine/Service 인스턴스 생성.
    """
    from src.backtest.engine import BacktestEngine
    from src.backtest.request import BacktestRequest
    from src.data.market_data import MarketDataRequest
    from src.data.service import MarketDataService
    from src.strategy import get_strategy

    engine = BacktestEngine()
    service = MarketDataService()

    params = dict(baseline)
    params[param_name] = value

    weight_pair = P5_WEIGHT_PAIRS.get(strategy_name, {})
    if param_name in weight_pair:
        complement_name = weight_pair[param_name]
        params[complement_name] = round(1.0 - value, 6)

    try:
        data = service.get(
            MarketDataRequest(symbol=symbol, timeframe=timeframe, start=start, end=end),
        )
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls.from_params(**params)
        portfolio = _create_portfolio(strategy_name, capital)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        result = engine.run(request)
    except Exception as e:
        logger.warning(f"  {param_name}={value}: {e}")
        return {
            "value": value,
            "sharpe_ratio": float("nan"),
            "total_return": float("nan"),
            "max_drawdown": float("nan"),
            "cagr": float("nan"),
            "total_trades": 0,
            "error": str(e),
        }
    else:
        return {
            "value": value,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "total_return": result.metrics.total_return,
            "max_drawdown": result.metrics.max_drawdown,
            "cagr": result.metrics.cagr,
            "total_trades": result.metrics.total_trades,
            "error": None,
        }


def _run_one_at_a_time_sweep_parallel(
    strategy_name: str,
    baseline: dict[str, Any],
    param_name: str,
    param_values: list[Any],
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    capital: Decimal,
) -> list[dict[str, Any]]:
    """sweep point 간 병렬 실행 (ProcessPoolExecutor)."""
    n_workers = min(_MAX_WORKERS, len(param_values))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _phase5_sweep_worker,
                strategy_name,
                baseline,
                param_name,
                value,
                symbol,
                timeframe,
                start,
                end,
                capital,
            ): value
            for value in param_values
        }
        results: list[dict[str, Any]] = []
        for future in as_completed(futures):
            val = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.warning(f"  {param_name}={val} worker error: {e}")
                results.append(
                    {
                        "value": val,
                        "sharpe_ratio": float("nan"),
                        "total_return": float("nan"),
                        "max_drawdown": float("nan"),
                        "cagr": float("nan"),
                        "total_trades": 0,
                        "error": str(e),
                    }
                )
    # Sort by value for deterministic ordering
    results.sort(key=lambda r: r["value"])
    return results


def _print_sweep_table(
    console: Console, strategy_name: str, analyses: list[dict[str, Any]]
) -> None:
    """스윕 결과 요약 테이블."""
    table = Table(title=f"Phase 5: {strategy_name}")
    table.add_column("Param", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Base Sharpe", justify="right")
    table.add_column("Best Sharpe", justify="right")
    table.add_column("Plateau", justify="center")
    table.add_column("Plateau #", justify="right")
    table.add_column("Plateau Range", justify="center")
    table.add_column("±20% Stable", justify="center")
    table.add_column("±20% Sharpe", justify="center")
    table.add_column("Verdict", justify="center")

    for a in analyses:
        plateau_range = (
            f"{a['plateau_range'][0]}~{a['plateau_range'][1]}"
            if a["plateau_range"][0] is not None
            else "N/A"
        )
        pm20_sharpe = (
            f"{a['pm20_min_sharpe']:.2f}~{a['pm20_max_sharpe']:.2f}"
            if a["pm20_min_sharpe"] is not None
            else "N/A"
        )
        base_sharpe = (
            f"{a['baseline_sharpe']:.2f}" if a.get("baseline_sharpe") is not None else "N/A"
        )
        verdict_style = "green bold" if a["verdict"] == "PASS" else "red bold"

        table.add_row(
            a["param"],
            str(a["baseline_value"]),
            base_sharpe,
            f"{a['best_sharpe']:.2f}",
            "[green]YES[/]" if a["plateau_exists"] else "[red]NO[/]",
            str(a["plateau_count"]),
            plateau_range,
            "[green]YES[/]" if a["pm20_stable"] else "[red]NO[/]",
            pm20_sharpe,
            f"[{verdict_style}]{a['verdict']}[/{verdict_style}]",
        )

    console.print(table)


def _print_detail_table(
    console: Console, param_name: str, results: list[dict[str, Any]], baseline_value: Any
) -> None:
    """파라미터별 상세 Sharpe 테이블."""
    table = Table(title=f"  {param_name} Sweep Detail", show_lines=False)
    table.add_column("Value", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("", justify="center")

    for r in results:
        is_baseline = r["value"] == baseline_value
        marker = "[yellow]*[/]" if is_baseline else ""
        sharpe_str = f"{r['sharpe_ratio']:.2f}" if not math.isnan(r["sharpe_ratio"]) else "ERR"
        ret_str = f"{r['total_return']:.1f}%" if not math.isnan(r["total_return"]) else "ERR"
        mdd_str = f"-{r['max_drawdown']:.1f}%" if not math.isnan(r["max_drawdown"]) else "ERR"
        cagr_str = f"{r['cagr']:.1f}%" if not math.isnan(r["cagr"]) else "ERR"

        table.add_row(
            str(r["value"]), sharpe_str, ret_str, mdd_str, cagr_str, str(r["total_trades"]), marker
        )

    console.print(table)


def _update_yaml_p5(result: dict[str, Any]) -> None:
    """Phase 5 결과를 YAML에 자동 기록."""
    from src.pipeline.models import PhaseId, PhaseVerdict, StrategyStatus
    from src.pipeline.store import StrategyStore

    store = StrategyStore()
    name = result["strategy"]
    if not store.exists(name):
        return

    verdict = PhaseVerdict(result["verdict"])
    details: dict[str, Any] = {
        "param_verdicts": result.get("param_verdicts", {}),
    }
    if result.get("fail_params"):
        details["fail_params"] = result["fail_params"]

    fail_str = ", ".join(result["fail_params"]) if result.get("fail_params") else "all stable"
    rationale = f"{verdict}: {fail_str}"

    store.record_phase(name, PhaseId.P5, verdict, details=details, rationale=rationale)
    if verdict == PhaseVerdict.FAIL:
        store.update_status(name, StrategyStatus.RETIRED)


def _load_p5_opt_config(strategy_name: str) -> dict[str, Any] | None:
    """P5 optimization JSON 결과에서 Phase 5 sweep 설정을 로드.

    Returns:
        P5 config dict (best_asset, baseline, sweeps) or None if not found.
    """
    from src.pipeline.store import StrategyStore

    json_path = _RESULTS_DIR / f"phase5_opt_{strategy_name}.json"
    if not json_path.exists():
        return None

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        g3_sweeps = raw.get("g3_sweeps")
        best_params = raw.get("optimization", {}).get("best_params")
    except Exception:
        logger.warning(f"Failed to load P5 optimization config for {strategy_name}")
        return None

    if not g3_sweeps or not best_params:
        return None

    # Get best_asset from pipeline YAML
    store = StrategyStore()
    if not store.exists(strategy_name):
        return None
    record = store.load(strategy_name)
    best_asset = record.best_asset
    if not best_asset:
        return None

    timeframe = resolve_timeframe(strategy_name)
    return {
        "best_asset": best_asset,
        "timeframe": timeframe,
        "baseline": best_params,
        "sweeps": g3_sweeps,
    }


def _run_strategy_sweeps(
    engine: BacktestEngine,
    strategy_name: str,
    config: dict[str, Any],
    data: Any,
    timeframe: str,
    capital: Decimal,
    *,
    use_parallel: bool,
    console: Console,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """단일 전략에 대해 모든 파라미터 sweep을 실행."""
    baseline = config["baseline"]
    sweeps = config["sweeps"]

    strategy_results: dict[str, Any] = {
        "best_asset": config["best_asset"],
        "baseline": baseline,
        "analyses": {},
        "raw_sweeps": {},
    }
    analyses: list[dict[str, Any]] = []
    n_runs = 0

    if use_parallel and len(sweeps) > 0:
        console.print("  [dim]Parallel sweep mode enabled[/dim]")

    for param_name, param_values in sweeps.items():
        logger.info(f"  Sweeping {param_name}: {param_values}")

        if use_parallel and len(param_values) > 1:
            sweep_results = _run_one_at_a_time_sweep_parallel(
                strategy_name=strategy_name,
                baseline=baseline,
                param_name=param_name,
                param_values=param_values,
                symbol=config["best_asset"],
                timeframe=timeframe,
                start=_DEFAULT_START,
                end=_DEFAULT_END,
                capital=capital,
            )
        else:
            sweep_results = _run_one_at_a_time_sweep(
                engine=engine,
                strategy_name=strategy_name,
                baseline=baseline,
                param_name=param_name,
                param_values=param_values,
                data=data,
                capital=capital,
            )
        n_runs += len(param_values)

        baseline_val = baseline.get(param_name)
        analysis = analyze_sweep(param_name, baseline_val, sweep_results)
        analyses.append(analysis)

        _print_detail_table(console, param_name, sweep_results, baseline_val)

        strategy_results["analyses"][param_name] = analysis
        strategy_results["raw_sweeps"][param_name] = sweep_results

    _print_sweep_table(console, strategy_name, analyses)
    return strategy_results, analyses, n_runs


def run_phase5_stability(
    strategies: list[str] | None,
    save_json: bool,
    console: Console,
    *,
    parallel: bool = True,
) -> None:
    """Phase 5 전체 실행: 파라미터 안정성 검증."""
    from src.backtest.engine import BacktestEngine
    from src.data.market_data import MarketDataRequest
    from src.data.service import MarketDataService

    _RESULTS_DIR.mkdir(exist_ok=True)

    # Build strategies_to_run: P5 opt JSON fallback → P5_STRATEGIES dict
    strategies_to_run: dict[str, dict[str, Any]] = {}
    target_names = strategies if strategies else list(P5_STRATEGIES.keys())

    for name in target_names:
        p5_opt_config = _load_p5_opt_config(name)
        if p5_opt_config is not None:
            strategies_to_run[name] = p5_opt_config
            logger.info(f"  {name}: using P5 optimized params")
        elif name in P5_STRATEGIES:
            strategies_to_run[name] = P5_STRATEGIES[name]
        elif strategies is not None:
            # Explicit request for a strategy not in either source — try P5 opt
            logger.warning(f"  {name}: not in P5_STRATEGIES and no P5 optimization JSON found")

    if not strategies_to_run:
        console.print(
            "[red]No matching strategies. Run phase5-run first or check P5_STRATEGIES.[/]"
        )
        return

    engine = BacktestEngine()
    service = MarketDataService()
    capital = _DEFAULT_CAPITAL
    t0 = time.perf_counter()

    all_results: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    total_runs = 0

    for strategy_name, config in strategies_to_run.items():
        console.rule(f"[bold]{strategy_name}[/] ({config['best_asset']})")

        timeframe = config.get("timeframe", "1D")
        data = service.get(
            MarketDataRequest(
                symbol=config["best_asset"],
                timeframe=timeframe,
                start=_DEFAULT_START,
                end=_DEFAULT_END,
            ),
        )
        strategy_results, analyses, n_runs = _run_strategy_sweeps(
            engine=engine,
            strategy_name=strategy_name,
            config=config,
            data=data,
            timeframe=timeframe,
            capital=capital,
            use_parallel=parallel,
            console=console,
        )
        total_runs += n_runs

        all_pass = all(a["verdict"] == "PASS" for a in analyses)
        overall = "PASS" if all_pass else "FAIL"
        fail_params = [a["param"] for a in analyses if a["verdict"] != "PASS"]

        strategy_results["overall_verdict"] = overall
        strategy_results["fail_params"] = fail_params
        all_results[strategy_name] = strategy_results

        verdict_style = "[green bold]PASS[/]" if overall == "PASS" else "[red bold]FAIL[/]"
        console.print(f"\n  Phase 5 Overall: {verdict_style}")
        if fail_params:
            console.print(f"  Failed params: {', '.join(fail_params)}")
        console.print()

        summary.append(
            {
                "strategy": strategy_name,
                "best_asset": config["best_asset"],
                "verdict": overall,
                "fail_params": fail_params,
                "param_verdicts": {a["param"]: a["verdict"] for a in analyses},
            }
        )

    elapsed = time.perf_counter() - t0

    # Final summary table
    console.rule("[bold]Phase 5 Summary[/]")
    final_table = Table(title="Phase 5 Parameter Stability Results")
    final_table.add_column("Strategy", style="cyan")
    final_table.add_column("Best Asset")
    final_table.add_column("Params Tested", justify="right")
    final_table.add_column("Verdict", justify="center")
    final_table.add_column("Failed Params")

    for s in summary:
        verdict_style = "green bold" if s["verdict"] == "PASS" else "red bold"
        n_params = len(s["param_verdicts"])
        n_pass = sum(1 for v in s["param_verdicts"].values() if v == "PASS")
        final_table.add_row(
            s["strategy"],
            s["best_asset"],
            f"{n_pass}/{n_params}",
            f"[{verdict_style}]{s['verdict']}[/{verdict_style}]",
            ", ".join(s["fail_params"]) if s["fail_params"] else "-",
        )

    console.print(final_table)
    console.print(f"\nTotal: {total_runs} backtests in {elapsed:.1f}s")

    # Save JSON
    if save_json:
        output_path = _RESULTS_DIR / "phase5_param_sweep.json"

        def json_default(obj: Any) -> Any:
            if isinstance(obj, float) and math.isnan(obj):
                return None
            return str(obj)

        output = {
            "meta": {
                "run_date": datetime.now(tz=UTC).isoformat(),
                "period": f"{_DEFAULT_START.date()} ~ {_DEFAULT_END.date()}",
                "initial_capital": str(capital),
                "total_runs": total_runs,
                "elapsed_seconds": round(elapsed, 1),
            },
            "summary": summary,
            "details": all_results,
        }
        output_path.write_text(json.dumps(output, indent=2, default=json_default), encoding="utf-8")
        console.print(f"[dim]Results saved: {output_path}[/dim]")

    # Update YAML
    for s in summary:
        _update_yaml_p5(s)
