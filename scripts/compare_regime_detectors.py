"""Regime Detector 비교 평가 스크립트.

8가지 감지기 구성을 동일 데이터에서 비교하여 최적 조합을 선정합니다.

비교 대상:
    1. rule_only: Rule-Based 단독
    2. hmm_only: HMM 단독 (expanding)
    3. hmm_sliding: HMM + sliding window + decay
    4. vol_only: Vol-Structure 단독
    5. msar_only: MSAR 단독
    6. ensemble_3det: Rule(0.4)+HMM(0.35)+Vol(0.25) 고정
    7. ensemble_4det: Rule+HMM+Vol+MSAR 고정
    8. ensemble_meta: Meta-learner stacking

평가 지표:
    - Forward Return Sharpe (10d) by regime
    - Directional Accuracy (5d)
    - Regime Stability (avg duration, transition count)
    - Economic Value (Regime-Adaptive Sharpe)
    - Probability Calibration
    - Detector간 일치도 (pairwise agreement)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.regime import (
    EnsembleRegimeDetector,
    EnsembleRegimeDetectorConfig,
    HMMDetectorConfig,
    MetaLearnerConfig,
    MSARDetectorConfig,
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeLabel,
    VolStructureDetectorConfig,
)
from src.regime.ensemble import SKLEARN_AVAILABLE
from src.regime.hmm_detector import HMM_AVAILABLE
from src.regime.msar_detector import MSAR_AVAILABLE
from src.regime.vol_detector import VolStructureDetector

# ── 설정 ──
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)
TIMEFRAME = "1D"
ANNUALIZATION = 365


def load_data(symbol: str) -> pd.DataFrame:
    """Silver 데이터 로드."""
    svc = MarketDataService()
    request = MarketDataRequest(symbol=symbol, timeframe=TIMEFRAME, start=START, end=END)
    ds = svc.get(request)
    return ds.ohlcv


# ── 감지기 구성 ──


def build_configs() -> dict[str, Any]:
    """8가지 감지기 구성 생성.

    Returns:
        {name: (detector_or_factory, is_ensemble)} dictionary
    """
    configs: dict[str, Any] = {}

    # 1. Rule-only
    configs["rule_only"] = ("rule", RegimeDetector(RegimeDetectorConfig()))

    # 2. HMM-only (expanding)
    if HMM_AVAILABLE:
        configs["hmm_only"] = (
            "ensemble",
            EnsembleRegimeDetector(
                EnsembleRegimeDetectorConfig(
                    hmm=HMMDetectorConfig(min_train_window=252, retrain_interval=21, n_iter=100),
                    weight_rule_based=0.0,
                    weight_hmm=1.0,
                )
            ),
        )

    # 3. HMM sliding + decay
    if HMM_AVAILABLE:
        configs["hmm_sliding"] = (
            "ensemble",
            EnsembleRegimeDetector(
                EnsembleRegimeDetectorConfig(
                    hmm=HMMDetectorConfig(
                        min_train_window=252,
                        retrain_interval=21,
                        n_iter=100,
                        sliding_window=504,
                        decay_half_life=126,
                    ),
                    weight_rule_based=0.0,
                    weight_hmm=1.0,
                )
            ),
        )

    # 4. Vol-Structure only
    configs["vol_only"] = ("vol", VolStructureDetector(VolStructureDetectorConfig()))

    # 5. MSAR only
    if MSAR_AVAILABLE:
        configs["msar_only"] = (
            "ensemble",
            EnsembleRegimeDetector(
                EnsembleRegimeDetectorConfig(
                    msar=MSARDetectorConfig(
                        k_regimes=2,
                        order=1,
                        min_train_window=252,
                        retrain_interval=21,
                        sliding_window=504,
                        switching_ar=False,
                    ),
                    weight_rule_based=0.0,
                    weight_msar=1.0,
                )
            ),
        )

    # 6. Ensemble 3-det
    if HMM_AVAILABLE:
        configs["ensemble_3det"] = (
            "ensemble",
            EnsembleRegimeDetector(
                EnsembleRegimeDetectorConfig(
                    hmm=HMMDetectorConfig(min_train_window=252, retrain_interval=21, n_iter=100),
                    vol_structure=VolStructureDetectorConfig(),
                    weight_rule_based=0.40,
                    weight_hmm=0.35,
                    weight_vol_structure=0.25,
                    min_hold_bars=5,
                )
            ),
        )

    # 7. Ensemble 4-det
    if HMM_AVAILABLE and MSAR_AVAILABLE:
        configs["ensemble_4det"] = (
            "ensemble",
            EnsembleRegimeDetector(
                EnsembleRegimeDetectorConfig(
                    hmm=HMMDetectorConfig(min_train_window=252, retrain_interval=21, n_iter=100),
                    vol_structure=VolStructureDetectorConfig(),
                    msar=MSARDetectorConfig(
                        k_regimes=2,
                        order=1,
                        min_train_window=252,
                        retrain_interval=21,
                        sliding_window=504,
                        switching_ar=False,
                    ),
                    weight_rule_based=0.30,
                    weight_hmm=0.25,
                    weight_vol_structure=0.20,
                    weight_msar=0.25,
                    min_hold_bars=5,
                )
            ),
        )

    # 8. Meta-learner
    if SKLEARN_AVAILABLE:
        configs["ensemble_meta"] = (
            "ensemble",
            EnsembleRegimeDetector(
                EnsembleRegimeDetectorConfig(
                    vol_structure=VolStructureDetectorConfig(),
                    ensemble_method="meta_learner",
                    meta_learner=MetaLearnerConfig(
                        train_window=252,
                        retrain_interval=63,
                        forward_return_window=20,
                    ),
                    min_hold_bars=5,
                )
            ),
        )

    return configs


# ── 평가 함수 ──


def classify_with_detector(
    det_type: str,
    detector: Any,
    closes: pd.Series,
) -> pd.DataFrame:
    """감지기로 레짐 분류.

    Returns:
        DataFrame with p_trending, p_ranging, p_volatile, regime_label
    """
    if det_type == "rule":
        return detector.classify_series(closes)
    elif det_type == "vol":
        result = detector.classify_series(closes)
        # Vol detector는 regime_label이 없으므로 추가
        probs = result[["p_trending", "p_ranging", "p_volatile"]]
        valid_mask: pd.Series = probs.notna().all(axis=1)  # type: ignore[assignment]
        label_map = {
            "p_trending": RegimeLabel.TRENDING,
            "p_ranging": RegimeLabel.RANGING,
            "p_volatile": RegimeLabel.VOLATILE,
        }
        raw_labels = pd.Series(np.nan, index=closes.index, dtype=object)
        if valid_mask.any():  # type: ignore[truthy-bool]
            idx_max: pd.Series = probs[valid_mask].idxmax(axis=1)  # type: ignore[assignment]
            raw_labels[valid_mask] = idx_max.map(label_map)
        result["regime_label"] = raw_labels
        return result
    else:  # ensemble
        return detector.classify_series(closes)


def evaluate_config(
    closes: pd.Series,
    regime_df: pd.DataFrame,
) -> dict[str, float]:
    """단일 구성 평가 → 핵심 지표 딕셔너리."""
    daily_ret = closes.pct_change()

    # Forward Return Sharpe (10d) by regime
    fwd_ret_10 = closes.pct_change(10).shift(-10)

    metrics: dict[str, float] = {}

    for label in RegimeLabel:
        mask = regime_df["regime_label"] == label
        regime_fwd = fwd_ret_10[mask].dropna()
        if len(regime_fwd) > 5:
            sharpe = (
                float(regime_fwd.mean() / regime_fwd.std()) * np.sqrt(ANNUALIZATION / 10)
                if regime_fwd.std() > 0
                else 0.0
            )
            metrics[f"fwd10_sharpe_{label.value}"] = sharpe
        else:
            metrics[f"fwd10_sharpe_{label.value}"] = np.nan

    # Directional Accuracy (5d)
    fwd_ret_5 = closes.pct_change(5).shift(-5)
    trending_mask = regime_df["regime_label"] == RegimeLabel.TRENDING
    trending_fwd = fwd_ret_5[trending_mask].dropna()
    if len(trending_fwd) > 0:
        metrics["dir_accuracy_5d"] = float((trending_fwd.abs() > 0.03).mean()) * 100
    else:
        metrics["dir_accuracy_5d"] = np.nan

    # Regime Stability
    labels = regime_df["regime_label"].dropna()
    if len(labels) > 1:
        transitions = float((labels != labels.shift(1)).sum())
        groups = (labels != labels.shift(1)).cumsum()
        run_lengths = labels.groupby(groups).size()
        metrics["avg_duration"] = float(run_lengths.mean())
        metrics["transition_count"] = transitions
    else:
        metrics["avg_duration"] = np.nan
        metrics["transition_count"] = np.nan

    # Economic Value (Regime-Adaptive)
    weight = pd.Series(0.0, index=closes.index)
    weight[regime_df["regime_label"] == RegimeLabel.TRENDING] = 1.0
    weight[regime_df["regime_label"] == RegimeLabel.RANGING] = 0.3
    weight[regime_df["regime_label"] == RegimeLabel.VOLATILE] = 0.0
    adaptive_ret = daily_ret * weight
    ret_clean = adaptive_ret.dropna()
    if len(ret_clean) > 2 and ret_clean.std() > 0:
        metrics["adaptive_sharpe"] = float(ret_clean.mean() / ret_clean.std() * np.sqrt(365))
    else:
        metrics["adaptive_sharpe"] = 0.0

    # Buy-Hold baseline
    bh_clean = daily_ret.dropna()
    if len(bh_clean) > 2 and bh_clean.std() > 0:
        metrics["bh_sharpe"] = float(bh_clean.mean() / bh_clean.std() * np.sqrt(365))
    else:
        metrics["bh_sharpe"] = 0.0

    return metrics


def compute_pairwise_agreement(
    results: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """감지기 쌍별 레짐 일치도."""
    names = list(results.keys())
    symbols = list(next(iter(results.values())).keys()) if results else []

    rows = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            agreements = []
            for symbol in symbols:
                df_a = results[name_a].get(symbol)
                df_b = results[name_b].get(symbol)
                if df_a is None or df_b is None:
                    continue
                labels_a = df_a["regime_label"].dropna()
                labels_b = df_b["regime_label"].dropna()
                common = labels_a.index.intersection(labels_b.index)
                if len(common) > 0:
                    agreement = float((labels_a[common] == labels_b[common]).mean()) * 100
                    agreements.append(agreement)
            if agreements:
                rows.append(
                    {
                        "detector_a": name_a,
                        "detector_b": name_b,
                        "avg_agreement%": np.mean(agreements),
                    }
                )
    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    """섹션 헤더 출력."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_results(
    all_metrics: dict[str, list[dict[str, float]]],
    all_regime_dfs: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """종합 결과 출력."""
    print_section("1. Forward 10d Sharpe by Regime")
    rows: list[dict[str, Any]] = []
    for name, metric_list in all_metrics.items():
        if not metric_list:
            continue
        metric_df = pd.DataFrame(metric_list)
        row: dict[str, Any] = {"config": name}
        for label in RegimeLabel:
            col = f"fwd10_sharpe_{label.value}"
            if col in metric_df:
                row[col] = float(metric_df[col].mean())
        rows.append(row)
    fwd_df = pd.DataFrame(rows).set_index("config")
    print(fwd_df.round(3).to_string())

    print_section("2. Directional Accuracy (5d) & Regime Stability")
    rows = []
    for name, metric_list in all_metrics.items():
        if not metric_list:
            continue
        metric_df = pd.DataFrame(metric_list)
        rows.append(
            {
                "config": name,
                "dir_accuracy_5d%": float(metric_df["dir_accuracy_5d"].mean()),
                "avg_duration": float(metric_df["avg_duration"].mean()),
                "transition_count": float(metric_df["transition_count"].mean()),
            }
        )
    stab_df = pd.DataFrame(rows).set_index("config")
    print(stab_df.round(2).to_string())

    print_section("3. Economic Value")
    rows = []
    for name, metric_list in all_metrics.items():
        if not metric_list:
            continue
        metric_df = pd.DataFrame(metric_list)
        rows.append(
            {
                "config": name,
                "adaptive_sharpe": float(metric_df["adaptive_sharpe"].mean()),
                "bh_sharpe": float(metric_df["bh_sharpe"].mean()),
                "improvement": float(
                    metric_df["adaptive_sharpe"].mean() - metric_df["bh_sharpe"].mean()
                ),
            }
        )
    econ_df = pd.DataFrame(rows).set_index("config")
    print(econ_df.round(3).to_string())

    print_section("4. Pairwise Agreement")
    agreement_df = compute_pairwise_agreement(all_regime_dfs)
    if not agreement_df.empty:
        print(agreement_df.round(1).to_string(index=False))

    _print_composite_ranking(all_metrics)


def _print_composite_ranking(all_metrics: dict[str, list[dict[str, float]]]) -> None:
    """Composite ranking 출력."""
    print_section("COMPOSITE RANKING")

    rank_data: dict[str, dict[str, float]] = {}
    for name, metric_list in all_metrics.items():
        if not metric_list:
            continue
        metric_df = pd.DataFrame(metric_list)
        rank_data[name] = {
            "adaptive_sharpe": float(metric_df["adaptive_sharpe"].mean()),
            "dir_accuracy_5d": float(metric_df["dir_accuracy_5d"].mean()),
            "avg_duration": float(metric_df["avg_duration"].mean()),
            "fwd10_sharpe_trending": float(
                metric_df.get("fwd10_sharpe_trending", pd.Series([0.0])).mean()
            ),
        }

    if not rank_data:
        return

    rank_df = pd.DataFrame(rank_data).T
    for col in rank_df.columns:
        rank_df[f"{col}_rank"] = rank_df[col].rank(ascending=False)

    rank_cols = [c for c in rank_df.columns if c.endswith("_rank")]
    rank_df["composite_rank"] = rank_df[rank_cols].sum(axis=1)
    rank_df = rank_df.sort_values("composite_rank")

    print(rank_df[["composite_rank", *rank_cols]].round(1).to_string())
    print(f"\n  BEST CONFIG: {rank_df.index[0]}")


def main() -> None:
    """메인 실행."""
    configs = build_configs()
    print(f"총 {len(configs)}개 감지기 구성 비교")
    print(f"심볼: {SYMBOLS}")
    print(f"기간: {START.date()} ~ {END.date()}")

    all_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in configs}
    all_regime_dfs: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in configs}

    for symbol in SYMBOLS:
        print(f"\n{'─' * 50}")
        print(f"  Processing: {symbol}")
        print(f"{'─' * 50}")

        try:
            df = load_data(symbol)
        except Exception as e:
            print(f"  데이터 로드 실패: {e}")
            continue

        closes: pd.Series = df["close"]  # type: ignore[assignment]
        print(f"  Loaded {len(df)} bars")

        for name, (det_type, detector) in configs.items():
            try:
                regime_df = classify_with_detector(det_type, detector, closes)
                metrics = evaluate_config(closes, regime_df)
                metrics["symbol"] = hash(symbol)  # for tracking
                all_metrics[name].append(metrics)
                all_regime_dfs[name][symbol] = regime_df
                valid_count = regime_df["regime_label"].notna().sum()
                print(f"    {name}: {valid_count} valid bars")
            except Exception as e:
                print(f"    {name}: FAILED ({e})")

    _print_results(all_metrics, all_regime_dfs)


if __name__ == "__main__":
    main()
