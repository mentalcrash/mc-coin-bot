"""Regime Detector 성능 평가 스크립트.

RegimeDetector의 레짐 분류가 실제 시장 움직임과 얼마나 일치하는지 평가합니다.

평가 지표:
1. Forward Return by Regime: 각 레짐 이후 실제 수익률
2. Forward Volatility by Regime: 각 레짐의 실제 변동성
3. Directional Accuracy: TRENDING 레짐의 방향성 적중률
4. Regime Stability: 레짐 전환 빈도 및 평균 지속 기간
5. Economic Value: 레짐별 long-only Sharpe
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.regime import (
    EnsembleRegimeDetector,
    EnsembleRegimeDetectorConfig,
    HMMDetectorConfig,
    RegimeLabel,
    VolStructureDetectorConfig,
)

# ── 설정 ──
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)
TIMEFRAME = "1D"
FORWARD_WINDOWS = [1, 5, 10, 20]  # 1일, 5일, 10일, 20일 후

ANNUALIZATION = 365


def load_data(symbol: str) -> pd.DataFrame:
    """Silver 데이터 로드."""
    svc = MarketDataService()
    request = MarketDataRequest(symbol=symbol, timeframe=TIMEFRAME, start=START, end=END)
    ds = svc.get(request)
    return ds.ohlcv


def evaluate_forward_returns(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """각 레짐의 forward return 분석."""
    close = df["close"]
    rows = []

    for window in FORWARD_WINDOWS:
        fwd_ret = close.pct_change(window).shift(-window)

        for label in RegimeLabel:
            mask = regime_df["regime_label"] == label
            regime_fwd = fwd_ret[mask].dropna()

            if len(regime_fwd) < 5:
                continue

            rows.append(
                {
                    "regime": label.value,
                    "fwd_window": window,
                    "count": len(regime_fwd),
                    "mean_return": float(regime_fwd.mean()) * 100,
                    "median_return": float(regime_fwd.median()) * 100,
                    "std_return": float(regime_fwd.std()) * 100,
                    "hit_rate_positive": float((regime_fwd > 0).mean()) * 100,
                    "sharpe": (
                        float(regime_fwd.mean() / regime_fwd.std())
                        * np.sqrt(ANNUALIZATION / window)
                        if regime_fwd.std() > 0
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_forward_volatility(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """각 레짐의 forward volatility 분석."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    rows = []

    for window in [5, 10, 20]:
        fwd_vol = log_ret.rolling(window).std().shift(-window)

        for label in RegimeLabel:
            mask = regime_df["regime_label"] == label
            regime_vol = fwd_vol[mask].dropna()

            if len(regime_vol) < 5:
                continue

            rows.append(
                {
                    "regime": label.value,
                    "fwd_window": window,
                    "mean_vol": float(regime_vol.mean()) * 100,
                    "median_vol": float(regime_vol.median()) * 100,
                }
            )

    return pd.DataFrame(rows)


def evaluate_directional_accuracy(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> dict[str, float]:
    """TRENDING 레짐의 방향성 적중률.

    TRENDING 레짐 = 시장이 한 방향으로 움직임을 의미.
    실제로 forward return의 절대값이 threshold 이상인지 확인.
    """
    close = df["close"]
    trending_mask = regime_df["regime_label"] == RegimeLabel.TRENDING
    ranging_mask = regime_df["regime_label"] == RegimeLabel.RANGING
    volatile_mask = regime_df["regime_label"] == RegimeLabel.VOLATILE

    results: dict[str, float] = {}

    for window in [5, 10]:
        fwd_ret = close.pct_change(window).shift(-window)

        # TRENDING: |fwd_return| > 3% = "실제로 방향성 있었다"
        threshold = 0.03
        trending_fwd = fwd_ret[trending_mask].dropna()
        if len(trending_fwd) > 0:
            directional_hit = float((trending_fwd.abs() > threshold).mean()) * 100
            results[f"trending_{window}d_directional_hit%"] = directional_hit

        # RANGING: |fwd_return| < 3% = "실제로 횡보였다"
        ranging_fwd = fwd_ret[ranging_mask].dropna()
        if len(ranging_fwd) > 0:
            ranging_hit = float((ranging_fwd.abs() < threshold).mean()) * 100
            results[f"ranging_{window}d_sideways_hit%"] = ranging_hit

        # VOLATILE: forward vol > median vol = "실제로 변동성 높았다"
        log_ret = np.log(close / close.shift(1))
        fwd_vol = log_ret.rolling(window).std().shift(-window)
        median_vol = fwd_vol.median()

        vol_fwd = fwd_vol[volatile_mask].dropna()
        if len(vol_fwd) > 0:
            vol_hit = float((vol_fwd > median_vol).mean()) * 100
            results[f"volatile_{window}d_highvol_hit%"] = vol_hit

    return results


def evaluate_regime_stability(
    regime_df: pd.DataFrame,
) -> dict[str, float]:
    """레짐 안정성 분석 (전환 빈도, 평균 지속 기간)."""
    labels = regime_df["regime_label"].dropna()
    transitions = (labels != labels.shift(1)).sum()
    total_bars = len(labels)

    # 레짐 run lengths
    groups = (labels != labels.shift(1)).cumsum()
    run_lengths = labels.groupby(groups).size()

    results = {
        "total_bars": float(total_bars),
        "total_transitions": float(transitions),
        "transition_rate_per_bar": float(transitions / total_bars) if total_bars > 0 else 0.0,
        "avg_regime_duration": float(run_lengths.mean()),
        "median_regime_duration": float(run_lengths.median()),
        "max_regime_duration": float(run_lengths.max()),
    }

    # 레짐별 분포
    dist = labels.value_counts(normalize=True)
    for label in RegimeLabel:
        key = f"pct_{label.value}"
        results[key] = float(dist.get(label, 0.0)) * 100

    return results


def evaluate_economic_value(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> dict[str, float]:
    """레짐 기반 전략의 경제적 가치.

    1) Buy-and-Hold baseline
    2) Trending-Only: TRENDING 레짐에서만 포지션
    3) Avoid-Volatile: VOLATILE 레짐 회피
    4) Regime-Adaptive: TRENDING=1.0, RANGING=0.3, VOLATILE=0.0
    """
    daily_ret = df["close"].pct_change()

    # Buy-and-Hold
    bh_sharpe = _sharpe(daily_ret)
    bh_cagr = _cagr(daily_ret)

    # TRENDING only
    trending_mask = regime_df["regime_label"] == RegimeLabel.TRENDING
    trending_ret = daily_ret.where(trending_mask, 0.0)
    trending_sharpe = _sharpe(trending_ret)
    trending_cagr = _cagr(trending_ret)

    # Avoid VOLATILE
    volatile_mask = regime_df["regime_label"] == RegimeLabel.VOLATILE
    avoid_vol_ret = daily_ret.where(~volatile_mask, 0.0)
    avoid_vol_sharpe = _sharpe(avoid_vol_ret)
    avoid_vol_cagr = _cagr(avoid_vol_ret)

    # Regime-Adaptive
    weight = pd.Series(0.0, index=df.index)
    weight[regime_df["regime_label"] == RegimeLabel.TRENDING] = 1.0
    weight[regime_df["regime_label"] == RegimeLabel.RANGING] = 0.3
    weight[regime_df["regime_label"] == RegimeLabel.VOLATILE] = 0.0
    adaptive_ret = daily_ret * weight
    adaptive_sharpe = _sharpe(adaptive_ret)
    adaptive_cagr = _cagr(adaptive_ret)

    return {
        "buy_hold_sharpe": bh_sharpe,
        "buy_hold_cagr%": bh_cagr,
        "trending_only_sharpe": trending_sharpe,
        "trending_only_cagr%": trending_cagr,
        "avoid_volatile_sharpe": avoid_vol_sharpe,
        "avoid_volatile_cagr%": avoid_vol_cagr,
        "adaptive_sharpe": adaptive_sharpe,
        "adaptive_cagr%": adaptive_cagr,
    }


def _sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe Ratio."""
    ret = returns.dropna()
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(ANNUALIZATION))


def _cagr(returns: pd.Series) -> float:
    """CAGR % 계산."""
    ret = returns.dropna()
    if len(ret) < 2:
        return 0.0
    cum = (1 + ret).prod()
    years = len(ret) / ANNUALIZATION
    if cum <= 0 or years <= 0:
        return 0.0
    return float((cum ** (1 / years) - 1) * 100)


def evaluate_probability_calibration(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """확률 캘리브레이션: p_trending이 높을 때 실제로 trending인 비율.

    확률을 구간별로 나누고, 각 구간의 실제 적중률 비교.
    """
    close = df["close"]
    fwd_ret_10 = close.pct_change(10).shift(-10)

    rows = []
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    for prob_col, metric_name, check_fn in [
        ("p_trending", "trending", lambda r: abs(r) > 0.03),  # 방향성 있음
        ("p_ranging", "ranging", lambda r: abs(r) < 0.03),  # 횡보
        ("p_volatile", "volatile", lambda r: abs(r) > 0.05),  # 큰 변동
    ]:
        prob = regime_df[prob_col].dropna()
        aligned_fwd = fwd_ret_10.reindex(prob.index).dropna()
        common_idx = prob.index.intersection(aligned_fwd.index)
        prob = prob[common_idx]
        aligned_fwd = aligned_fwd[common_idx]

        prob_bins = pd.cut(prob, bins=bins, labels=bin_labels, include_lowest=True)

        for bin_label in bin_labels:
            mask = prob_bins == bin_label
            if mask.sum() < 10:
                continue

            actual_hit = float(aligned_fwd[mask].apply(check_fn).mean()) * 100

            rows.append(
                {
                    "metric": metric_name,
                    "prob_bin": bin_label,
                    "count": int(mask.sum()),
                    "actual_hit%": actual_hit,
                }
            )

    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    """섹션 헤더 출력."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main() -> None:
    """메인 실행."""
    # 앙상블: Rule-Based(0.40) + HMM(0.35) + Vol-Structure(0.25)
    ensemble_config = EnsembleRegimeDetectorConfig(
        hmm=HMMDetectorConfig(min_train_window=252, retrain_interval=21, n_iter=100),
        vol_structure=VolStructureDetectorConfig(),
        weight_rule_based=0.40,
        weight_hmm=0.35,
        weight_vol_structure=0.25,
        min_hold_bars=5,
    )
    detector = EnsembleRegimeDetector(ensemble_config)

    all_fwd_returns: list[pd.DataFrame] = []
    all_fwd_vol: list[pd.DataFrame] = []
    all_directional: list[dict[str, float]] = []
    all_stability: list[dict[str, float]] = []
    all_economic: list[dict[str, float]] = []
    all_calibration: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        print(f"\n{'─' * 50}")
        print(f"  Processing: {symbol}")
        print(f"{'─' * 50}")

        try:
            df = load_data(symbol)
        except Exception as e:
            print(f"  ⚠ 데이터 로드 실패: {e}")
            continue

        print(f"  Loaded {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})")

        # 레짐 분류
        regime_df = detector.classify_series(df["close"])
        valid = regime_df["regime_label"].notna()
        print(f"  Regime classified: {valid.sum()} valid bars (warmup: {(~valid).sum()})")

        # 레짐 분포
        dist = regime_df["regime_label"].value_counts()
        for label, count in dist.items():
            pct = count / valid.sum() * 100
            print(f"    {label}: {count} bars ({pct:.1f}%)")

        # 평가
        fwd_ret = evaluate_forward_returns(df, regime_df)
        fwd_ret["symbol"] = symbol
        all_fwd_returns.append(fwd_ret)

        fwd_vol = evaluate_forward_volatility(df, regime_df)
        fwd_vol["symbol"] = symbol
        all_fwd_vol.append(fwd_vol)

        directional = evaluate_directional_accuracy(df, regime_df)
        directional["symbol"] = symbol
        all_directional.append(directional)

        stability = evaluate_regime_stability(regime_df)
        stability["symbol"] = symbol
        all_stability.append(stability)

        economic = evaluate_economic_value(df, regime_df)
        economic["symbol"] = symbol
        all_economic.append(economic)

        calibration = evaluate_probability_calibration(df, regime_df)
        calibration["symbol"] = symbol
        all_calibration.append(calibration)

    # ── 종합 결과 출력 ──
    print_section("1. Forward Returns by Regime (전 심볼 평균)")
    fwd_all = pd.concat(all_fwd_returns, ignore_index=True)
    summary = (
        fwd_all.groupby(["regime", "fwd_window"])
        .agg(
            mean_return=("mean_return", "mean"),
            std_return=("std_return", "mean"),
            hit_rate_positive=("hit_rate_positive", "mean"),
            sharpe=("sharpe", "mean"),
            count=("count", "sum"),
        )
        .round(2)
    )
    print(summary.to_string())

    print_section("2. Forward Volatility by Regime")
    vol_all = pd.concat(all_fwd_vol, ignore_index=True)
    vol_summary = (
        vol_all.groupby(["regime", "fwd_window"])
        .agg(
            mean_vol=("mean_vol", "mean"),
            median_vol=("median_vol", "mean"),
        )
        .round(4)
    )
    print(vol_summary.to_string())

    print_section("3. Directional Accuracy (적중률)")
    dir_df = pd.DataFrame(all_directional)
    print(dir_df.set_index("symbol").to_string())

    # 심볼별 평균 적중률
    numeric_cols = [c for c in dir_df.columns if c != "symbol"]
    means = dir_df[numeric_cols].mean()
    print("\n--- Cross-Symbol Average ---")
    for col, val in means.items():
        print(f"  {col}: {val:.1f}%")

    print_section("4. Regime Stability")
    stab_df = pd.DataFrame(all_stability)
    cols_to_show = [
        "symbol",
        "total_bars",
        "total_transitions",
        "transition_rate_per_bar",
        "avg_regime_duration",
        "median_regime_duration",
        "pct_trending",
        "pct_ranging",
        "pct_volatile",
    ]
    print(stab_df[cols_to_show].to_string(index=False))

    print_section("5. Economic Value (레짐 기반 전략 vs Buy-and-Hold)")
    econ_df = pd.DataFrame(all_economic)
    print(econ_df.to_string(index=False))

    # 평균
    print("\n--- Cross-Symbol Average ---")
    numeric_econ = [c for c in econ_df.columns if c != "symbol"]
    for col in numeric_econ:
        print(f"  {col}: {econ_df[col].mean():.2f}")

    print_section("6. Probability Calibration (확률 캘리브레이션)")
    cal_all = pd.concat(all_calibration, ignore_index=True)
    cal_summary = (
        cal_all.groupby(["metric", "prob_bin"])
        .agg(
            actual_hit_pct=("actual_hit%", "mean"),
            total_count=("count", "sum"),
        )
        .round(1)
    )
    print(cal_summary.to_string())

    # ── 최종 요약 ──
    print_section("FINAL VERDICT")

    avg_trending_sharpe = fwd_all[
        (fwd_all["regime"] == "trending") & (fwd_all["fwd_window"] == 10)
    ]["sharpe"].mean()
    avg_ranging_sharpe = fwd_all[(fwd_all["regime"] == "ranging") & (fwd_all["fwd_window"] == 10)][
        "sharpe"
    ].mean()
    avg_volatile_sharpe = fwd_all[
        (fwd_all["regime"] == "volatile") & (fwd_all["fwd_window"] == 10)
    ]["sharpe"].mean()

    print("  10-day Forward Sharpe:")
    print(f"    TRENDING: {avg_trending_sharpe:.2f}")
    print(f"    RANGING:  {avg_ranging_sharpe:.2f}")
    print(f"    VOLATILE: {avg_volatile_sharpe:.2f}")

    dir_means = dir_df[[c for c in dir_df.columns if c != "symbol"]].mean()
    trending_5d_hit = dir_means.get("trending_5d_directional_hit%", 0)
    ranging_5d_hit = dir_means.get("ranging_5d_sideways_hit%", 0)
    volatile_5d_hit = dir_means.get("volatile_5d_highvol_hit%", 0)

    print("\n  5-day Accuracy:")
    print(f"    TRENDING directional hit:  {trending_5d_hit:.1f}%")
    print(f"    RANGING sideways hit:      {ranging_5d_hit:.1f}%")
    print(f"    VOLATILE high-vol hit:     {volatile_5d_hit:.1f}%")

    avg_bh_sharpe = econ_df["buy_hold_sharpe"].mean()
    avg_adaptive_sharpe = econ_df["adaptive_sharpe"].mean()
    improvement = avg_adaptive_sharpe - avg_bh_sharpe

    print("\n  Economic Value (Regime-Adaptive vs Buy-Hold):")
    print(f"    Buy-Hold Sharpe:    {avg_bh_sharpe:.2f}")
    print(f"    Adaptive Sharpe:    {avg_adaptive_sharpe:.2f}")
    print(f"    Improvement:        {improvement:+.2f}")

    if improvement > 0.1:
        print(f"\n  ✅ 레짐 분석기가 경제적 가치를 제공합니다 (Sharpe +{improvement:.2f})")
    elif improvement > -0.1:
        print(f"\n  ⚠️  레짐 분석기의 경제적 가치가 미미합니다 (Sharpe {improvement:+.2f})")
    else:
        print(f"\n  ❌ 레짐 분석기가 오히려 성과를 악화시킵니다 (Sharpe {improvement:+.2f})")


if __name__ == "__main__":
    main()
