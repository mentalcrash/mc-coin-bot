"""Hysteresis 패리티 테스트 — VBT vs EDA 결과 비교.

두 계층으로 검증:
1. Hysteresis 함수 패리티: apply_hysteresis (벡터화) vs update 내부 incremental hysteresis
   → 동일 raw labels 입력 시 동일 결과
2. End-to-end 패리티: classify_series vs update 반복
   → 안정 구간(전환 없음)에서 정확 일치
   → 전환 구간에서는 경계 tolerance 허용 (rolling 계산 차이)

시나리오:
1. 안정 추세 (전환 없음) — 정확 일치
2. 단일 깨끗한 전환 — 경계 tolerance 허용
3. A→B→A 진동 (hysteresis 억제) — 경계 tolerance 허용
4. min_hold_bars 경계값 — 경계 tolerance 허용
5. 3-state 순환 — 경계 tolerance 허용
6. min_hold_bars=1 (pass-through) — 경계 tolerance 허용
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime.config import RegimeDetectorConfig, RegimeLabel
from src.regime.detector import RegimeDetector, apply_hysteresis

# ── Price Series Helpers ──────────────────────────────────────


def _make_trending_series(n: int, start: float = 100.0, daily_return: float = 0.02) -> pd.Series:
    """강한 상승 추세 가격 시리즈 생성."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return))
    return pd.Series(prices, dtype=float)


def _make_ranging_series(n: int, center: float = 100.0, amplitude: float = 0.5) -> pd.Series:
    """횡보 가격 시리즈 생성."""
    rng = np.random.RandomState(42)
    noise = rng.uniform(-amplitude, amplitude, n)
    prices = center + noise
    return pd.Series(prices, dtype=float)


def _make_volatile_series(n: int, start: float = 100.0, volatility: float = 0.08) -> pd.Series:
    """고변동 가격 시리즈 생성."""
    rng = np.random.RandomState(123)
    log_returns = rng.normal(0, volatility, n - 1)
    prices = [start]
    for lr in log_returns:
        prices.append(prices[-1] * np.exp(lr))
    return pd.Series(prices, dtype=float)


def _concat_series(*parts: pd.Series) -> pd.Series:
    """여러 시리즈를 연결 (가격 연속성 유지)."""
    combined: list[float] = []
    for i, part in enumerate(parts):
        values = part.values.tolist()
        if i > 0 and combined:
            ratio = combined[-1] / values[0]
            values = [v * ratio for v in values]
        combined.extend(values)
    return pd.Series(combined, dtype=float)


def _run_vbt(detector: RegimeDetector, prices: pd.Series) -> list[str | None]:
    """classify_series (VBT) 결과에서 regime_label 추출."""
    result = detector.classify_series(prices)
    labels: list[str | None] = []
    for val in result["regime_label"]:
        if pd.isna(val):
            labels.append(None)
        else:
            labels.append(str(val))
    return labels


def _run_eda(config: RegimeDetectorConfig, prices: pd.Series) -> list[str | None]:
    """update (EDA bar-by-bar) 결과에서 regime_label 추출."""
    detector = RegimeDetector(config)
    symbol = "TEST/USDT"
    labels: list[str | None] = []
    for close in prices:
        state = detector.update(symbol, float(close))
        if state is None:
            labels.append(None)
        else:
            labels.append(str(state.label.value))
    return labels


# ── 1. Hysteresis Function Parity ─────────────────────────────


class TestHysteresisFunctionParity:
    """apply_hysteresis (벡터화) vs incremental hysteresis (EDA update) 패리티.

    동일한 raw labels를 입력하면 동일한 결과가 나오는지 검증합니다.
    """

    @staticmethod
    def _run_incremental_hysteresis(
        raw_labels: list[str],
        min_hold_bars: int,
    ) -> list[str]:
        """EDA update()의 hysteresis 로직을 단독 재현."""
        if min_hold_bars <= 1:
            return list(raw_labels)

        result: list[str] = []
        current_label = raw_labels[0]
        pending_label: str | None = None
        pending_count = 0

        result.append(current_label)

        for i in range(1, len(raw_labels)):
            raw = raw_labels[i]
            if raw == current_label:
                pending_label = None
                pending_count = 0
                result.append(current_label)
            elif raw == pending_label:
                pending_count += 1
                if pending_count >= min_hold_bars:
                    current_label = raw
                    pending_label = None
                    pending_count = 0
                    result.append(current_label)
                else:
                    result.append(current_label)
            else:
                pending_label = raw
                pending_count = 1
                result.append(current_label)

        return result

    @pytest.mark.parametrize("min_hold_bars", [1, 3, 5])
    def test_stable_labels(self, min_hold_bars: int) -> None:
        """전환 없는 안정 라벨 시퀀스."""
        raw = ["trending"] * 50
        raw_series = pd.Series(raw, dtype=object)

        vbt_result = apply_hysteresis(raw_series, min_hold_bars).tolist()
        eda_result = self._run_incremental_hysteresis(raw, min_hold_bars)

        assert vbt_result == eda_result

    @pytest.mark.parametrize("min_hold_bars", [3, 5])
    def test_clean_transition(self, min_hold_bars: int) -> None:
        """깨끗한 전환: A → B (B가 min_hold_bars 연속)."""
        raw = ["trending"] * 30 + ["volatile"] * (min_hold_bars + 10)
        raw_series = pd.Series(raw, dtype=object)

        vbt_result = apply_hysteresis(raw_series, min_hold_bars).tolist()
        eda_result = self._run_incremental_hysteresis(raw, min_hold_bars)

        assert vbt_result == eda_result

    @pytest.mark.parametrize("min_hold_bars", [3, 5])
    def test_oscillation_suppressed(self, min_hold_bars: int) -> None:
        """진동 억제: A→B→A 반복이 min_hold_bars 미만이면 전환 안 됨."""
        short = min_hold_bars - 1
        raw = (
            ["trending"] * 20
            + ["volatile"] * short
            + ["trending"] * short
            + ["volatile"] * short
            + ["trending"] * 20
        )
        raw_series = pd.Series(raw, dtype=object)

        vbt_result = apply_hysteresis(raw_series, min_hold_bars).tolist()
        eda_result = self._run_incremental_hysteresis(raw, min_hold_bars)

        assert vbt_result == eda_result

    @pytest.mark.parametrize("min_hold_bars", [3, 5])
    def test_exact_boundary(self, min_hold_bars: int) -> None:
        """정확히 min_hold_bars에서 전환 확정."""
        raw = ["trending"] * 20 + ["ranging"] * min_hold_bars + ["trending"] * 10
        raw_series = pd.Series(raw, dtype=object)

        vbt_result = apply_hysteresis(raw_series, min_hold_bars).tolist()
        eda_result = self._run_incremental_hysteresis(raw, min_hold_bars)

        assert vbt_result == eda_result

    def test_three_state_cycle(self) -> None:
        """3-state 순환: trending→volatile→ranging."""
        min_hold_bars = 3
        raw = ["trending"] * 15 + ["volatile"] * 15 + ["ranging"] * 15
        raw_series = pd.Series(raw, dtype=object)

        vbt_result = apply_hysteresis(raw_series, min_hold_bars).tolist()
        eda_result = self._run_incremental_hysteresis(raw, min_hold_bars)

        assert vbt_result == eda_result

    def test_passthrough_min_hold_1(self) -> None:
        """min_hold_bars=1: hysteresis bypass."""
        raw = ["trending", "volatile", "trending", "ranging", "volatile"]
        raw_series = pd.Series(raw, dtype=object)

        vbt_result = apply_hysteresis(raw_series, 1).tolist()
        eda_result = self._run_incremental_hysteresis(raw, 1)

        assert vbt_result == eda_result
        assert vbt_result == raw  # pass-through


# ── 2. End-to-End Parity ──────────────────────────────────────


class TestEndToEndParity:
    """classify_series vs update 반복 end-to-end 패리티.

    안정 구간(전환 없음)에서는 정확 일치를 검증하고,
    전환 구간에서는 경계 tolerance를 허용합니다.

    Note: VBT는 전체 시리즈에서 pd.rolling()을 사용하고,
    EDA는 제한된 deque 버퍼를 사용하므로 전환 경계에서
    rolling 계산 차이가 발생할 수 있습니다.
    """

    def _assert_parity_exact(
        self,
        config: RegimeDetectorConfig,
        prices: pd.Series,
    ) -> None:
        """VBT와 EDA 결과가 valid 구간에서 정확히 일치하는지 검증."""
        detector_vbt = RegimeDetector(config)
        vbt_labels = _run_vbt(detector_vbt, prices)
        eda_labels = _run_eda(config, prices)

        assert len(vbt_labels) == len(eda_labels) == len(prices)

        mismatches: list[tuple[int, str | None, str | None]] = []
        for i, (vbt, eda) in enumerate(zip(vbt_labels, eda_labels, strict=True)):
            if vbt is None or eda is None:
                continue
            if vbt != eda:
                mismatches.append((i, vbt, eda))

        if mismatches:
            detail = "\n".join(
                f"  bar {i}: VBT={v}, EDA={e}" for i, v, e in mismatches[:10]
            )
            pytest.fail(
                f"{len(mismatches)} mismatches out of {len(prices)} bars:\n{detail}"
            )

    def _assert_parity_tolerant(
        self,
        config: RegimeDetectorConfig,
        prices: pd.Series,
        *,
        max_mismatch_ratio: float = 0.20,
    ) -> None:
        """VBT와 EDA가 대부분 일치하는지 검증 (전환 경계 tolerance 허용)."""
        detector_vbt = RegimeDetector(config)
        vbt_labels = _run_vbt(detector_vbt, prices)
        eda_labels = _run_eda(config, prices)

        assert len(vbt_labels) == len(eda_labels) == len(prices)

        valid_count = 0
        mismatch_count = 0
        for vbt, eda in zip(vbt_labels, eda_labels, strict=True):
            if vbt is None or eda is None:
                continue
            valid_count += 1
            if vbt != eda:
                mismatch_count += 1

        if valid_count == 0:
            return

        ratio = mismatch_count / valid_count
        assert ratio <= max_mismatch_ratio, (
            f"Mismatch ratio {ratio:.1%} exceeds {max_mismatch_ratio:.0%} "
            f"({mismatch_count}/{valid_count} bars)"
        )

    @pytest.mark.parametrize("min_hold_bars", [3, 5])
    def test_stable_trend_exact_match(self, min_hold_bars: int) -> None:
        """시나리오 1: 안정 추세 (전환 없음) — 정확 일치."""
        config = RegimeDetectorConfig(
            rv_short_window=5,
            rv_long_window=20,
            er_window=10,
            min_hold_bars=min_hold_bars,
        )
        prices = _make_trending_series(100, daily_return=0.02)
        self._assert_parity_exact(config, prices)

    @pytest.mark.parametrize("min_hold_bars", [3, 5])
    def test_single_transition_tolerant(self, min_hold_bars: int) -> None:
        """시나리오 2: 추세 → 횡보 전환 — 경계 tolerance."""
        config = RegimeDetectorConfig(
            rv_short_window=5,
            rv_long_window=20,
            er_window=10,
            min_hold_bars=min_hold_bars,
        )
        trend = _make_trending_series(80, daily_return=0.02)
        ranging = _make_ranging_series(80, amplitude=0.3)
        prices = _concat_series(trend, ranging)
        self._assert_parity_tolerant(config, prices)

    @pytest.mark.parametrize("min_hold_bars", [3, 5])
    def test_oscillation_tolerant(self, min_hold_bars: int) -> None:
        """시나리오 3: A→B→A 진동 — 경계 tolerance."""
        config = RegimeDetectorConfig(
            rv_short_window=5,
            rv_long_window=20,
            er_window=10,
            min_hold_bars=min_hold_bars,
        )
        parts = []
        for i in range(4):
            if i % 2 == 0:
                parts.append(_make_trending_series(40, daily_return=0.02))
            else:
                parts.append(_make_ranging_series(40, amplitude=0.3))
        prices = _concat_series(*parts)
        self._assert_parity_tolerant(config, prices)

    @pytest.mark.parametrize("min_hold_bars", [3, 5])
    def test_boundary_hold_bars_tolerant(self, min_hold_bars: int) -> None:
        """시나리오 4: min_hold_bars 경계값 — 경계 tolerance."""
        config = RegimeDetectorConfig(
            rv_short_window=5,
            rv_long_window=20,
            er_window=10,
            min_hold_bars=min_hold_bars,
        )
        trend = _make_trending_series(60, daily_return=0.02)
        ranging = _make_ranging_series(min_hold_bars + 40, amplitude=0.3)
        prices = _concat_series(trend, ranging)
        self._assert_parity_tolerant(config, prices)

    def test_three_state_cycle_tolerant(self) -> None:
        """시나리오 5: trending → volatile → ranging — 경계 tolerance."""
        config = RegimeDetectorConfig(
            rv_short_window=5,
            rv_long_window=20,
            er_window=10,
            min_hold_bars=3,
        )
        trend = _make_trending_series(60, daily_return=0.02)
        volatile = _make_volatile_series(60, volatility=0.08)
        ranging = _make_ranging_series(60, amplitude=0.3)
        prices = _concat_series(trend, volatile, ranging)
        self._assert_parity_tolerant(config, prices)

    def test_passthrough_min_hold_1_tolerant(self) -> None:
        """시나리오 6: min_hold_bars=1 — 경계 tolerance."""
        config = RegimeDetectorConfig(
            rv_short_window=5,
            rv_long_window=20,
            er_window=10,
            min_hold_bars=1,
        )
        trend = _make_trending_series(50, daily_return=0.02)
        ranging = _make_ranging_series(50, amplitude=0.3)
        prices = _concat_series(trend, ranging)
        self._assert_parity_tolerant(config, prices)


# ── 3. Basic Properties ──────────────────────────────────────


class TestHysteresisBasicProperties:
    """Hysteresis 기본 속성 검증."""

    def test_vbt_warmup_produces_nan(self) -> None:
        """VBT classify_series의 warmup 구간은 NaN."""
        config = RegimeDetectorConfig(rv_short_window=5, rv_long_window=20, er_window=10)
        detector = RegimeDetector(config)
        prices = _make_trending_series(30)
        result = detector.classify_series(prices)
        nan_count = result["regime_label"].isna().sum()
        assert nan_count >= config.rv_long_window

    def test_eda_warmup_returns_none(self) -> None:
        """EDA update의 warmup 구간은 None 반환."""
        config = RegimeDetectorConfig(rv_short_window=5, rv_long_window=20, er_window=10)
        detector = RegimeDetector(config)
        prices = _make_trending_series(30)
        none_count = 0
        for close in prices:
            state = detector.update("TEST/USDT", float(close))
            if state is None:
                none_count += 1
        assert none_count >= config.rv_long_window

    def test_consistent_label_values(self) -> None:
        """VBT와 EDA 모두 RegimeLabel 값을 사용한다."""
        valid_labels = {label.value for label in RegimeLabel}
        config = RegimeDetectorConfig(rv_short_window=5, rv_long_window=20, er_window=10)
        prices = _make_trending_series(50)

        # VBT
        detector = RegimeDetector(config)
        result = detector.classify_series(prices)
        for val in result["regime_label"].dropna():
            assert str(val) in valid_labels

        # EDA
        detector2 = RegimeDetector(config)
        for close in prices:
            state = detector2.update("TEST/USDT", float(close))
            if state is not None:
                assert state.label.value in valid_labels
