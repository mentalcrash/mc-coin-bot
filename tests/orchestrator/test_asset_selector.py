"""Tests for AssetSelector — Pod 내 에셋 선별 FSM."""

from __future__ import annotations

import pytest

from src.orchestrator.asset_selector import AssetSelector, _cross_sectional_rank
from src.orchestrator.config import AssetSelectorConfig
from src.orchestrator.models import AssetLifecycleState

# ── Helpers ─────────────────────────────────────────────────────


def _make_config(**overrides: object) -> AssetSelectorConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "exclude_score_threshold": 0.20,
        "include_score_threshold": 0.35,
        "exclude_confirmation_bars": 3,
        "include_confirmation_bars": 2,
        "min_exclusion_bars": 5,
        "ramp_steps": 3,
        "min_active_assets": 2,
        "sharpe_lookback": 20,
        "return_lookback": 10,
    }
    defaults.update(overrides)
    return AssetSelectorConfig(**defaults)  # type: ignore[arg-type]


def _make_selector(
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"),
    **config_overrides: object,
) -> AssetSelector:
    cfg = _make_config(**config_overrides)
    return AssetSelector(config=cfg, symbols=symbols)


def _make_returns(
    n: int = 30,
    base_return: float = 0.01,
) -> list[float]:
    """n개의 약간 변동하는 수익률 (분산 ≠ 0)."""
    return [base_return + 0.001 * (i % 5 - 2) for i in range(n)]


def _make_bad_returns(n: int = 30) -> list[float]:
    """성과 미달 수익률 (음수, 변동 있음)."""
    return [-0.02 + 0.001 * (i % 3 - 1) for i in range(n)]


def _set_score(selector: AssetSelector, symbol: str, score: float) -> None:
    """테스트용: 직접 점수 설정."""
    selector._states[symbol].score = score


def _call_transition(selector: AssetSelector, symbol: str) -> None:
    """테스트용: FSM 전이만 실행."""
    selector._transition(symbol)


def _get_state(selector: AssetSelector, symbol: str) -> AssetLifecycleState:
    return selector._states[symbol].state


def _get_multiplier(selector: AssetSelector, symbol: str) -> float:
    return selector._states[symbol].multiplier


# ── TestCrossSectionalRank ───────────────────────────────────────


class TestCrossSectionalRank:
    def test_single_asset(self) -> None:
        result = _cross_sectional_rank({"BTC": 1.0})
        assert result["BTC"] == pytest.approx(0.5)

    def test_two_assets(self) -> None:
        result = _cross_sectional_rank({"BTC": 1.0, "ETH": 2.0})
        assert result["BTC"] == pytest.approx(0.0)
        assert result["ETH"] == pytest.approx(1.0)

    def test_three_assets(self) -> None:
        result = _cross_sectional_rank({"A": 1.0, "B": 2.0, "C": 3.0})
        assert result["A"] == pytest.approx(0.0)
        assert result["B"] == pytest.approx(0.5)
        assert result["C"] == pytest.approx(1.0)

    def test_tied_values(self) -> None:
        result = _cross_sectional_rank({"A": 1.0, "B": 1.0, "C": 2.0})
        # A, B tied at ranks 0,1 → avg 0.5 → normalized 0.5/2=0.25
        assert result["A"] == pytest.approx(0.25)
        assert result["B"] == pytest.approx(0.25)
        assert result["C"] == pytest.approx(1.0)

    def test_all_equal(self) -> None:
        result = _cross_sectional_rank({"A": 1.0, "B": 1.0})
        # Both tied → avg rank 0.5 → 0.5/1=0.5
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == pytest.approx(0.5)


# ── TestAssetSelectorInit ────────────────────────────────────────


class TestAssetSelectorInit:
    def test_all_active_initially(self) -> None:
        selector = _make_selector()
        for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"):
            assert _get_state(selector, symbol) == AssetLifecycleState.ACTIVE
            assert _get_multiplier(selector, symbol) == pytest.approx(1.0)

    def test_multipliers_all_one(self) -> None:
        selector = _make_selector()
        mults = selector.multipliers
        assert all(v == pytest.approx(1.0) for v in mults.values())

    def test_active_symbols_all(self) -> None:
        selector = _make_selector()
        assert len(selector.active_symbols) == 4

    def test_all_excluded_false(self) -> None:
        selector = _make_selector()
        assert selector.all_excluded is False

    def test_disabled_no_transition(self) -> None:
        selector = _make_selector(enabled=False)
        returns = {s: _make_bad_returns(30) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")}
        closes = dict.fromkeys(returns, 100.0)
        for _ in range(20):
            selector.on_bar(returns, closes)
        # 모든 에셋 여전히 ACTIVE
        for symbol in returns:
            assert _get_state(selector, symbol) == AssetLifecycleState.ACTIVE


# ── TestActiveToUnderperforming ──────────────────────────────────


class TestActiveToUnderperforming:
    def test_exclude_after_confirmation(self) -> None:
        """score < threshold, N bars 연속 → UNDERPERFORMING."""
        selector = _make_selector(exclude_confirmation_bars=3)
        _set_score(selector, "BTC/USDT", 0.15)  # < 0.20

        for _ in range(3):
            _call_transition(selector, "BTC/USDT")

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.UNDERPERFORMING

    def test_no_exclude_before_confirmation(self) -> None:
        """confirmation 미달 → 상태 유지."""
        selector = _make_selector(exclude_confirmation_bars=3)
        _set_score(selector, "BTC/USDT", 0.15)

        for _ in range(2):
            _call_transition(selector, "BTC/USDT")

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.ACTIVE

    def test_confirmation_reset_on_score_recovery(self) -> None:
        """score 회복 → confirmation 리셋."""
        selector = _make_selector(exclude_confirmation_bars=3)
        _set_score(selector, "BTC/USDT", 0.15)

        _call_transition(selector, "BTC/USDT")
        _call_transition(selector, "BTC/USDT")

        # Score 회복
        _set_score(selector, "BTC/USDT", 0.25)
        _call_transition(selector, "BTC/USDT")

        # Confirmation 리셋됨 → 여전히 ACTIVE
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.ACTIVE
        assert selector._states["BTC/USDT"].confirmation_count == 0


# ── TestRampDown ─────────────────────────────────────────────────


class TestRampDown:
    def test_ramp_3_steps(self) -> None:
        """3단계 ramp down: 1.0 → 0.66 → 0.33 → 0.0."""
        selector = _make_selector(
            exclude_confirmation_bars=1,
            ramp_steps=3,
        )
        _set_score(selector, "BTC/USDT", 0.15)

        # Confirm → UNDERPERFORMING + 1st ramp
        _call_transition(selector, "BTC/USDT")
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.UNDERPERFORMING
        assert _get_multiplier(selector, "BTC/USDT") == pytest.approx(2.0 / 3.0)

        # 2nd ramp
        _call_transition(selector, "BTC/USDT")
        assert _get_multiplier(selector, "BTC/USDT") == pytest.approx(1.0 / 3.0)

        # 3rd ramp → COOLDOWN
        _call_transition(selector, "BTC/USDT")
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.COOLDOWN
        assert _get_multiplier(selector, "BTC/USDT") == pytest.approx(0.0)


# ── TestCooldown ─────────────────────────────────────────────────


class TestCooldown:
    def test_cooldown_minimum_bars(self) -> None:
        """cooldown 기간 미경과 → RE_ENTRY 불가."""
        selector = _make_selector(
            exclude_confirmation_bars=1,
            include_confirmation_bars=1,
            ramp_steps=1,
            min_exclusion_bars=5,
        )
        # Force to COOLDOWN
        st = selector._states["BTC/USDT"]
        st.state = AssetLifecycleState.COOLDOWN
        st.multiplier = 0.0
        st.cooldown_bars = 0

        _set_score(selector, "BTC/USDT", 0.50)  # > include threshold

        for _ in range(4):  # 4 < 5
            _call_transition(selector, "BTC/USDT")

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.COOLDOWN

    def test_cooldown_to_reentry(self) -> None:
        """cooldown 경과 + score 충족 + confirmation → RE_ENTRY."""
        selector = _make_selector(
            include_confirmation_bars=2,
            ramp_steps=3,
            min_exclusion_bars=3,
        )
        st = selector._states["BTC/USDT"]
        st.state = AssetLifecycleState.COOLDOWN
        st.multiplier = 0.0
        st.cooldown_bars = 0

        _set_score(selector, "BTC/USDT", 0.50)

        # 3 bars cooldown
        for _ in range(3):
            _call_transition(selector, "BTC/USDT")
        # cooldown_bars=3 >= min=3, but confirmation still needed

        # 2 bars confirmation
        _call_transition(selector, "BTC/USDT")  # conf=1
        _call_transition(selector, "BTC/USDT")  # conf=2 → RE_ENTRY

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.RE_ENTRY


# ── TestRampUp ───────────────────────────────────────────────────


class TestRampUp:
    def test_ramp_up_3_steps(self) -> None:
        """3단계 ramp up: 0.0 → 0.33 → 0.66 → 1.0 → ACTIVE."""
        selector = _make_selector(ramp_steps=3)
        st = selector._states["BTC/USDT"]
        st.state = AssetLifecycleState.RE_ENTRY
        st.multiplier = 0.0
        st.ramp_position = 0
        _set_score(selector, "BTC/USDT", 0.50)

        # Step 1: 1/3
        _call_transition(selector, "BTC/USDT")
        assert _get_multiplier(selector, "BTC/USDT") == pytest.approx(1.0 / 3.0)

        # Step 2: 2/3
        _call_transition(selector, "BTC/USDT")
        assert _get_multiplier(selector, "BTC/USDT") == pytest.approx(2.0 / 3.0)

        # Step 3: 3/3 → ACTIVE
        _call_transition(selector, "BTC/USDT")
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.ACTIVE
        assert _get_multiplier(selector, "BTC/USDT") == pytest.approx(1.0)

    def test_reentry_to_underperforming_on_score_drop(self) -> None:
        """RE_ENTRY 도중 score 재하락 → UNDERPERFORMING."""
        selector = _make_selector(
            exclude_confirmation_bars=2,
            ramp_steps=5,
        )
        st = selector._states["BTC/USDT"]
        st.state = AssetLifecycleState.RE_ENTRY
        st.multiplier = 0.33
        st.ramp_position = 1
        _set_score(selector, "BTC/USDT", 0.10)  # < exclude threshold

        # 2 bars confirmation
        _call_transition(selector, "BTC/USDT")
        _call_transition(selector, "BTC/USDT")

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.UNDERPERFORMING


# ── TestHardExclusion ────────────────────────────────────────────


class TestHardExclusion:
    def test_hard_exclude_immediate(self) -> None:
        """Sharpe < -1.0 AND DD > 15% → 즉시 COOLDOWN."""
        selector = _make_selector(min_active_assets=1)
        # Generate terrible returns: deep variable losses (var > 0 required for Sharpe)
        bad_returns = [-0.05 + 0.01 * (i % 3 - 1) for i in range(30)]
        returns = {
            "BTC/USDT": bad_returns,
            "ETH/USDT": _make_returns(30, 0.01),
            "SOL/USDT": _make_returns(30, 0.01),
            "BNB/USDT": _make_returns(30, 0.01),
        }
        closes = dict.fromkeys(returns, 100.0)

        selector.on_bar(returns, closes)

        # BTC should be hard excluded (very negative Sharpe + high DD)
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.COOLDOWN
        assert _get_multiplier(selector, "BTC/USDT") == pytest.approx(0.0)

    def test_no_hard_exclude_if_only_sharpe_bad(self) -> None:
        """Sharpe만 나쁘고 DD 조건 미충족 → hard exclude 안 됨."""
        selector = _make_selector()
        # Mildly bad: negative but low DD
        returns = {s: _make_returns(30, -0.001) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")}
        closes = dict.fromkeys(returns, 100.0)

        selector.on_bar(returns, closes)

        # All still ACTIVE (mild losses)
        for symbol in returns:
            assert _get_state(selector, symbol) == AssetLifecycleState.ACTIVE


# ── TestMinActiveAssets ──────────────────────────────────────────


class TestMinActiveAssets:
    def test_min_active_prevents_exclusion(self) -> None:
        """min_active=3, 4 에셋 중 2개 이미 제외 → 3번째 제외 불가."""
        selector = _make_selector(
            min_active_assets=3,
            exclude_confirmation_bars=1,
            ramp_steps=1,
        )
        # 2개 제외
        for symbol in ("ETH/USDT", "SOL/USDT"):
            selector._states[symbol].state = AssetLifecycleState.COOLDOWN
            selector._states[symbol].multiplier = 0.0

        # BTC 제외 시도 → 방지 (active=4-2=2, 제외 후 1 < min_active=3)
        _set_score(selector, "BTC/USDT", 0.10)
        _call_transition(selector, "BTC/USDT")

        # BNB + BTC만 active(2개), min_active=3이므로 BTC 제외 불가
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.ACTIVE

    def test_min_active_allows_when_enough(self) -> None:
        """active 충분 → 제외 허용."""
        selector = _make_selector(
            min_active_assets=2,
            exclude_confirmation_bars=1,
            ramp_steps=1,
        )
        _set_score(selector, "BTC/USDT", 0.10)
        _call_transition(selector, "BTC/USDT")

        # 4 에셋, 1개 제외해도 3개 active ≥ min 2 → 허용
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.UNDERPERFORMING


# ── TestWhipsawPrevention ────────────────────────────────────────


class TestWhipsawPrevention:
    def test_score_oscillation_no_transition(self) -> None:
        """Score 0.19 → 0.22 → 0.18 진동 → 상태 유지 (confirmation 미달)."""
        selector = _make_selector(exclude_confirmation_bars=3)

        # Below threshold
        _set_score(selector, "BTC/USDT", 0.19)
        _call_transition(selector, "BTC/USDT")  # conf=1

        # Above threshold → reset
        _set_score(selector, "BTC/USDT", 0.22)
        _call_transition(selector, "BTC/USDT")  # conf=0

        # Below again
        _set_score(selector, "BTC/USDT", 0.18)
        _call_transition(selector, "BTC/USDT")  # conf=1

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.ACTIVE

    def test_hysteresis_gap(self) -> None:
        """Score between exclude(0.20) and include(0.35) → 현재 상태 유지."""
        selector = _make_selector(
            exclude_confirmation_bars=1,
            include_confirmation_bars=1,
            ramp_steps=1,
            min_exclusion_bars=1,
        )

        # BTC → COOLDOWN (force)
        st = selector._states["BTC/USDT"]
        st.state = AssetLifecycleState.COOLDOWN
        st.multiplier = 0.0
        st.cooldown_bars = 10  # > min_exclusion_bars

        # Score 0.25: > exclude(0.20), < include(0.35) → 재진입 안 됨
        _set_score(selector, "BTC/USDT", 0.25)
        _call_transition(selector, "BTC/USDT")

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.COOLDOWN


# ── TestOnBarIntegration ─────────────────────────────────────────


class TestOnBarIntegration:
    def test_on_bar_computes_scores(self) -> None:
        """on_bar()이 scores를 업데이트."""
        selector = _make_selector()
        returns = {s: _make_returns(30) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")}
        closes = dict.fromkeys(returns, 100.0)

        selector.on_bar(returns, closes)

        scores = selector.scores
        for s in returns:
            assert 0.0 <= scores[s] <= 1.0

    def test_insufficient_data_neutral_score(self) -> None:
        """데이터 < 10봉 → neutral score (0.5)."""
        selector = _make_selector()
        returns = {s: [0.01] * 5 for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")}
        closes = dict.fromkeys(returns, 100.0)

        selector.on_bar(returns, closes)

        for s in returns:
            assert selector.scores[s] == pytest.approx(0.5)

    def test_bad_asset_eventually_excluded(self) -> None:
        """한 에셋만 나쁜 수익률 → 결국 제외."""
        selector = _make_selector(
            exclude_confirmation_bars=2,
            ramp_steps=2,
            min_active_assets=1,
        )
        returns = {
            "BTC/USDT": _make_returns(30, -0.03),
            "ETH/USDT": _make_returns(30, 0.02),
            "SOL/USDT": _make_returns(30, 0.02),
            "BNB/USDT": _make_returns(30, 0.01),
        }
        closes = dict.fromkeys(returns, 100.0)

        # 여러 번 호출하여 FSM 진행
        for _ in range(20):
            selector.on_bar(returns, closes)

        # BTC는 worst → 제외 예상
        btc_state = _get_state(selector, "BTC/USDT")
        assert btc_state in (
            AssetLifecycleState.UNDERPERFORMING,
            AssetLifecycleState.COOLDOWN,
        )

    def test_equal_returns_no_exclusion(self) -> None:
        """모든 에셋 동일 수익률 → 모두 ACTIVE."""
        selector = _make_selector()
        returns = {s: _make_returns(30, 0.01) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")}
        closes = dict.fromkeys(returns, 100.0)

        for _ in range(10):
            selector.on_bar(returns, closes)

        for s in returns:
            assert _get_state(selector, s) == AssetLifecycleState.ACTIVE


# ── TestSerialization ────────────────────────────────────────────


class TestAssetSelectorSerialization:
    def test_round_trip(self) -> None:
        """to_dict() → restore_from_dict() 왕복."""
        selector = _make_selector()
        # Modify state
        selector._states["BTC/USDT"].state = AssetLifecycleState.COOLDOWN
        selector._states["BTC/USDT"].multiplier = 0.0
        selector._states["BTC/USDT"].cooldown_bars = 15
        selector._states["BTC/USDT"].score = 0.12
        selector._states["ETH/USDT"].state = AssetLifecycleState.RE_ENTRY
        selector._states["ETH/USDT"].multiplier = 0.33
        selector._states["ETH/USDT"].ramp_position = 1

        data = selector.to_dict()

        # Restore to new instance
        selector2 = _make_selector()
        selector2.restore_from_dict(data)

        assert _get_state(selector2, "BTC/USDT") == AssetLifecycleState.COOLDOWN
        assert _get_multiplier(selector2, "BTC/USDT") == pytest.approx(0.0)
        assert selector2._states["BTC/USDT"].cooldown_bars == 15
        assert selector2._states["BTC/USDT"].score == pytest.approx(0.12)
        assert _get_state(selector2, "ETH/USDT") == AssetLifecycleState.RE_ENTRY
        assert _get_multiplier(selector2, "ETH/USDT") == pytest.approx(0.33)

    def test_restore_unknown_symbol_ignored(self) -> None:
        """저장에 있지만 현재 config에 없는 심볼 → 무시."""
        selector = _make_selector(symbols=("BTC/USDT", "ETH/USDT"))
        data = {
            "states": {
                "BTC/USDT": {
                    "state": "cooldown",
                    "multiplier": 0.0,
                    "score": 0.1,
                    "confirmation_count": 0,
                    "cooldown_bars": 10,
                    "ramp_position": 0,
                },
                "UNKNOWN/USDT": {
                    "state": "active",
                    "multiplier": 1.0,
                    "score": 0.5,
                    "confirmation_count": 0,
                    "cooldown_bars": 0,
                    "ramp_position": 0,
                },
            }
        }
        selector.restore_from_dict(data)

        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.COOLDOWN
        # UNKNOWN/USDT 무시됨 (키 자체가 없음)
        assert "UNKNOWN/USDT" not in selector._states

    def test_restore_empty_data(self) -> None:
        """빈 데이터 → 상태 변경 없음."""
        selector = _make_selector()
        selector.restore_from_dict({})
        for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"):
            assert _get_state(selector, s) == AssetLifecycleState.ACTIVE


# ── TestAllExcluded ──────────────────────────────────────────────


class TestAllExcluded:
    def test_all_excluded_true(self) -> None:
        """모든 에셋 multiplier=0 → all_excluded=True."""
        selector = _make_selector()
        for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"):
            selector._states[s].multiplier = 0.0
        assert selector.all_excluded is True

    def test_one_active_not_all_excluded(self) -> None:
        """하나라도 active → all_excluded=False."""
        selector = _make_selector()
        for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            selector._states[s].multiplier = 0.0
        selector._states["BNB/USDT"].multiplier = 0.33
        assert selector.all_excluded is False


# ── TestMetricHelpers ────────────────────────────────────────────


class TestMetricHelpers:
    def test_sharpe_positive(self) -> None:
        returns = [0.01 + 0.001 * (i % 5 - 2) for i in range(30)]
        sharpe = AssetSelector._compute_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_zero_vol(self) -> None:
        returns = [0.0] * 30
        sharpe = AssetSelector._compute_sharpe(returns)
        assert sharpe == pytest.approx(0.0)

    def test_sharpe_insufficient_data(self) -> None:
        sharpe = AssetSelector._compute_sharpe([0.01])
        assert sharpe == pytest.approx(0.0)

    def test_cumulative_return(self) -> None:
        returns = [0.01] * 10
        cum = AssetSelector._compute_cumulative_return(returns)
        expected = (1.01**10) - 1.0
        assert cum == pytest.approx(expected)

    def test_max_drawdown(self) -> None:
        returns = [0.1, -0.2, 0.05]
        dd = AssetSelector._compute_max_drawdown(returns)
        # After +10%: equity=1.1, peak=1.1
        # After -20%: equity=0.88, peak=1.1, dd=(1.1-0.88)/1.1=0.2
        assert dd == pytest.approx(0.2)

    def test_max_drawdown_no_loss(self) -> None:
        returns = [0.01, 0.02, 0.01]
        dd = AssetSelector._compute_max_drawdown(returns)
        assert dd == pytest.approx(0.0)


# ── TestEdgeCases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_two_symbols(self) -> None:
        """2개 심볼로도 정상 동작."""
        selector = _make_selector(
            symbols=("BTC/USDT", "ETH/USDT"),
            min_active_assets=1,
        )
        returns = {
            "BTC/USDT": _make_returns(30, 0.01),
            "ETH/USDT": _make_returns(30, 0.01),
        }
        closes = dict.fromkeys(returns, 100.0)
        selector.on_bar(returns, closes)
        assert len(selector.active_symbols) == 2

    def test_single_symbol_never_excluded(self) -> None:
        """1개 심볼 + min_active=1 → 제외 불가."""
        selector = _make_selector(
            symbols=("BTC/USDT",),
            min_active_assets=1,
            exclude_confirmation_bars=1,
        )
        _set_score(selector, "BTC/USDT", 0.10)
        _call_transition(selector, "BTC/USDT")
        assert _get_state(selector, "BTC/USDT") == AssetLifecycleState.ACTIVE

    def test_asset_states_property(self) -> None:
        selector = _make_selector()
        states = selector.asset_states
        assert len(states) == 4
        assert all(v == AssetLifecycleState.ACTIVE for v in states.values())
