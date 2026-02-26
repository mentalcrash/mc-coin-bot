"""Tests for StrategyPod — 전략별 독립 실행 단위."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.orchestrator.config import AssetSelectorConfig, PodConfig
from src.orchestrator.models import AssetLifecycleState, LifecycleState
from src.orchestrator.pod import _MAX_BUFFER_SIZE, StrategyPod
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1). strength = abs(close-open)/open."""

    @property
    def name(self) -> str:
        return "test_simple"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = (df["close"] > df["open"]).astype(int) * 2 - 1
        strength = ((df["close"] - df["open"]).abs() / df["open"]).shift(1).fillna(0.01)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ── Helpers ───────────────────────────────────────────────────────


def _make_pod_config(**overrides: object) -> PodConfig:
    defaults: dict[str, object] = {
        "pod_id": "pod-a",
        "strategy_name": "tsmom",
        "symbols": ("BTC/USDT",),
        "initial_fraction": 0.10,
        "max_fraction": 0.40,
        "min_fraction": 0.02,
    }
    defaults.update(overrides)
    return PodConfig(**defaults)  # type: ignore[arg-type]


def _make_pod(
    *,
    symbols: tuple[str, ...] = ("BTC/USDT",),
    capital_fraction: float = 0.25,
    warmup: int = 3,
    pod_id: str = "pod-a",
) -> StrategyPod:
    """테스트용 StrategyPod 생성. warmup을 낮게 설정."""
    config = _make_pod_config(pod_id=pod_id, symbols=symbols)
    strategy = SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=capital_fraction)
    # warmup을 테스트에 유리하게 오버라이드
    pod._warmup = warmup
    return pod


def _make_bar_data(open_: float, close: float) -> dict[str, float]:
    return {
        "open": open_,
        "high": max(open_, close) * 1.01,
        "low": min(open_, close) * 0.99,
        "close": close,
        "volume": 1000.0,
    }


def _feed_warmup(pod: StrategyPod, symbol: str, n: int, base_ts: datetime) -> datetime:
    """n개 bar를 Pod에 공급하여 warmup을 채운다. 마지막 ts 반환."""
    ts = base_ts
    for i in range(n):
        bar = _make_bar_data(open_=100.0 + i, close=101.0 + i)
        pod.compute_signal(symbol, bar, ts)
        ts += timedelta(days=1)
    return ts


# ── TestPodCreation ───────────────────────────────────────────────


class TestPodCreation:
    def test_creates_with_defaults(self) -> None:
        pod = _make_pod()
        assert pod is not None

    def test_pod_id_matches_config(self) -> None:
        pod = _make_pod(pod_id="my-pod")
        assert pod.pod_id == "my-pod"

    def test_symbols_match_config(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT", "ETH/USDT"))
        assert pod.symbols == ("BTC/USDT", "ETH/USDT")

    def test_initial_state_incubation(self) -> None:
        pod = _make_pod()
        assert pod.state == LifecycleState.INCUBATION

    def test_initial_capital_fraction(self) -> None:
        pod = _make_pod(capital_fraction=0.3)
        assert pod.capital_fraction == pytest.approx(0.3)


# ── TestPodSignalComputation ─────────────────────────────────────


class TestPodSignalComputation:
    def test_no_signal_before_warmup(self) -> None:
        """warmup 미충족 시 None 반환."""
        pod = _make_pod(warmup=5)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        bar = _make_bar_data(100.0, 105.0)
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is None

    def test_signal_after_warmup(self) -> None:
        """warmup 충족 후 (direction, strength) 반환."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        # warmup 채우기 (3개 bar)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        # 3번째 bar → 시그널 생성
        bar = _make_bar_data(100.0, 105.0)
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is not None
        direction, strength = result
        assert isinstance(direction, int)
        assert isinstance(strength, float)

    def test_signal_direction_long(self) -> None:
        """close > open → LONG (+1)."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)  # close > open
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is not None
        direction, _ = result
        assert direction == 1

    def test_signal_direction_short(self) -> None:
        """close < open → SHORT (-1)."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(110.0, 100.0)  # close < open
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is not None
        direction, _ = result
        assert direction == -1

    def test_target_weight_stored(self) -> None:
        """compute_signal 후 _target_weights 업데이트."""
        pod = _make_pod(warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)
        pod.compute_signal("BTC/USDT", bar, ts)
        weights = pod.get_target_weights()
        assert "BTC/USDT" in weights
        assert weights["BTC/USDT"] != 0.0


# ── TestPodGlobalWeights ────────────────────────────────────────


class TestPodGlobalWeights:
    def test_global_weights_scaled(self) -> None:
        """global_weight = internal * capital_fraction."""
        pod = _make_pod(capital_fraction=0.5, warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)
        pod.compute_signal("BTC/USDT", bar, ts)

        internal = pod.get_target_weights()["BTC/USDT"]
        global_ = pod.get_global_weights()["BTC/USDT"]
        assert global_ == pytest.approx(internal * 0.5)

    def test_global_weights_zero_fraction(self) -> None:
        """capital_fraction=0 → 모든 global weight = 0."""
        pod = _make_pod(capital_fraction=0.0, warmup=3)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)
        bar = _make_bar_data(100.0, 110.0)
        pod.compute_signal("BTC/USDT", bar, ts)

        for v in pod.get_global_weights().values():
            assert v == pytest.approx(0.0)

    def test_global_weights_multiple_symbols(self) -> None:
        """멀티 심볼 global weights."""
        pod = _make_pod(
            symbols=("BTC/USDT", "ETH/USDT"),
            capital_fraction=0.4,
            warmup=3,
        )
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        # 두 심볼 warmup + 시그널
        for sym in ("BTC/USDT", "ETH/USDT"):
            sym_ts = ts
            for i in range(3):
                bar = _make_bar_data(100.0 + i, 101.0 + i)
                pod.compute_signal(sym, bar, sym_ts)
                sym_ts += timedelta(days=1)

        gw = pod.get_global_weights()
        assert len(gw) == 2
        assert "BTC/USDT" in gw
        assert "ETH/USDT" in gw


# ── TestPodPositionUpdate ────────────────────────────────────────


class TestPodPositionUpdate:
    def test_position_created_on_first_fill(self) -> None:
        """첫 체결 시 포지션 생성."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 0.1, 50000.0, 5.0, is_buy=True)
        assert "BTC/USDT" in pod._positions
        pos = pod._positions["BTC/USDT"]
        assert pos.notional_usd == pytest.approx(0.1 * 50000.0)
        assert pod.performance.trade_count == 1

    def test_trade_count_increments(self) -> None:
        """체결마다 trade_count 증가."""
        pod = _make_pod()
        pod.update_position("BTC/USDT", 0.1, 50000.0, 5.0, is_buy=True)
        pod.update_position("BTC/USDT", 0.05, 51000.0, 3.0, is_buy=False)
        assert pod.performance.trade_count == 2


# ── TestPodVirtualPosition ─────────────────────────────────────


class TestPodVirtualPosition:
    """Virtual Position: target weight에서 직접 포지션 설정."""

    def test_virtual_position_from_target_long(self) -> None:
        """양수 target_weight → 롱 가상 포지션 생성."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        pos = pod._positions["BTC/USDT"]
        # qty = 0.5 * 10000 / 50000 = 0.1
        assert pos.quantity == pytest.approx(0.1)
        assert pos.avg_entry_price == pytest.approx(50000.0)
        assert pos.target_weight == pytest.approx(0.5)
        assert pod.performance.trade_count == 1

    def test_virtual_position_from_target_short(self) -> None:
        """음수 target_weight → 숏 가상 포지션 생성."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        pod.update_position_from_target("BTC/USDT", -0.3, 50000.0)
        pos = pod._positions["BTC/USDT"]
        # qty = -0.3 * 10000 / 50000 = -0.06
        assert pos.quantity == pytest.approx(-0.06)
        assert pos.avg_entry_price == pytest.approx(50000.0)
        assert pod.performance.trade_count == 1

    def test_virtual_position_partial_close(self) -> None:
        """포지션 축소 시 realized PnL 발생."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        # 롱 진입 @ 50000
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        # 부분 청산 @ 55000 (가격 상승)
        pod.update_position_from_target("BTC/USDT", 0.2, 55000.0)
        pos = pod._positions["BTC/USDT"]
        # new_qty = 0.2 * 10000 / 55000 ≈ 0.0364
        # old_qty = 0.1, signed_fill = 0.0364 - 0.1 = -0.0636 (축소)
        # close_qty = min(0.0636, 0.1) = 0.0636
        # realized = (55000 - 50000) * 0.0636 = 318.18
        assert pos.realized_pnl > 0.0
        assert pod.performance.trade_count == 2

    def test_virtual_position_flip(self) -> None:
        """롱→숏 방향 전환."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        pod.update_position_from_target("BTC/USDT", -0.3, 52000.0)
        pos = pod._positions["BTC/USDT"]
        # 방향 전환 → avg_entry = flip price
        assert pos.quantity < 0
        assert pos.avg_entry_price == pytest.approx(52000.0)
        # realized = (52000 - 50000) * 0.1 = 200 (전체 롱 청산)
        assert pos.realized_pnl == pytest.approx(200.0)

    def test_virtual_position_no_change_skipped(self) -> None:
        """target 미변경 시 trade_count 증가 없음."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        assert pod.performance.trade_count == 1
        # 동일 target → 스킵
        pod.update_position_from_target("BTC/USDT", 0.5, 55000.0)
        assert pod.performance.trade_count == 1

    def test_attribute_fee(self) -> None:
        """수수료 귀속 시 realized_pnl 감소."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        pod.attribute_fee("BTC/USDT", 5.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.realized_pnl == pytest.approx(-5.0)

    def test_attribute_fee_no_position_noop(self) -> None:
        """포지션 없는 심볼에 fee → no-op."""
        pod = _make_pod()
        pod.attribute_fee("BTC/USDT", 5.0)
        assert "BTC/USDT" not in pod._positions

    def test_compute_signal_updates_virtual_position(self) -> None:
        """compute_signal() 호출 시 가상 포지션 자동 갱신."""
        pod = _make_pod(warmup=3)
        pod.set_base_equity(10000.0)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        ts = _feed_warmup(pod, "BTC/USDT", 2, ts)

        bar = _make_bar_data(100.0, 110.0)
        result = pod.compute_signal("BTC/USDT", bar, ts)
        assert result is not None

        # compute_signal이 가상 포지션을 갱신했는지 확인
        assert "BTC/USDT" in pod._positions
        pos = pod._positions["BTC/USDT"]
        # direction=+1 (close > open), target_weight > 0
        weights = pod.get_target_weights()
        assert weights["BTC/USDT"] != 0.0
        assert pos.target_weight == pytest.approx(weights["BTC/USDT"])

    def test_mtm_with_virtual_position(self) -> None:
        """가상 포지션 기반 MTM P&L 정확성."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        # 롱 진입 @ 50000
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        # qty = 0.1
        # MTM equity = 10000 + (55000 - 50000) * 0.1 = 10500
        equity = pod.compute_mtm_equity({"BTC/USDT": 55000.0})
        assert equity == pytest.approx(10500.0)

    def test_virtual_position_zero_base_equity(self) -> None:
        """base_equity=0 → target_qty=0."""
        pod = _make_pod()
        # base_equity 미설정 (0)
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(0.0)


# ── TestVirtualPositionShortModes ─────────────────────────────


class TestVirtualPositionShortModes:
    """ShortMode 3가지(DISABLED/HEDGE_ONLY/FULL) 시나리오별 가상 포지션 검증.

    각 모드에서 전략이 출력하는 (direction, strength) 패턴을 시뮬레이션하여
    update_position_from_target()이 올바르게 동작하는지 확인합니다.
    """

    def test_disabled_long_flat_long(self) -> None:
        """DISABLED: Long(+0.5) → Flat(0) → Long(+0.5).

        숏 신호 없이 롱/플랫만 전환.
        """
        pod = _make_pod()
        pod.set_base_equity(10000.0)

        # Step 1: Long 진입 @ 50000
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(0.1)  # 0.5 * 10000 / 50000
        assert pos.avg_entry_price == pytest.approx(50000.0)
        assert pos.realized_pnl == pytest.approx(0.0)

        # Step 2: Flat (신호 소멸) @ 55000 — 가격 상승 후 청산
        pod.update_position_from_target("BTC/USDT", 0.0, 55000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(0.0)
        # realized = (55000 - 50000) * 0.1 = 500
        assert pos.realized_pnl == pytest.approx(500.0)
        assert pos.avg_entry_price == pytest.approx(0.0)  # 완전 청산

        # Step 3: 다시 Long 진입 @ 53000
        pod.update_position_from_target("BTC/USDT", 0.5, 53000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity > 0
        assert pos.avg_entry_price == pytest.approx(53000.0)
        # 이전 realized 유지
        assert pos.realized_pnl == pytest.approx(500.0)
        assert pod.performance.trade_count == 3

    def test_disabled_never_negative_qty(self) -> None:
        """DISABLED: target_weight가 항상 >= 0 → quantity 음수 불가."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)

        for weight in [0.3, 0.8, 0.0, 0.5, 0.0]:
            pod.update_position_from_target("BTC/USDT", weight, 50000.0)
            pos = pod._positions["BTC/USDT"]
            assert pos.quantity >= 0.0

    def test_hedge_only_long_flat_short_long(self) -> None:
        """HEDGE_ONLY: Long(+0.5) → Flat(0) → Short(-0.24) → Long(+0.5).

        드로다운 시 감쇄된 숏 진입, 이후 롱 복귀.
        strength=0.3, hedge_strength_ratio=0.8 → 숏 weight = -0.3*0.8 = -0.24
        """
        pod = _make_pod()
        pod.set_base_equity(10000.0)

        # Step 1: Long 진입 @ 50000
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(0.1)
        assert pos.quantity > 0

        # Step 2: 시장 하락 → 시그널 소멸 → Flat @ 45000
        pod.update_position_from_target("BTC/USDT", 0.0, 45000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(0.0)
        # realized = (45000 - 50000) * 0.1 = -500 (손실)
        assert pos.realized_pnl == pytest.approx(-500.0)

        # Step 3: 드로다운 심화 → 조건부 숏 진입 (감쇄) @ 43000
        hedge_weight = -0.24  # direction=-1, strength=0.3*0.8
        pod.update_position_from_target("BTC/USDT", hedge_weight, 43000.0)
        pos = pod._positions["BTC/USDT"]
        # qty = -0.24 * 10000 / 43000 ≈ -0.0558
        assert pos.quantity < 0
        assert pos.avg_entry_price == pytest.approx(43000.0)

        # Step 4: 회복 → 롱 복귀 @ 44000
        pod.update_position_from_target("BTC/USDT", 0.5, 44000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity > 0
        # 숏 청산 realized: (43000 - 44000) * 0.0558 ≈ -55.8 (숏 손실)
        # 누적 realized: -500 + (숏 realized)
        assert pos.realized_pnl < -500.0  # 추가 손실 발생
        assert pos.avg_entry_price == pytest.approx(44000.0)
        assert pod.performance.trade_count == 4

    def test_hedge_only_short_strength_attenuated(self) -> None:
        """HEDGE_ONLY: 숏 weight 절대값이 롱보다 작음 (감쇄 반영)."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)

        # Long: weight = +0.5
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        long_qty = abs(pod._positions["BTC/USDT"].quantity)

        # Short (hedge): weight = -0.24 (감쇄)
        pod.update_position_from_target("BTC/USDT", -0.24, 50000.0)
        short_qty = abs(pod._positions["BTC/USDT"].quantity)

        # 숏 수량 < 롱 수량 (감쇄됨)
        assert short_qty < long_qty

    def test_full_long_short_long(self) -> None:
        """FULL: Long(+0.5) → Short(-0.5) → Long(+0.5).

        자유로운 양방향 전환, 감쇄 없음.
        """
        pod = _make_pod()
        pod.set_base_equity(10000.0)

        # Step 1: Long @ 50000
        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        pos = pod._positions["BTC/USDT"]
        long_qty = pos.quantity
        assert long_qty == pytest.approx(0.1)

        # Step 2: 방향 전환 Short @ 48000 (가격 하락)
        pod.update_position_from_target("BTC/USDT", -0.5, 48000.0)
        pos = pod._positions["BTC/USDT"]
        # qty = -0.5 * 10000 / 48000 ≈ -0.1042
        assert pos.quantity < 0
        assert pos.avg_entry_price == pytest.approx(48000.0)  # flip → 새 진입가
        # realized = (48000 - 50000) * 0.1 = -200 (롱 전체 청산 손실)
        assert pos.realized_pnl == pytest.approx(-200.0)

        # Step 3: 다시 Long @ 46000 (가격 추가 하락 → 숏 이익)
        pod.update_position_from_target("BTC/USDT", 0.5, 46000.0)
        pos = pod._positions["BTC/USDT"]
        assert pos.quantity > 0
        assert pos.avg_entry_price == pytest.approx(46000.0)
        # 숏 청산 realized: (48000 - 46000) * 0.1042 ≈ 208.3
        # 누적: -200 + 208.3 = +8.3
        assert pos.realized_pnl > 0  # 숏이 롱 손실을 만회
        assert pod.performance.trade_count == 3

    def test_full_symmetric_long_short(self) -> None:
        """FULL: 롱과 숏의 수량 절대값이 동일 (감쇄 없음)."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)
        price = 50000.0

        pod.update_position_from_target("BTC/USDT", 0.5, price)
        long_qty = abs(pod._positions["BTC/USDT"].quantity)

        pod.update_position_from_target("BTC/USDT", -0.5, price)
        short_qty = abs(pod._positions["BTC/USDT"].quantity)

        # 동일 가격에서 롱/숏 수량 절대값 동일
        assert long_qty == pytest.approx(short_qty)

    def test_full_consecutive_same_direction_no_extra_trade(self) -> None:
        """FULL: 연속 동일 방향 target → trade_count 1회만."""
        pod = _make_pod()
        pod.set_base_equity(10000.0)

        pod.update_position_from_target("BTC/USDT", 0.5, 50000.0)
        assert pod.performance.trade_count == 1

        # 동일 weight 반복 → 스킵
        pod.update_position_from_target("BTC/USDT", 0.5, 52000.0)
        assert pod.performance.trade_count == 1

        pod.update_position_from_target("BTC/USDT", 0.5, 48000.0)
        assert pod.performance.trade_count == 1

    def test_all_modes_mtm_consistency(self) -> None:
        """3가지 모드 공통: MTM equity = base + unrealized PnL."""
        for weight in [0.5, -0.5, 0.0]:
            pod = _make_pod()
            pod.set_base_equity(10000.0)
            if abs(weight) > 1e-8:
                pod.update_position_from_target("BTC/USDT", weight, 50000.0)

            mtm = pod.compute_mtm_equity({"BTC/USDT": 55000.0})

            if abs(weight) < 1e-8:
                assert mtm == pytest.approx(10000.0)  # 플랫
            elif weight > 0:
                assert mtm > 10000.0  # 롱 + 가격 상승 = 이익
            else:
                assert mtm < 10000.0  # 숏 + 가격 상승 = 손실


# ── TestPodHelpers ──────────────────────────────────────────────


class TestPodHelpers:
    def test_accepts_symbol_true(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        assert pod.accepts_symbol("BTC/USDT") is True

    def test_accepts_symbol_false(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        assert pod.accepts_symbol("ETH/USDT") is False

    def test_is_active_vs_retired(self) -> None:
        pod = _make_pod()
        assert pod.is_active is True
        pod.state = LifecycleState.RETIRED
        assert pod.is_active is False

    def test_buffer_size_limit(self) -> None:
        """MAX_BUFFER_SIZE 초과 시 오래된 데이터 trim."""
        pod = _make_pod(warmup=1)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(_MAX_BUFFER_SIZE + 50):
            bar = _make_bar_data(100.0 + i * 0.01, 101.0 + i * 0.01)
            pod.compute_signal("BTC/USDT", bar, ts)
            ts += timedelta(days=1)
        assert len(pod._buffers["BTC/USDT"]) == _MAX_BUFFER_SIZE

    def test_record_daily_return(self) -> None:
        pod = _make_pod()
        pod.record_daily_return(0.01)
        pod.record_daily_return(-0.005)
        assert pod.performance.live_days == 2
        assert len(pod.daily_returns_series) == 2

    def test_inject_warmup(self) -> None:
        """inject_warmup으로 버퍼 주입."""
        pod = _make_pod(warmup=5)
        ts_list = [datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(5)]
        bars = [_make_bar_data(100.0 + i, 101.0 + i) for i in range(5)]
        pod.inject_warmup("BTC/USDT", bars, ts_list)
        assert len(pod._buffers["BTC/USDT"]) == 5

    def test_inject_warmup_length_mismatch(self) -> None:
        pod = _make_pod()
        with pytest.raises(ValueError, match="length mismatch"):
            pod.inject_warmup("BTC/USDT", [_make_bar_data(100, 101)], [])

    def test_compute_signal_wrong_symbol_returns_none(self) -> None:
        """Pod에 등록되지 않은 심볼 → None."""
        pod = _make_pod(symbols=("BTC/USDT",))
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        result = pod.compute_signal("ETH/USDT", _make_bar_data(100, 101), ts)
        assert result is None


# ── TestPodAssetSelector ─────────────────────────────────────────


class TestPodAssetSelector:
    """Pod + AssetSelector 통합 테스트."""

    def _make_selector_pod(
        self,
        symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
        warmup: int = 3,
    ) -> StrategyPod:
        asc = AssetSelectorConfig(
            enabled=True,
            exclude_score_threshold=0.20,
            include_score_threshold=0.35,
            exclude_confirmation_bars=2,
            include_confirmation_bars=2,
            ramp_steps=3,
            min_active_assets=1,
            sharpe_lookback=20,
            return_lookback=10,
        )
        config = _make_pod_config(symbols=symbols, asset_selector=asc)
        strategy = SimpleTestStrategy()
        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)
        pod._warmup = warmup
        return pod

    def test_selector_initialized(self) -> None:
        pod = self._make_selector_pod()
        assert pod.asset_selector is not None
        assert len(pod.asset_selector.active_symbols) == 3

    def test_no_selector_when_disabled(self) -> None:
        asc = AssetSelectorConfig(enabled=False)
        config = _make_pod_config(asset_selector=asc)
        strategy = SimpleTestStrategy()
        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)
        assert pod.asset_selector is None

    def test_no_selector_when_none(self) -> None:
        pod = _make_pod()
        assert pod.asset_selector is None

    def test_excluded_asset_returns_zero_strength(self) -> None:
        """제외된 에셋은 strength=0 반환."""
        pod = self._make_selector_pod(symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"))
        assert pod._asset_selector is not None

        # Force BTC to COOLDOWN (multiplier=0)
        pod._asset_selector._states["BTC/USDT"].state = AssetLifecycleState.COOLDOWN
        pod._asset_selector._states["BTC/USDT"].multiplier = 0.0

        # _apply_asset_weight should return 0 for excluded asset
        result = pod._apply_asset_weight("BTC/USDT", 0.8)
        assert result == 0.0

        # ETH still active → non-zero (1/N normalized)
        result_eth = pod._apply_asset_weight("ETH/USDT", 0.8)
        assert result_eth > 0.0
        # 3 symbols, no allocator, selector_mult=1.0 → 0.8 / 3
        assert result_eth == pytest.approx(0.8 / 3)

    def test_serialization_round_trip(self) -> None:
        """Pod + selector 직렬화 왕복."""
        pod = self._make_selector_pod()
        assert pod._asset_selector is not None

        # Modify selector state
        pod._asset_selector._states["BTC/USDT"].state = AssetLifecycleState.COOLDOWN
        pod._asset_selector._states["BTC/USDT"].multiplier = 0.0

        data = pod.to_dict()
        assert data.get("asset_selector") is not None

        # Restore
        pod2 = self._make_selector_pod()
        pod2.restore_from_dict(data)
        assert pod2._asset_selector is not None
        assert pod2._asset_selector._states["BTC/USDT"].state == AssetLifecycleState.COOLDOWN
        assert pod2._asset_selector._states["BTC/USDT"].multiplier == 0.0


class TestApplyAssetWeightNormalization:
    """_apply_asset_weight 1/N 정규화 테스트."""

    def test_no_allocator_divides_by_n(self) -> None:
        """No allocator → strength / N 반환."""
        symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT")
        pod = _make_pod(symbols=symbols)
        # No allocator, no selector → strength / 4
        result = pod._apply_asset_weight("BTC/USDT", 0.8)
        assert result == pytest.approx(0.8 / 4)

    def test_no_allocator_single_symbol(self) -> None:
        """1-symbol pod → strength / 1 = strength."""
        pod = _make_pod(symbols=("BTC/USDT",))
        result = pod._apply_asset_weight("BTC/USDT", 0.8)
        assert result == pytest.approx(0.8)

    def test_no_allocator_total_sums_to_strength(self) -> None:
        """전 심볼 합산 = strength (equal weight)."""
        symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT")
        pod = _make_pod(symbols=symbols)
        strength = 0.8
        total = sum(pod._apply_asset_weight(s, strength) for s in symbols)
        assert total == pytest.approx(strength)

    def test_with_allocator_no_n_multiply(self) -> None:
        """Allocator weight * strength (no *N multiply)."""
        from src.orchestrator.asset_allocator import (
            AssetAllocationConfig,
            IntraPodAllocator,
        )

        symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
        pod = _make_pod(symbols=symbols)
        alloc_config = AssetAllocationConfig()
        allocator = IntraPodAllocator(config=alloc_config, symbols=symbols)
        pod._asset_allocator = allocator
        # equal_weight allocator → each weight = 1/3
        result = pod._apply_asset_weight("BTC/USDT", 0.9)
        assert result == pytest.approx(0.9 / 3)

    def test_with_allocator_total_sums_to_strength(self) -> None:
        """Allocator 사용 시에도 전 심볼 합산 ≈ strength."""
        from src.orchestrator.asset_allocator import (
            AssetAllocationConfig,
            IntraPodAllocator,
        )

        symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
        pod = _make_pod(symbols=symbols)
        alloc_config = AssetAllocationConfig()
        allocator = IntraPodAllocator(config=alloc_config, symbols=symbols)
        pod._asset_allocator = allocator
        strength = 0.6
        total = sum(pod._apply_asset_weight(s, strength) for s in symbols)
        assert total == pytest.approx(strength)
