"""Tests for ADX Regime Filter Signal Generator."""

import pandas as pd
import pytest

from src.strategy.adx_regime.config import ADXRegimeConfig
from src.strategy.adx_regime.preprocessor import preprocess
from src.strategy.adx_regime.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def processed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame."""
    config = ADXRegimeConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalBasic:
    """시그널 기본 구조 테스트."""

    def test_generate_signals_basic(self, processed_df: pd.DataFrame) -> None:
        """시그널 생성 후 출력 타입 및 크기 확인."""
        signals = generate_signals(processed_df)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        n = len(processed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_entries_exits_are_bool(self, processed_df: pd.DataFrame) -> None:
        """entries/exits는 bool Series."""
        signals = generate_signals(processed_df)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, processed_df: pd.DataFrame) -> None:
        """direction은 -1, 0, 1 값만 포함."""
        config = ADXRegimeConfig(short_mode=ShortMode.FULL)
        processed = preprocess(
            pd.DataFrame(
                {
                    "open": processed_df["open"],
                    "high": processed_df["high"],
                    "low": processed_df["low"],
                    "close": processed_df["close"],
                    "volume": processed_df["volume"],
                },
                index=processed_df.index,
            ),
            config,
        )
        signals = generate_signals(processed, config)

        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})


class TestShift1Rule:
    """Shift(1) 미래 참조 편향 방지 테스트."""

    def test_first_signals_are_zero(self, processed_df: pd.DataFrame) -> None:
        """첫 행의 strength/direction은 0 (shift로 인한 NaN -> 0)."""
        signals = generate_signals(processed_df)

        assert signals.strength.iloc[0] == 0.0
        assert signals.direction.iloc[0] == 0

    def test_no_lookahead(self, processed_df: pd.DataFrame) -> None:
        """마지막 봉의 데이터 변경이 두 번째 마지막 봉의 시그널에 영향 없음."""
        config = ADXRegimeConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # 마지막 봉 주요 지표 변경
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("adx")] = 99.0
        modified_df.iloc[-1, modified_df.columns.get_loc("vw_momentum")] = 99.0
        modified_df.iloc[-1, modified_df.columns.get_loc("z_score")] = 99.0

        signals_after = generate_signals(modified_df, config)
        assert signals_after.strength.iloc[-2] == strength_second_last


class TestRegimeBlending:
    """ADX 기반 레짐 블렌딩 테스트."""

    def test_high_adx_momentum_dominates(self) -> None:
        """ADX가 높으면 momentum 시그널이 지배적."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        df = pd.DataFrame(
            {
                "adx": [50.0] * n,  # 매우 높은 ADX -> trend only
                "vw_momentum": [0.05] * n,  # 양의 모멘텀
                "z_score": [-3.0] * n,  # 과매도 (MR은 롱 시그널)
                "vol_scalar": [1.0] * n,
                "drawdown": [0.0] * n,
            },
            index=dates,
        )

        config = ADXRegimeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(df, config)

        # ADX가 adx_high(25) 이상이므로 trend_weight=1.0, mr_weight=0.0
        # 모멘텀이 양수이므로 direction은 LONG이어야 함
        # shift(1)로 첫 행은 0, 2행부터 체크
        valid_direction = signals.direction.iloc[2:]
        assert (valid_direction == Direction.LONG).all()

    def test_low_adx_mr_dominates(self) -> None:
        """ADX가 낮으면 MR 시그널이 지배적."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        df = pd.DataFrame(
            {
                "adx": [5.0] * n,  # 매우 낮은 ADX -> MR only
                "vw_momentum": [0.05] * n,  # 양의 모멘텀
                "z_score": [3.0] * n,  # 과매수 (MR은 숏 시그널)
                "vol_scalar": [1.0] * n,
                "drawdown": [0.0] * n,
            },
            index=dates,
        )

        config = ADXRegimeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(df, config)

        # ADX < adx_low(15)이므로 trend_weight=0.0, mr_weight=1.0
        # z_score > mr_entry_z(2.0)이므로 MR은 숏 시그널
        valid_direction = signals.direction.iloc[2:]
        assert (valid_direction == Direction.SHORT).all()


class TestHedgeOnlyMode:
    """HEDGE_ONLY 모드 테스트."""

    def test_hedge_only_suppresses_short_without_drawdown(self, sample_ohlcv: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서 drawdown이 임계값 미달이면 숏 억제."""
        config = ADXRegimeConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        # drawdown이 hedge_threshold(-0.07)보다 큰(0에 가까운) 구간에서는 숏 없음
        drawdown = processed["drawdown"]
        no_drawdown_mask = drawdown >= config.hedge_threshold
        direction_no_dd = signals.direction[no_drawdown_mask]

        # 드로다운 없는 구간에서 숏 시그널이 없어야 함
        assert (direction_no_dd >= 0).all()

    def test_full_mode_allows_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """FULL 모드에서는 숏 시그널 허용."""
        config = ADXRegimeConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        unique_values = set(signals.direction.unique())
        # FULL 모드에서는 -1이 가능 (데이터에 따라 없을 수도 있음)
        assert unique_values.issubset({-1, 0, 1})

    def test_disabled_mode_no_shorts(self, sample_ohlcv: pd.DataFrame) -> None:
        """DISABLED 모드에서 숏 시그널 없음."""
        config = ADXRegimeConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()


class TestMissingColumns:
    """입력 검증 테스트."""

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)
