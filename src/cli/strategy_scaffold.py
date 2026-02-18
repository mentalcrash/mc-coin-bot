"""전략 스캐폴딩 CLI — 4-file 전략 boilerplate 자동 생성.

Usage:
    uv run mcbot strategy scaffold hurst-adaptive \\
        --tf 1D --short-mode HEDGE_ONLY \\
        --indicators "hurst_exponent,efficiency_ratio" \\
        --data-sources ohlcv
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(name="strategy", help="Strategy scaffolding tools", no_args_is_help=True)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STRATEGY_DIR = _PROJECT_ROOT / "src" / "strategy"
_TEST_DIR = _PROJECT_ROOT / "tests" / "strategy"
_INIT_FILE = _STRATEGY_DIR / "__init__.py"


def _to_snake(name: str) -> str:
    """kebab-case → snake_case."""
    return name.replace("-", "_")


def _to_class(name: str) -> str:
    """kebab-case → PascalCase."""
    return "".join(w.capitalize() for w in name.split("-"))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _render_init(snake: str, class_name: str) -> str:
    return f'''"""{class_name} Strategy."""

from src.strategy.{snake}.config import {class_name}Config
from src.strategy.{snake}.preprocessor import preprocess
from src.strategy.{snake}.signal import generate_signals
from src.strategy.{snake}.strategy import {class_name}Strategy

__all__ = [
    "{class_name}Config",
    "{class_name}Strategy",
    "generate_signals",
    "preprocess",
]
'''


def _render_config(
    snake: str,
    class_name: str,
    short_mode: str,
    indicators: list[str],
) -> str:
    indicator_comment = ", ".join(indicators) if indicators else "none"
    return f'''"""{class_name} Strategy Configuration."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class {class_name}Config(BaseModel):
    """Configuration for {class_name} strategy.

    Indicators: {indicator_comment}
    """

    model_config = ConfigDict(frozen=True)

    # --- Core parameters (sweep targets) ---
    lookback: int = Field(default=20, ge=5, le=200, description="Primary lookback window")
    vol_target: float = Field(default=0.35, ge=0.05, le=1.0, description="Annualized vol target")

    # --- Fixed parameters ---
    vol_window: int = Field(default=30, ge=5, le=200, description="Volatility window")
    min_volatility: float = Field(default=0.05, ge=0.01, le=0.50, description="Min vol clamp")
    annualization_factor: float = Field(default=365.0, description="Annualization factor")
    short_mode: ShortMode = ShortMode.{short_mode}

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Config 일관성 검증."""
        return self

    def warmup_periods(self) -> int:
        """필요 최소 캔들 수."""
        return max(self.lookback, self.vol_window) + 1
'''


def _render_preprocessor(
    snake: str,
    class_name: str,
    indicators: list[str],
) -> str:
    imports = ""
    body_lines: list[str] = []
    for ind in indicators:
        imports += f"    {ind},\n"
        body_lines.append(f'    result["{ind}"] = {ind}(close_series)  # TODO: add params')

    indicator_block = (
        "\n".join(body_lines) if body_lines else "    # TODO: add indicator calculations"
    )

    return f'''"""{class_name} Preprocessor — vectorized indicator computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.market.indicators import (
    log_returns,
    realized_volatility,
{imports})
from src.strategy.{snake}.config import {class_name}Config


def preprocess(df: pd.DataFrame, config: {class_name}Config) -> pd.DataFrame:
    """지표 사전 계산 (벡터화).

    Args:
        df: OHLCV DataFrame
        config: 전략 설정

    Returns:
        지표 컬럼이 추가된 DataFrame
    """
    required = {{"close", "high", "low"}}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing columns: {{missing}}"
        raise ValueError(msg)

    result = df.copy()
    close_series: pd.Series = result["close"]  # type: ignore[assignment]

    # Returns & Volatility
    returns = log_returns(close_series)
    result["realized_vol"] = realized_volatility(
        returns, window=config.vol_window, annualization_factor=config.annualization_factor,
    )
    clamped_vol = np.maximum(result["realized_vol"], config.min_volatility)
    result["vol_scalar"] = config.vol_target / clamped_vol

    # Indicators
{indicator_block}

    logger.debug(
        "{class_name} preprocess | bars={{}}",
        len(result),
    )
    return result
'''


def _render_signal(snake: str, class_name: str) -> str:
    return f'''"""{class_name} Signal Generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.models.types import Direction
from src.strategy.{snake}.config import {class_name}Config
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import StrategySignals


def generate_signals(
    df: pd.DataFrame,
    config: {class_name}Config | None = None,
) -> StrategySignals:
    """시그널 생성.

    Args:
        df: preprocess() 결과 DataFrame
        config: 전략 설정

    Returns:
        StrategySignals (entries, exits, direction, strength)
    """
    if config is None:
        config = {class_name}Config()

    # TODO: implement signal logic
    raw_direction = pd.Series(0, index=df.index, dtype=int)
    raw_strength = pd.Series(0.0, index=df.index)

    # Shift(1) — lookahead bias 방지
    signal_shifted = raw_strength.shift(1).fillna(0.0)

    direction = pd.Series(
        np.sign(signal_shifted).fillna(0).astype(int),
        index=df.index,
        name="direction",
    )
    strength = pd.Series(signal_shifted, index=df.index, name="strength")

    # ShortMode
    if config.short_mode == ShortMode.DISABLED:
        short_mask = direction == Direction.SHORT
        direction = direction.where(~short_mask, Direction.NEUTRAL)
        strength = strength.where(~short_mask, 0.0)

    # Entry / Exit
    prev_dir = direction.shift(1).fillna(0)
    long_entry = (direction == Direction.LONG) & (prev_dir != Direction.LONG)
    short_entry = (direction == Direction.SHORT) & (prev_dir != Direction.SHORT)
    entries = pd.Series(long_entry | short_entry, index=df.index, name="entries")

    to_neutral = (direction == Direction.NEUTRAL) & (prev_dir != Direction.NEUTRAL)
    reversal = direction * prev_dir < 0
    exits = pd.Series(to_neutral | reversal, index=df.index, name="exits")

    n_long = int((direction == Direction.LONG).sum())
    n_short = int((direction == Direction.SHORT).sum())
    logger.info("{class_name} signals | Long: {{}}, Short: {{}}", n_long, n_short)

    return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)
'''


def _render_strategy(snake: str, class_name: str, kebab: str) -> str:
    return f'''"""{class_name} Strategy Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.{snake}.config import {class_name}Config
from src.strategy.{snake}.preprocessor import preprocess
from src.strategy.{snake}.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("{kebab}")
class {class_name}Strategy(BaseStrategy):
    """{class_name} Strategy.

    Example:
        >>> strategy = {class_name}Strategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: {class_name}Config | None = None) -> None:
        self._config = config or {class_name}Config()

    @classmethod
    def from_params(cls, **params: Any) -> {class_name}Strategy:
        """파라미터로 인스턴스 생성."""
        config = {class_name}Config(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "{class_name}"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> {class_name}Config:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """권장 PortfolioManagerConfig."""
        return {{
            "max_leverage_cap": 1.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }}

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        return {{
            "lookback": f"{{cfg.lookback}}",
            "vol_target": f"{{cfg.vol_target:.0%}}",
            "mode": cfg.short_mode.name,
        }}
'''


def _render_test(snake: str, class_name: str, kebab: str) -> str:
    return f'''"""Tests for {class_name} strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.{snake}.config import {class_name}Config
from src.strategy.{snake}.preprocessor import preprocess
from src.strategy.{snake}.signal import generate_signals
from src.strategy.{snake}.strategy import {class_name}Strategy


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200일 OHLCV 샘플."""
    np.random.seed(42)
    n = 200
    base = 50000.0
    close = base + np.cumsum(np.random.randn(n) * 300)
    close = np.maximum(close, base * 0.5)
    return pd.DataFrame(
        {{
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }},
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestConfig:
    """Config 검증."""

    def test_default(self) -> None:
        cfg = {class_name}Config()
        assert cfg.vol_target > 0
        assert cfg.warmup_periods() > 0

    def test_from_params(self) -> None:
        s = {class_name}Strategy.from_params(vol_target=0.25)
        assert s.config.vol_target == 0.25


class TestPreprocess:
    """Preprocessor 검증."""

    def test_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = {class_name}Config()
        result = preprocess(sample_ohlcv, cfg)
        assert "realized_vol" in result.columns
        assert "vol_scalar" in result.columns

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({{"foo": [1, 2, 3]}})
        with pytest.raises(ValueError, match="Missing"):
            preprocess(df, {class_name}Config())


class TestSignal:
    """Signal 생성 검증."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = {class_name}Config()
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        assert len(signals.direction) == len(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.exits) == len(sample_ohlcv)

    def test_direction_values(self, sample_ohlcv: pd.DataFrame) -> None:
        cfg = {class_name}Config()
        processed = preprocess(sample_ohlcv, cfg)
        signals = generate_signals(processed, cfg)
        assert set(signals.direction.unique()).issubset({{-1, 0, 1}})


class TestStrategy:
    """Strategy 클래스 검증."""

    def test_registry_name(self) -> None:
        from src.strategy.registry import get_strategy
        cls = get_strategy("{kebab}")
        assert cls is {class_name}Strategy

    def test_run(self, sample_ohlcv: pd.DataFrame) -> None:
        strategy = {class_name}Strategy()
        processed, signals = strategy.run(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)

    def test_name(self) -> None:
        assert {class_name}Strategy().name == "{class_name}"

    def test_required_columns(self) -> None:
        s = {class_name}Strategy()
        assert "close" in s.required_columns
'''


# ---------------------------------------------------------------------------
# CLI Command
# ---------------------------------------------------------------------------


@app.command()
def scaffold(
    name: str = typer.Argument(help="전략 이름 (kebab-case, e.g. hurst-adaptive)"),
    tf: str = typer.Option("1D", "--tf", help="Target timeframe"),
    short_mode: str = typer.Option("HEDGE_ONLY", "--short-mode", help="DISABLED|HEDGE_ONLY|FULL"),
    indicators_csv: str = typer.Option("", "--indicators", help="Comma-separated indicator names"),
    data_sources: str = typer.Option("ohlcv", "--data-sources", help="ohlcv|derivatives|onchain"),
) -> None:
    """전략 boilerplate 4-file 구조 생성."""
    snake = _to_snake(name)
    class_name = _to_class(name)
    indicators = [i.strip() for i in indicators_csv.split(",") if i.strip()]

    # Validate short_mode
    valid_modes = {"DISABLED", "HEDGE_ONLY", "FULL"}
    if short_mode not in valid_modes:
        typer.echo(f"Invalid short_mode: {short_mode}. Choose from {valid_modes}")
        raise typer.Exit(code=1)

    # Create directories
    strategy_dir = _STRATEGY_DIR / snake
    test_dir = _TEST_DIR / snake

    if strategy_dir.exists():
        typer.echo(f"Strategy directory already exists: {strategy_dir}")
        raise typer.Exit(code=1)

    strategy_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Write files
    files: dict[Path, str] = {
        strategy_dir / "__init__.py": _render_init(snake, class_name),
        strategy_dir / "config.py": _render_config(snake, class_name, short_mode, indicators),
        strategy_dir / "preprocessor.py": _render_preprocessor(snake, class_name, indicators),
        strategy_dir / "signal.py": _render_signal(snake, class_name),
        strategy_dir / "strategy.py": _render_strategy(snake, class_name, name),
        test_dir / f"test_{snake}.py": _render_test(snake, class_name, name),
    }

    for path, content in files.items():
        path.write_text(content)
        logger.info("Created: {}", path.relative_to(_PROJECT_ROOT))

    # Append import to strategy __init__.py
    import_line = f"import src.strategy.{snake}  # 전략 등록 side effect\n"
    init_content = _INIT_FILE.read_text()

    # Insert before 'from src.strategy.base' line (alphabetical order)
    marker = "from src.strategy.base import BaseStrategy"
    if import_line.strip() not in init_content:
        init_content = init_content.replace(marker, f"{import_line}{marker}")
        _INIT_FILE.write_text(init_content)
        logger.info("Updated: src/strategy/__init__.py")

    typer.echo(f"Scaffolded '{name}' strategy ({len(files)} files)")
    typer.echo(f"  src/strategy/{snake}/")
    typer.echo(f"  tests/strategy/{snake}/")
    typer.echo("\nNext: implement preprocessor.py and signal.py")
