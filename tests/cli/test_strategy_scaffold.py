"""Tests for strategy scaffold CLI."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.cli.strategy_scaffold import (
    _render_config,
    _render_init,
    _render_preprocessor,
    _render_signal,
    _render_strategy,
    _render_test,
    _to_class,
    _to_snake,
)


class TestNameConversion:
    """이름 변환 유틸리티."""

    def test_to_snake(self) -> None:
        assert _to_snake("hurst-adaptive") == "hurst_adaptive"
        assert _to_snake("oi-diverge") == "oi_diverge"

    def test_to_class(self) -> None:
        assert _to_class("hurst-adaptive") == "HurstAdaptive"
        assert _to_class("oi-diverge") == "OiDiverge"


class TestTemplateValidity:
    """생성된 템플릿이 유효한 Python인지 검증."""

    @pytest.fixture
    def params(self) -> dict[str, str | list[str]]:
        return {
            "snake": "test_strat",
            "class_name": "TestStrat",
            "kebab": "test-strat",
            "short_mode": "HEDGE_ONLY",
            "indicators": ["rsi", "atr"],
        }

    def test_init_valid_python(self, params: dict[str, str | list[str]]) -> None:
        code = _render_init(str(params["snake"]), str(params["class_name"]))
        ast.parse(code)

    def test_config_valid_python(self, params: dict[str, str | list[str]]) -> None:
        code = _render_config(
            str(params["snake"]),
            str(params["class_name"]),
            str(params["short_mode"]),
            list(params["indicators"]),  # type: ignore[arg-type]
        )
        ast.parse(code)

    def test_preprocessor_valid_python(self, params: dict[str, str | list[str]]) -> None:
        code = _render_preprocessor(
            str(params["snake"]),
            str(params["class_name"]),
            list(params["indicators"]),  # type: ignore[arg-type]
        )
        ast.parse(code)

    def test_signal_valid_python(self, params: dict[str, str | list[str]]) -> None:
        code = _render_signal(str(params["snake"]), str(params["class_name"]))
        ast.parse(code)

    def test_strategy_valid_python(self, params: dict[str, str | list[str]]) -> None:
        code = _render_strategy(
            str(params["snake"]),
            str(params["class_name"]),
            str(params["kebab"]),
        )
        ast.parse(code)

    def test_test_valid_python(self, params: dict[str, str | list[str]]) -> None:
        code = _render_test(
            str(params["snake"]),
            str(params["class_name"]),
            str(params["kebab"]),
        )
        ast.parse(code)


class TestScaffoldRegister:
    """strategy.py에 @register 포함 확인."""

    def test_register_name(self) -> None:
        code = _render_strategy("hurst_adaptive", "HurstAdaptive", "hurst-adaptive")
        assert '@register("hurst-adaptive")' in code

    def test_class_name(self) -> None:
        code = _render_strategy("hurst_adaptive", "HurstAdaptive", "hurst-adaptive")
        assert "class HurstAdaptiveStrategy(BaseStrategy):" in code


class TestScaffoldCreateFiles:
    """scaffold 커맨드가 파일을 올바르게 생성하는지 확인."""

    def test_scaffold_creates_files(self, tmp_path: Path) -> None:
        """임시 디렉토리에서 파일 생성 확인."""
        import src.cli.strategy_scaffold as mod

        strategy_dir = tmp_path / "src" / "strategy"
        test_dir = tmp_path / "tests" / "strategy"
        init_file = strategy_dir / "__init__.py"

        strategy_dir.mkdir(parents=True)
        init_file.write_text("from src.strategy.base import BaseStrategy\n")

        orig = mod._STRATEGY_DIR, mod._TEST_DIR, mod._INIT_FILE, mod._PROJECT_ROOT
        try:
            mod._STRATEGY_DIR = strategy_dir
            mod._TEST_DIR = test_dir
            mod._INIT_FILE = init_file
            mod._PROJECT_ROOT = tmp_path

            mod.scaffold(
                name="test-strat",
                tf="1D",
                short_mode="FULL",
                indicators_csv="rsi,atr",
                data_sources="ohlcv",
            )

            snake = mod._to_snake("test-strat")
            assert (strategy_dir / snake / "__init__.py").exists()
            assert (strategy_dir / snake / "config.py").exists()
            assert (strategy_dir / snake / "preprocessor.py").exists()
            assert (strategy_dir / snake / "signal.py").exists()
            assert (strategy_dir / snake / "strategy.py").exists()
            assert (test_dir / snake / f"test_{snake}.py").exists()
        finally:
            mod._STRATEGY_DIR, mod._TEST_DIR, mod._INIT_FILE, mod._PROJECT_ROOT = orig

    def test_scaffold_init_import(self, tmp_path: Path) -> None:
        """__init__.py에 import 라인이 추가되는지 확인."""
        import src.cli.strategy_scaffold as mod

        strategy_dir = tmp_path / "src" / "strategy"
        test_dir = tmp_path / "tests" / "strategy"
        init_file = strategy_dir / "__init__.py"

        strategy_dir.mkdir(parents=True)
        init_file.write_text("from src.strategy.base import BaseStrategy\n")

        orig = mod._STRATEGY_DIR, mod._TEST_DIR, mod._INIT_FILE, mod._PROJECT_ROOT
        try:
            mod._STRATEGY_DIR = strategy_dir
            mod._TEST_DIR = test_dir
            mod._INIT_FILE = init_file
            mod._PROJECT_ROOT = tmp_path

            mod.scaffold(
                name="my-strat",
                tf="1D",
                short_mode="HEDGE_ONLY",
                indicators_csv="",
                data_sources="ohlcv",
            )

            content = init_file.read_text()
            assert "import src.strategy.my_strat" in content
        finally:
            mod._STRATEGY_DIR, mod._TEST_DIR, mod._INIT_FILE, mod._PROJECT_ROOT = orig
