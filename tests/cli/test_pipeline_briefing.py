"""Tests for pipeline p1-briefing and next CLI commands."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.pipeline import _PHASE_ACTIONS, _print_anti_patterns, app
from src.pipeline.models import (
    PhaseId,
    PhaseResult,
    PhaseVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.store import StrategyStore

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    name: str = "test-strat",
    display_name: str = "Test Strategy",
    status: StrategyStatus = StrategyStatus.TESTING,
    timeframe: str = "12H",
    phases: dict[PhaseId, PhaseResult] | None = None,
) -> StrategyRecord:
    return StrategyRecord(
        meta=StrategyMeta(
            name=name,
            display_name=display_name,
            category="Trend Following",
            timeframe=timeframe,
            short_mode="HEDGE_ONLY",
            status=status,
            created_at=date(2026, 1, 1),
        ),
        phases=phases or {},
    )


@pytest.fixture()
def strategy_dir(tmp_path: Path) -> Path:
    """전략 YAML이 저장될 임시 디렉토리."""
    store = StrategyStore(base_dir=tmp_path)
    # Active strategy (all phases PASS)
    store.save(
        _make_record(
            name="anchor-mom",
            display_name="Anchor Mom",
            status=StrategyStatus.ACTIVE,
            phases={
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P2: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P3: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P4: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P5: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P6: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P7: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            },
        )
    )
    # Testing strategy (P1-P3 PASS, needs P4)
    store.save(
        _make_record(
            name="squeeze-adaptive",
            display_name="Squeeze Adaptive",
            status=StrategyStatus.TESTING,
            phases={
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P2: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P3: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            },
        )
    )
    # Retired strategy (FAIL at P4)
    store.save(
        _make_record(
            name="failed-strat",
            display_name="Failed Strategy",
            status=StrategyStatus.RETIRED,
            phases={
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P4: PhaseResult(status=PhaseVerdict.FAIL, date=date(2026, 1, 1)),
            },
        )
    )
    return tmp_path


@pytest.fixture()
def lesson_dir(tmp_path: Path) -> Path:
    """교훈 YAML이 저장될 임시 디렉토리."""
    import yaml

    lesson_path = tmp_path / "lessons"
    lesson_path.mkdir()
    lesson1 = {
        "id": 1,
        "title": "ML Look-Ahead Bias",
        "body": "Test lesson body.",
        "category": "strategy-design",
        "tags": ["ML"],
        "strategies": ["ctrend"],
        "timeframes": ["12H"],
        "added_at": "2026-01-01",
    }
    lesson2 = {
        "id": 2,
        "title": "4H TF Death",
        "body": "4H 전략 전멸.",
        "category": "data-resolution",
        "tags": ["4H"],
        "strategies": [],
        "timeframes": ["4H"],
        "added_at": "2026-01-02",
    }
    lesson3 = {
        "id": 3,
        "title": "Single Indicator Decay",
        "body": "단일 지표 trend-following 과적합.",
        "category": "strategy-design",
        "tags": [],
        "strategies": [],
        "timeframes": [],
        "added_at": "2026-01-03",
    }
    for lesson in [lesson1, lesson2, lesson3]:
        path = lesson_path / f"{lesson['id']:03d}.yaml"
        path.write_text(yaml.dump(lesson, allow_unicode=True), encoding="utf-8")
    return lesson_path


# ---------------------------------------------------------------------------
# TestPhaseActions
# ---------------------------------------------------------------------------


class TestPhaseActions:
    """_PHASE_ACTIONS 매핑 테스트."""

    def test_all_phases_covered(self) -> None:
        """P1~P7 전부 매핑."""
        for pid in PhaseId:
            assert pid.value in _PHASE_ACTIONS, f"{pid} not in _PHASE_ACTIONS"

    def test_action_tuple_format(self) -> None:
        """각 엔트리가 (cli_cmd, skill_cmd) 튜플."""
        for phase, (cli, skill) in _PHASE_ACTIONS.items():
            assert isinstance(cli, str), f"{phase} cli is not str"
            assert isinstance(skill, str), f"{phase} skill is not str"
            assert skill.startswith("/"), f"{phase} skill should start with /"


# ---------------------------------------------------------------------------
# TestPrintAntiPatterns
# ---------------------------------------------------------------------------


class TestPrintAntiPatterns:
    """_print_anti_patterns 헬퍼 테스트."""

    SAMPLE_MD = """\
### 실패 패턴 요약 (빠른 참조)

| 패턴 | 해당 전략 | 결론 |
|------|----------|------|
| 단일 지표 trend-following | TSMOM, Enhanced | OOS Decay 56~92% |
| 동일 TF Mom+MR 블렌딩 | Mom-MR Blend | Alpha 상쇄 |
"""

    def test_extracts_patterns(self, capsys: pytest.CaptureFixture[str]) -> None:
        """패턴 테이블에서 행 추출."""
        _print_anti_patterns(self.SAMPLE_MD)
        captured = capsys.readouterr()
        assert "단일 지표 trend-following" in captured.out
        assert "Alpha 상쇄" in captured.out

    def test_no_marker(self, capsys: pytest.CaptureFixture[str]) -> None:
        """실패 패턴 요약 마커 없을 때."""
        _print_anti_patterns("No marker here")
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ---------------------------------------------------------------------------
# TestNextCommand
# ---------------------------------------------------------------------------


class TestNextCommand:
    """pipeline next 명령 테스트."""

    def test_next_table(self, strategy_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--name 없이 실행 시 테이블 출력."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )
        result = runner.invoke(app, ["next"])
        assert result.exit_code == 0
        assert "Next Actions" in result.output
        assert "squeeze-adaptive" in result.output

    def test_next_detail(self, strategy_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--name 지정 시 상세 출력."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )
        result = runner.invoke(app, ["next", "--name", "squeeze-adaptive"])
        assert result.exit_code == 0
        assert "P4" in result.output
        assert "squeeze-adaptive" in result.output

    def test_next_complete_strategy(
        self, strategy_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """완료 전략 상세 출력."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )
        result = runner.invoke(app, ["next", "--name", "anchor-mom"])
        assert result.exit_code == 0
        assert "COMPLETE" in result.output

    def test_next_blocked_strategy(
        self, strategy_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAIL 전략 상세 출력."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )
        result = runner.invoke(app, ["next", "--name", "failed-strat"])
        assert result.exit_code == 0
        assert "BLOCKED" in result.output

    def test_next_not_found(self, strategy_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """존재하지 않는 전략."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )
        result = runner.invoke(app, ["next", "--name", "nonexistent"])
        assert result.exit_code == 1

    def test_next_status_filter(self, strategy_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--status 필터."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )
        result = runner.invoke(app, ["next", "--status", "TESTING"])
        assert result.exit_code == 0
        assert "squeeze-adaptive" in result.output
        # ACTIVE/RETIRED strategies should not appear in actionable table
        assert "anchor-mom" not in result.output


# ---------------------------------------------------------------------------
# TestP1BriefingCommand
# ---------------------------------------------------------------------------


class TestP1BriefingCommand:
    """pipeline p1-briefing 명령 테스트."""

    def test_briefing_sections(
        self,
        strategy_dir: Path,
        lesson_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """4개 섹션 모두 출력."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )

        from src.pipeline.lesson_store import LessonStore

        monkeypatch.setattr(
            "src.pipeline.lesson_store.LessonStore",
            lambda base_dir=lesson_dir: LessonStore(base_dir=base_dir),
        )

        # discarded-strategies.md 임시 파일
        discarded_dir = tmp_path / ".claude" / "skills" / "p1-research" / "references"
        discarded_dir.mkdir(parents=True)
        discarded_file = discarded_dir / "discarded-strategies.md"
        discarded_file.write_text(
            "### 실패 패턴 요약\n\n"
            "| 패턴 | 해당 전략 | 결론 |\n"
            "|------|----------|------|\n"
            "| 단일 지표 | TSMOM | Decay |\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        # Mock DataCatalogStore to avoid reading real catalogs
        from src.catalog.models import DatasetEntry, DataType

        mock_datasets = [
            DatasetEntry(
                id="btc_metrics",
                name="BTC On-chain Metrics",
                data_type=DataType.ONCHAIN,
                source_id="coinmetrics",
            ),
            DatasetEntry(
                id="funding_rate",
                name="Funding Rate",
                data_type=DataType.DERIVATIVES,
                source_id="binance_futures",
            ),
            DatasetEntry(
                id="ohlcv_1m",
                name="OHLCV 1m",
                data_type=DataType.OHLCV,
                source_id="binance_spot",
            ),
        ]

        from unittest.mock import MagicMock

        mock_catalog = MagicMock()
        mock_catalog.load_all.return_value = mock_datasets
        monkeypatch.setattr("src.catalog.store.DataCatalogStore", lambda **_kw: mock_catalog)

        result = runner.invoke(app, ["p1-briefing", "--tf", "12H"])
        assert result.exit_code == 0
        assert "P1 Briefing: 12H" in result.output
        assert "Pipeline Status" in result.output
        assert "Lessons for 12H" in result.output
        assert "Anti-Patterns" in result.output
        assert "Alternative Data Sources" in result.output

    def test_briefing_lesson_counts(
        self,
        strategy_dir: Path,
        lesson_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """12H TF 교훈 + strategy-design 교훈 합집합."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )

        from src.pipeline.lesson_store import LessonStore

        monkeypatch.setattr(
            "src.pipeline.lesson_store.LessonStore",
            lambda base_dir=lesson_dir: LessonStore(base_dir=base_dir),
        )
        monkeypatch.chdir(tmp_path)

        from unittest.mock import MagicMock

        mock_catalog = MagicMock()
        mock_catalog.load_all.return_value = []
        monkeypatch.setattr("src.catalog.store.DataCatalogStore", lambda **_kw: mock_catalog)

        result = runner.invoke(app, ["p1-briefing", "--tf", "12H"])
        assert result.exit_code == 0
        # Lesson #1 (12H + strategy-design) + #3 (strategy-design only) = 2 hits
        assert "2 hits" in result.output
        # Lesson #2 (4H, data-resolution) should not appear
        assert "4H TF Death" not in result.output

    def test_briefing_non_ohlcv_only(
        self,
        strategy_dir: Path,
        lesson_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OHLCV 데이터셋은 alternative에 포함되지 않는다."""
        monkeypatch.setattr(
            "src.cli.pipeline.StrategyStore", lambda: StrategyStore(base_dir=strategy_dir)
        )

        from src.pipeline.lesson_store import LessonStore

        monkeypatch.setattr(
            "src.pipeline.lesson_store.LessonStore",
            lambda base_dir=lesson_dir: LessonStore(base_dir=base_dir),
        )
        monkeypatch.chdir(tmp_path)

        from unittest.mock import MagicMock

        from src.catalog.models import DatasetEntry, DataType

        mock_datasets = [
            DatasetEntry(
                id="ohlcv_1m", name="OHLCV", data_type=DataType.OHLCV, source_id="binance_spot"
            ),
            DatasetEntry(
                id="dxy_daily", name="DXY", data_type=DataType.MACRO, source_id="yfinance"
            ),
        ]
        mock_catalog = MagicMock()
        mock_catalog.load_all.return_value = mock_datasets
        monkeypatch.setattr("src.catalog.store.DataCatalogStore", lambda **_kw: mock_catalog)

        result = runner.invoke(app, ["p1-briefing", "--tf", "1D"])
        assert result.exit_code == 0
        assert "1 datasets" in result.output
        assert "ohlcv_1m" not in result.output
        assert "dxy_daily" in result.output
