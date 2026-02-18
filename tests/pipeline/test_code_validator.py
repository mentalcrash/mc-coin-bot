"""Tests for src/pipeline/code_validator.py."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from src.pipeline.code_validator import Violation, validate_file, validate_source

if TYPE_CHECKING:
    from pathlib import Path


# ─── C1: Lookahead Bias ──────────────────────────────────────────


class TestLookaheadBias:
    def test_detect_shift_negative(self) -> None:
        source = "df.shift(-1)"
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C1"
        assert violations[0].severity == "error"
        assert "shift(-1)" in violations[0].message

    def test_allow_shift_positive(self) -> None:
        source = "df.shift(1)"
        violations = validate_source(source)
        assert not any(v.rule_id == "C1" for v in violations)

    def test_detect_chained_shift(self) -> None:
        source = 'df["close"].shift(-2)'
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C1"
        assert "shift(-2)" in violations[0].message

    def test_detect_shift_keyword(self) -> None:
        source = "df.shift(periods=-3)"
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C1"

    def test_cv_ignore_suppresses_c1(self) -> None:
        source = dedent("""\
            # cv-ignore: C1
            df.shift(-1)
        """)
        violations = validate_source(source)
        assert not any(v.rule_id == "C1" for v in violations)

    def test_cv_ignore_inline(self) -> None:
        source = "df.shift(-1)  # cv-ignore: C1"
        violations = validate_source(source)
        assert not any(v.rule_id == "C1" for v in violations)

    def test_shift_zero_allowed(self) -> None:
        source = "df.shift(0)"
        violations = validate_source(source)
        assert not any(v.rule_id == "C1" for v in violations)


# ─── C2: Row-wise Iteration ──────────────────────────────────────


class TestIterrows:
    def test_detect_iterrows(self) -> None:
        source = "for row in df.iterrows(): pass"
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C2"
        assert "iterrows" in violations[0].message

    def test_detect_itertuples(self) -> None:
        source = "for row in df.itertuples(): pass"
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C2"
        assert "itertuples" in violations[0].message

    def test_detect_apply_axis1(self) -> None:
        source = "df.apply(func, axis=1)"
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C2"
        assert "apply(axis=1)" in violations[0].message

    def test_allow_apply_axis0(self) -> None:
        source = "df.apply(func, axis=0)"
        violations = validate_source(source)
        assert not any(v.rule_id == "C2" for v in violations)

    def test_allow_apply_no_axis(self) -> None:
        source = "df.apply(func)"
        violations = validate_source(source)
        assert not any(v.rule_id == "C2" for v in violations)


# ─── C3: inplace=True ────────────────────────────────────────────


class TestInplace:
    def test_detect_inplace_true(self) -> None:
        source = "df.drop(columns=['a'], inplace=True)"
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C3"
        assert violations[0].severity == "warning"
        assert "inplace=True" in violations[0].message

    def test_allow_inplace_false(self) -> None:
        source = "df.drop(columns=['a'], inplace=False)"
        violations = validate_source(source)
        assert not any(v.rule_id == "C3" for v in violations)

    def test_severity_is_warning(self) -> None:
        source = "df.fillna(0, inplace=True)"
        violations = validate_source(source)
        assert violations[0].severity == "warning"


# ─── C4: Bare Except ─────────────────────────────────────────────


class TestBareExcept:
    def test_detect_bare_except(self) -> None:
        source = dedent("""\
            try:
                pass
            except:
                pass
        """)
        violations = validate_source(source)
        assert len(violations) == 1
        assert violations[0].rule_id == "C4"
        assert violations[0].severity == "warning"

    def test_allow_except_with_type(self) -> None:
        source = dedent("""\
            try:
                pass
            except Exception:
                pass
        """)
        violations = validate_source(source)
        assert not any(v.rule_id == "C4" for v in violations)

    def test_cv_ignore_suppresses_c4(self) -> None:
        source = dedent("""\
            try:
                pass
            except:  # cv-ignore: C4
                pass
        """)
        violations = validate_source(source)
        assert not any(v.rule_id == "C4" for v in violations)

    def test_allow_except_value_error(self) -> None:
        source = dedent("""\
            try:
                pass
            except ValueError:
                pass
        """)
        violations = validate_source(source)
        assert not any(v.rule_id == "C4" for v in violations)


# ─── validate_file ────────────────────────────────────────────────


class TestValidateFile:
    def test_validate_file_with_violations(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.py"
        src.write_text("df.shift(-1)\n", encoding="utf-8")
        violations = validate_file(src)
        assert len(violations) == 1
        assert violations[0].rule_id == "C1"

    def test_validate_file_clean(self, tmp_path: Path) -> None:
        src = tmp_path / "clean.py"
        src.write_text("x = 1\n", encoding="utf-8")
        violations = validate_file(src)
        assert violations == []

    def test_validate_file_not_found(self, tmp_path: Path) -> None:
        try:
            validate_file(tmp_path / "nonexistent.py")
            raise AssertionError("Should raise FileNotFoundError")
        except FileNotFoundError:
            pass


# ─── Multiple Violations ─────────────────────────────────────────


class TestMultipleViolations:
    def test_multiple_rules_detected(self) -> None:
        source = dedent("""\
            df.shift(-1)
            for row in df.iterrows():
                pass
            df.drop(columns=['a'], inplace=True)
            try:
                pass
            except:
                pass
        """)
        violations = validate_source(source)
        rule_ids = {v.rule_id for v in violations}
        assert rule_ids == {"C1", "C2", "C3", "C4"}

    def test_multiple_same_rule(self) -> None:
        source = dedent("""\
            df.shift(-1)
            df.shift(-2)
            df.shift(-3)
        """)
        violations = validate_source(source)
        assert len(violations) == 3
        assert all(v.rule_id == "C1" for v in violations)

    def test_cv_ignore_multiple_rules(self) -> None:
        source = dedent("""\
            # cv-ignore: C1, C3
            df.shift(-1)
        """)
        violations = validate_source(source)
        assert not any(v.rule_id == "C1" for v in violations)

    def test_violation_dataclass_fields(self) -> None:
        source = "df.shift(-1)"
        violations = validate_source(source, filename="test.py")
        v = violations[0]
        assert isinstance(v, Violation)
        assert v.rule_id == "C1"
        assert v.severity == "error"
        assert v.line == 1
        assert v.col >= 0
        assert isinstance(v.message, str)

    def test_syntax_error_returns_parse_violation(self) -> None:
        source = "def foo(:"
        violations = validate_source(source, filename="broken.py")
        assert len(violations) == 1
        assert violations[0].rule_id == "PARSE"
        assert violations[0].severity == "error"
