"""AST Code Validator -- 의미적 코드 검증.

stdlib ast.NodeVisitor를 사용하여 ruff/pyright가 탐지하지 못하는
의미적 오류를 검사합니다.

Rules Applied:
    - stdlib ast 모듈만 사용 (외부 의존성 없음)
    - ``# cv-ignore: C1`` 코멘트로 규칙별 비활성화

Rules:
    C1  df.shift(-N) where N > 0 (lookahead bias)       error
    C2  .iterrows() / .itertuples() / .apply(axis=1)    error
    C3  inplace=True keyword argument                    warning
    C4  bare except: without exception type              warning
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

# ─── Models ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Violation:
    """Single code violation detected by the AST validator.

    Attributes:
        rule_id: Rule identifier (C1, C2, C3, C4).
        severity: ``"error"`` or ``"warning"``.
        line: 1-based line number.
        col: 0-based column offset.
        message: Human-readable explanation.
    """

    rule_id: str
    severity: Literal["error", "warning"]
    line: int
    col: int
    message: str


# ─── Ignore Comment Parser ────────────────────────────────────────

_CV_IGNORE_RE = re.compile(r"#\s*cv-ignore:\s*([A-Z0-9,\s]+)", re.IGNORECASE)


def _parse_ignored_rules(comment: str) -> set[str]:
    """Extract rule IDs from a ``# cv-ignore: C1, C3`` comment."""
    match = _CV_IGNORE_RE.search(comment)
    if not match:
        return set()
    return {r.strip().upper() for r in match.group(1).split(",")}


# ─── AST Visitor ──────────────────────────────────────────────────


class CodeValidator(ast.NodeVisitor):
    """Walk AST nodes and collect :class:`Violation` instances.

    Args:
        source_lines: Source code split into lines (for ignore-comment lookup).
        filename: Filename used in diagnostic messages.
    """

    def __init__(self, source_lines: list[str], filename: str = "<string>") -> None:
        self._lines = source_lines
        self._filename = filename
        self.violations: list[Violation] = []

    # ── helpers ────────────────────────────────────────────────────

    def _is_ignored(self, line: int, rule_id: str) -> bool:
        """Return ``True`` if *rule_id* is suppressed at *line*.

        Checks both the line itself and the line immediately above.
        """
        rule_upper = rule_id.upper()
        for check_line in (line, line - 1):
            if 1 <= check_line <= len(self._lines):
                ignored = _parse_ignored_rules(self._lines[check_line - 1])
                if rule_upper in ignored:
                    return True
        return False

    def _add(
        self,
        rule_id: str,
        severity: Literal["error", "warning"],
        node: ast.AST,
        message: str,
    ) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        if not self._is_ignored(line, rule_id):
            self.violations.append(Violation(rule_id, severity, line, col, message))

    # ── C1: lookahead bias  df.shift(-N) ──────────────────────────

    def _check_shift_lookahead(self, node: ast.Call) -> None:
        """Detect ``df.shift(-N)`` where N > 0."""
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "shift":
            return

        self._check_shift_positional(node)
        self._check_shift_keyword(node)

    def _check_shift_positional(self, node: ast.Call) -> None:
        """Check positional arg: ``shift(-1)``, ``shift(-2)``."""
        if not node.args:
            return
        arg = node.args[0]
        if (
            isinstance(arg, ast.UnaryOp)
            and isinstance(arg.op, ast.USub)
            and isinstance(arg.operand, ast.Constant)
            and isinstance(arg.operand.value, (int, float))
            and arg.operand.value > 0
        ):
            self._add(
                "C1",
                "error",
                node,
                f"Lookahead bias: shift(-{arg.operand.value}) accesses future data",
            )

    def _check_shift_keyword(self, node: ast.Call) -> None:
        """Check keyword arg: ``shift(periods=-1)``."""
        for kw in node.keywords:
            if (
                kw.arg == "periods"
                and isinstance(kw.value, ast.UnaryOp)
                and isinstance(kw.value.op, ast.USub)
                and isinstance(kw.value.operand, ast.Constant)
                and isinstance(kw.value.operand.value, (int, float))
                and kw.value.operand.value > 0
            ):
                self._add(
                    "C1",
                    "error",
                    node,
                    f"Lookahead bias: shift(periods=-{kw.value.operand.value}) accesses future data",
                )

    # ── C2: row-wise iteration ────────────────────────────────────

    _ROW_ITER_METHODS = frozenset({"iterrows", "itertuples"})

    def _check_row_iteration(self, node: ast.Call) -> None:
        """Detect ``.iterrows()``, ``.itertuples()``, ``.apply(axis=1)``."""
        func = node.func
        if not isinstance(func, ast.Attribute):
            return

        if func.attr in self._ROW_ITER_METHODS:
            self._add(
                "C2",
                "error",
                node,
                f"Row-wise iteration: .{func.attr}() -- use vectorized operations",
            )
            return

        if func.attr == "apply":
            for kw in node.keywords:
                if kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                    self._add(
                        "C2",
                        "error",
                        node,
                        "Row-wise iteration: .apply(axis=1) -- use vectorized operations",
                    )

    # ── C3: inplace=True ──────────────────────────────────────────

    def _check_inplace(self, node: ast.Call) -> None:
        """Detect ``inplace=True`` keyword argument."""
        for kw in node.keywords:
            if (
                kw.arg == "inplace"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                self._add(
                    "C3",
                    "warning",
                    node,
                    "inplace=True -- use immutable operations instead",
                )

    # ── visitor entry points ──────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        """Check C1, C2, C3 on function calls."""
        self._check_shift_lookahead(node)
        self._check_row_iteration(node)
        self._check_inplace(node)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Check C4: bare ``except:`` without exception type."""
        if node.type is None:
            self._add(
                "C4",
                "warning",
                node,
                "Bare except: -- specify an exception type",
            )
        self.generic_visit(node)


# ─── Public API ───────────────────────────────────────────────────


def validate_source(source: str, filename: str = "<string>") -> list[Violation]:
    """Parse *source* and return all detected :class:`Violation` instances.

    Args:
        source: Python source code string.
        filename: Used for error context (not checked on disk).

    Returns:
        List of violations found (empty if clean).
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return [
            Violation(
                rule_id="PARSE",
                severity="error",
                line=0,
                col=0,
                message=f"Failed to parse {filename}",
            )
        ]

    lines = source.splitlines()
    visitor = CodeValidator(lines, filename)
    visitor.visit(tree)
    return visitor.violations


def validate_file(path: Path) -> list[Violation]:
    """Read a file from *path* and validate its contents.

    Args:
        path: Path to a Python source file.

    Returns:
        List of violations found (empty if clean).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    source = path.read_text(encoding="utf-8")
    return validate_source(source, filename=str(path))
