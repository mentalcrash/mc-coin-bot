"""Validation level definitions.

검증 레벨을 정의합니다. 개발 단계에 따라 적절한 검증 수준을 선택합니다.

Rules Applied:
    - #10 Python Standards: StrEnum for string-based enums
"""

from enum import StrEnum


class ValidationLevel(StrEnum):
    """검증 레벨.

    개발 단계에 따라 적절한 검증 수준을 제공합니다.

    Attributes:
        QUICK: 매 iteration용 - IS/OOS Split (70/30), ~2x cost
        MILESTONE: 중요 변경 시 - Walk-Forward (5 fold), ~5x cost
        FINAL: 전략 완성 시 - CPCV + Monte Carlo, ~20x cost

    Example:
        >>> level = ValidationLevel.QUICK
        >>> print(level.value)
        'quick'
    """

    QUICK = "quick"
    MILESTONE = "milestone"
    FINAL = "final"

    @property
    def description(self) -> str:
        """레벨별 설명 반환."""
        descriptions = {
            ValidationLevel.QUICK: "IS/OOS Split (70/30) - 매 iteration용",
            ValidationLevel.MILESTONE: "Walk-Forward (5 fold) - 중요 변경 시",
            ValidationLevel.FINAL: "CPCV + Monte Carlo - 전략 완성 시",
        }
        return descriptions[self]

    @property
    def estimated_cost_multiplier(self) -> int:
        """예상 계산 비용 배수."""
        costs = {
            ValidationLevel.QUICK: 2,
            ValidationLevel.MILESTONE: 5,
            ValidationLevel.FINAL: 20,
        }
        return costs[self]
