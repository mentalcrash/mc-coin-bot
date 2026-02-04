"""Strategy Registry for dynamic strategy loading.

이 모듈은 전략을 이름으로 등록하고 조회할 수 있는 Registry를 제공합니다.
CLI와 전략 간의 결합도를 제거하여 OCP(Open-Closed Principle)를 준수합니다.

Rules Applied:
    - #02 Clean Code: Dependency Inversion via Registry
    - #10 Python Standards: Modern typing, decorators
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from src.strategy.base import BaseStrategy

T = TypeVar("T", bound="BaseStrategy")

# 전략 레지스트리 (모듈 레벨 싱글톤)
_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(name: str):
    """전략 등록 데코레이터.

    클래스에 이 데코레이터를 적용하면 전략이 Registry에 등록됩니다.
    등록된 전략은 get_strategy()로 조회할 수 있습니다.

    Args:
        name: 전략 식별자 (예: "tsmom", "adaptive-breakout")

    Returns:
        데코레이터 함수

    Example:
        >>> @register("my-strategy")
        ... class MyStrategy(BaseStrategy):
        ...     ...
        >>>
        >>> strategy_class = get_strategy("my-strategy")
        >>> strategy = strategy_class()
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _STRATEGY_REGISTRY:
            existing = _STRATEGY_REGISTRY[name].__name__
            msg = f"Strategy '{name}' is already registered by {existing}"
            raise ValueError(msg)
        _STRATEGY_REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def get_strategy(name: str) -> type[BaseStrategy]:
    """이름으로 전략 클래스를 반환합니다.

    Args:
        name: 전략 식별자

    Returns:
        등록된 전략 클래스

    Raises:
        KeyError: 전략이 등록되지 않은 경우
    """
    if name not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        msg = f"Strategy '{name}' not found. Available: [{available}]"
        raise KeyError(msg)
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """등록된 모든 전략 이름을 반환합니다.

    Returns:
        전략 이름 목록 (알파벳 순)
    """
    return sorted(_STRATEGY_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """전략이 등록되어 있는지 확인합니다.

    Args:
        name: 전략 식별자

    Returns:
        등록 여부
    """
    return name in _STRATEGY_REGISTRY


def clear_registry() -> None:
    """레지스트리를 초기화합니다.

    Warning:
        테스트 목적으로만 사용하세요. 프로덕션에서는 호출하지 마세요.
    """
    _STRATEGY_REGISTRY.clear()
