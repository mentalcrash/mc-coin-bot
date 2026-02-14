"""Tests for Pod pause/resume functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.orchestrator.models import LifecycleState
from src.orchestrator.pod import StrategyPod


def _make_pod(pod_id: str = "test-pod") -> StrategyPod:
    """테스트용 StrategyPod 생성."""
    config = MagicMock()
    config.pod_id = pod_id
    config.symbols = ("BTC/USDT",)

    strategy = MagicMock()
    strategy.config = None

    return StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)


class TestPodPause:
    """Pause/Resume 기본 동작 테스트."""

    def test_initial_not_paused(self) -> None:
        pod = _make_pod()
        assert pod.paused is False
        assert pod.is_active is True

    def test_pause_makes_inactive(self) -> None:
        pod = _make_pod()
        pod.pause()
        assert pod.paused is True
        assert pod.is_active is False

    def test_resume_makes_active(self) -> None:
        pod = _make_pod()
        pod.pause()
        assert pod.is_active is False
        pod.resume()
        assert pod.paused is False
        assert pod.is_active is True

    def test_retired_and_paused(self) -> None:
        """RETIRED 상태이면 paused=False여도 is_active=False."""
        pod = _make_pod()
        pod.state = LifecycleState.RETIRED
        assert pod.is_active is False

    def test_retired_overrides_resume(self) -> None:
        """RETIRED + resume → 여전히 is_active=False."""
        pod = _make_pod()
        pod.pause()
        pod.state = LifecycleState.RETIRED
        pod.resume()
        assert pod.is_active is False


class TestPodPauseSerialization:
    """Pause 상태 직렬화 테스트."""

    def test_to_dict_includes_paused(self) -> None:
        pod = _make_pod()
        pod.pause()
        data = pod.to_dict()
        assert data["paused"] is True

    def test_roundtrip(self) -> None:
        """to_dict → restore_from_dict 왕복."""
        pod = _make_pod()
        pod.pause()
        data = pod.to_dict()

        pod2 = _make_pod()
        pod2.restore_from_dict(data)  # type: ignore[arg-type]
        assert pod2.paused is True
        assert pod2.is_active is False

    def test_restore_unpaused(self) -> None:
        pod = _make_pod()
        data = pod.to_dict()

        pod2 = _make_pod()
        pod2.pause()  # paused initially
        pod2.restore_from_dict(data)  # type: ignore[arg-type]
        assert pod2.paused is False
        assert pod2.is_active is True

    def test_restore_missing_paused_field(self) -> None:
        """paused 필드 없으면 기본값 유지."""
        pod = _make_pod()
        pod.restore_from_dict({"state": "incubation"})  # type: ignore[arg-type]
        # paused not touched → stays False (default)
        assert pod.paused is False
