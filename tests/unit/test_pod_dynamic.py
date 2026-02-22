"""Tests for Pod dynamic asset management (surveillance integration)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.orchestrator.asset_allocator import AssetAllocationConfig, IntraPodAllocator
from src.orchestrator.asset_selector import AssetSelector
from src.orchestrator.config import AssetSelectorConfig, PodConfig
from src.orchestrator.models import AssetLifecycleState
from src.orchestrator.pod import StrategyPod


def _make_pod(
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT"),
    max_assets: int | None = 5,
    with_selector: bool = False,
    with_allocator: bool = False,
) -> StrategyPod:
    """Helper: StrategyPod with mock strategy."""
    strategy = MagicMock()
    strategy.name = "test"
    strategy.config = None

    selector_cfg = None
    if with_selector:
        selector_cfg = AssetSelectorConfig(enabled=True, min_active_assets=1)

    alloc_cfg = None
    if with_allocator:
        alloc_cfg = AssetAllocationConfig()

    config = PodConfig(
        pod_id="test-pod",
        strategy_name="test",
        symbols=symbols,
        max_assets=max_assets,
        asset_selector=selector_cfg,
        asset_allocation=alloc_cfg,
    )
    return StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)


class TestAddAsset:
    """Pod.add_asset() 테스트."""

    def test_add_asset_success(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",), max_assets=3)
        assert pod.add_asset("ETH/USDT") is True
        assert "ETH/USDT" in pod.symbols
        assert pod.accepts_symbol("ETH/USDT") is True
        assert len(pod.symbols) == 2

    def test_add_asset_duplicate(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        assert pod.add_asset("BTC/USDT") is False

    def test_add_asset_max_capacity(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT", "ETH/USDT"), max_assets=2)
        assert pod.add_asset("SOL/USDT") is False
        assert "SOL/USDT" not in pod.symbols

    def test_add_asset_no_limit(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",), max_assets=None)
        assert pod.add_asset("ETH/USDT") is True
        assert pod.add_asset("SOL/USDT") is True
        assert pod.add_asset("XRP/USDT") is True
        assert len(pod.symbols) == 4

    def test_add_asset_with_selector(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",), with_selector=True)
        pod.add_asset("ETH/USDT")
        assert pod._asset_selector is not None
        states = pod._asset_selector.asset_states
        assert "ETH/USDT" in states
        assert states["ETH/USDT"] == AssetLifecycleState.ACTIVE

    def test_add_asset_with_allocator(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",), with_allocator=True)
        pod.add_asset("ETH/USDT")
        assert pod._asset_allocator is not None
        weights = pod._asset_allocator.weights
        assert "ETH/USDT" in weights
        # EW: each should be 0.5
        assert abs(weights["BTC/USDT"] - 0.5) < 1e-6
        assert abs(weights["ETH/USDT"] - 0.5) < 1e-6

    def test_add_asset_initializes_returns(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        pod.add_asset("ETH/USDT")
        assert "ETH/USDT" in pod._asset_returns
        assert pod._asset_returns["ETH/USDT"] == []


class TestCleanupExcludedAsset:
    """Pod.cleanup_excluded_asset() 테스트."""

    def test_cleanup_keeps_returns(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT", "ETH/USDT"))
        # Simulate some buffer/return data
        pod._buffers["ETH/USDT"] = [{"close": 100.0}]
        pod._timestamps["ETH/USDT"] = []
        pod._target_weights["ETH/USDT"] = 0.5
        pod._asset_returns["ETH/USDT"] = [0.01, 0.02]

        pod.cleanup_excluded_asset("ETH/USDT")

        assert "ETH/USDT" not in pod._buffers
        assert "ETH/USDT" not in pod._timestamps
        assert "ETH/USDT" not in pod._target_weights
        # Returns preserved
        assert "ETH/USDT" in pod._asset_returns
        assert pod._asset_returns["ETH/USDT"] == [0.01, 0.02]

    def test_cleanup_nonexistent_symbol(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        # Should not raise
        pod.cleanup_excluded_asset("DOGE/USDT")


class TestRuntimeSymbols:
    """_runtime_symbols 관련 테스트."""

    def test_config_symbols_preserved(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT", "ETH/USDT"))
        pod.add_asset("SOL/USDT")
        assert pod.config_symbols == ("BTC/USDT", "ETH/USDT")
        assert "SOL/USDT" in pod.symbols

    def test_accepts_runtime_symbol(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        assert pod.accepts_symbol("ETH/USDT") is False
        pod.add_asset("ETH/USDT")
        assert pod.accepts_symbol("ETH/USDT") is True

    def test_serialization_runtime_symbols(self) -> None:
        pod = _make_pod(symbols=("BTC/USDT",))
        pod.add_asset("ETH/USDT")
        pod.add_asset("SOL/USDT")

        data = pod.to_dict()
        assert "runtime_symbols" in data
        assert sorted(data["runtime_symbols"]) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        # Restore
        pod2 = _make_pod(symbols=("BTC/USDT",))
        pod2.restore_from_dict(data)
        assert pod2.accepts_symbol("ETH/USDT") is True
        assert pod2.accepts_symbol("SOL/USDT") is True

    def test_apply_asset_weight_uses_runtime(self) -> None:
        """1/N should use runtime symbols count, not config."""
        pod = _make_pod(symbols=("BTC/USDT",))
        pod.add_asset("ETH/USDT")
        pod.add_asset("SOL/USDT")
        # 3 runtime symbols → 1/3 weight
        weight = pod._apply_asset_weight("BTC/USDT", 1.0)
        assert abs(weight - 1.0 / 3) < 1e-6


class TestAssetSelectorExtensions:
    """AssetSelector add_symbol / flag_permanently_excluded."""

    def test_add_symbol(self) -> None:
        cfg = AssetSelectorConfig(enabled=True, min_active_assets=1)
        selector = AssetSelector(config=cfg, symbols=("BTC/USDT",))
        selector.add_symbol("ETH/USDT")
        assert "ETH/USDT" in selector.multipliers
        assert selector.multipliers["ETH/USDT"] == 1.0
        assert selector.asset_states["ETH/USDT"] == AssetLifecycleState.ACTIVE

    def test_add_symbol_duplicate(self) -> None:
        cfg = AssetSelectorConfig(enabled=True, min_active_assets=1)
        selector = AssetSelector(config=cfg, symbols=("BTC/USDT",))
        selector.add_symbol("BTC/USDT")  # should be no-op
        assert len(selector.multipliers) == 1

    def test_flag_permanently_excluded(self) -> None:
        cfg = AssetSelectorConfig(enabled=True, min_active_assets=1)
        selector = AssetSelector(config=cfg, symbols=("BTC/USDT", "ETH/USDT"))
        selector.flag_permanently_excluded("ETH/USDT")
        state = selector.asset_states["ETH/USDT"]
        assert state == AssetLifecycleState.COOLDOWN
        assert selector.multipliers["ETH/USDT"] == 0.0

    def test_flag_already_excluded(self) -> None:
        cfg = AssetSelectorConfig(enabled=True, min_active_assets=1)
        selector = AssetSelector(config=cfg, symbols=("BTC/USDT", "ETH/USDT"))
        selector.flag_permanently_excluded("ETH/USDT")
        selector.flag_permanently_excluded("ETH/USDT")  # no-op


class TestIntraPodAllocatorExtensions:
    """IntraPodAllocator add_symbol."""

    def test_add_symbol_ew_rebalance(self) -> None:
        cfg = AssetAllocationConfig()
        allocator = IntraPodAllocator(config=cfg, symbols=("BTC/USDT",))
        assert abs(allocator.weights["BTC/USDT"] - 1.0) < 1e-6

        allocator.add_symbol("ETH/USDT")
        assert len(allocator.weights) == 2
        assert abs(allocator.weights["BTC/USDT"] - 0.5) < 1e-6
        assert abs(allocator.weights["ETH/USDT"] - 0.5) < 1e-6

    def test_add_symbol_duplicate(self) -> None:
        cfg = AssetAllocationConfig()
        allocator = IntraPodAllocator(config=cfg, symbols=("BTC/USDT",))
        allocator.add_symbol("BTC/USDT")  # no-op
        assert len(allocator.weights) == 1
