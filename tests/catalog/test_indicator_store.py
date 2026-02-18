"""Tests for src/catalog/indicator_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.catalog.indicator_models import IndicatorCategory
from src.catalog.indicator_store import IndicatorCatalogStore

_SAMPLE_YAML = {
    "indicators": [
        {
            "id": "rsi",
            "name": "RSI",
            "module": "oscillators",
            "category": "oscillator",
            "description": "상대강도지수",
            "default_params": {"period": 14},
            "used_by": ["ctrend"],
            "alpha_potential": "medium",
        },
        {
            "id": "hurst_exponent",
            "name": "Hurst Exponent",
            "module": "composite",
            "category": "composite",
            "description": "시계열 지속성 판별",
            "default_params": {"lags": 50},
            "used_by": [],
            "alpha_potential": "high",
            "notes": "미사용 고가치 지표",
        },
        {
            "id": "atr",
            "name": "ATR",
            "module": "trend",
            "category": "trend",
            "description": "Average True Range",
            "default_params": {"period": 14},
            "used_by": ["ctrend", "anchor-mom"],
            "alpha_potential": "low",
        },
    ]
}


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    path = tmp_path / "indicators.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def store(store_path: Path) -> IndicatorCatalogStore:
    return IndicatorCatalogStore(path=store_path)


class TestIndicatorCatalogStore:
    def test_load_all(self, store: IndicatorCatalogStore) -> None:
        indicators = store.load_all()
        assert len(indicators) == 3

    def test_load_single(self, store: IndicatorCatalogStore) -> None:
        ind = store.load("rsi")
        assert ind.name == "RSI"
        assert ind.category == IndicatorCategory.OSCILLATOR
        assert "ctrend" in ind.used_by

    def test_load_not_found(self, store: IndicatorCatalogStore) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            store.load("nonexistent")

    def test_file_not_found(self, tmp_path: Path) -> None:
        s = IndicatorCatalogStore(path=tmp_path / "missing.yaml")
        with pytest.raises(FileNotFoundError):
            s.load_all()

    def test_get_by_category(self, store: IndicatorCatalogStore) -> None:
        oscillators = store.get_by_category("oscillator")
        assert len(oscillators) == 1
        assert oscillators[0].id == "rsi"

    def test_get_unused(self, store: IndicatorCatalogStore) -> None:
        unused = store.get_unused()
        assert len(unused) == 1
        assert unused[0].id == "hurst_exponent"

    def test_get_by_potential(self, store: IndicatorCatalogStore) -> None:
        high = store.get_by_potential("high")
        assert len(high) == 1
        assert high[0].id == "hurst_exponent"

    def test_get_by_strategy(self, store: IndicatorCatalogStore) -> None:
        ctrend_inds = store.get_by_strategy("ctrend")
        assert len(ctrend_inds) == 2
        ids = [i.id for i in ctrend_inds]
        assert "rsi" in ids
        assert "atr" in ids

    def test_get_by_strategy_anchor_mom(self, store: IndicatorCatalogStore) -> None:
        am_inds = store.get_by_strategy("anchor-mom")
        assert len(am_inds) == 1
        assert am_inds[0].id == "atr"

    def test_cache_hit(self, store: IndicatorCatalogStore) -> None:
        store.load_all()
        assert store._catalog is not None
        indicators = store.load_all()
        assert len(indicators) == 3


class TestRealIndicatorCatalog:
    """실제 catalogs/indicators.yaml 로드 테스트."""

    def test_load_real_yaml(self) -> None:
        real_path = Path("catalogs/indicators.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/indicators.yaml not found")
        store = IndicatorCatalogStore(path=real_path)
        indicators = store.load_all()
        assert len(indicators) == 50

    def test_all_categories_present(self) -> None:
        real_path = Path("catalogs/indicators.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/indicators.yaml not found")
        store = IndicatorCatalogStore(path=real_path)
        cats = {i.category for i in store.load_all()}
        assert IndicatorCategory.TREND in cats
        assert IndicatorCategory.OSCILLATOR in cats
        assert IndicatorCategory.VOLATILITY in cats
        assert IndicatorCategory.COMPOSITE in cats

    def test_unused_indicators_exist(self) -> None:
        real_path = Path("catalogs/indicators.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/indicators.yaml not found")
        store = IndicatorCatalogStore(path=real_path)
        unused = store.get_unused()
        assert len(unused) >= 10
        ids = [i.id for i in unused]
        assert "hurst_exponent" in ids

    def test_high_potential_indicators(self) -> None:
        real_path = Path("catalogs/indicators.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/indicators.yaml not found")
        store = IndicatorCatalogStore(path=real_path)
        high = store.get_by_potential("high")
        assert len(high) >= 5
        ids = [i.id for i in high]
        assert "hurst_exponent" in ids
        assert "fractal_dimension" in ids

    def test_indicators_match_module_all(self) -> None:
        """catalogs/indicators.yaml IDs가 indicators.__all__과 일치."""
        real_path = Path("catalogs/indicators.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/indicators.yaml not found")

        from src.market.indicators import __all__ as module_all

        store = IndicatorCatalogStore(path=real_path)
        catalog_ids = {i.id for i in store.load_all()}
        module_set = set(module_all)

        # YAML에 있지만 모듈에 없는 지표
        yaml_only = catalog_ids - module_set
        # 모듈에 있지만 YAML에 없는 지표
        module_only = module_set - catalog_ids

        assert not yaml_only, f"YAML only: {yaml_only}"
        assert not module_only, f"Module only: {module_only}"

    def test_ctrend_strategy_indicators(self) -> None:
        real_path = Path("catalogs/indicators.yaml")
        if not real_path.exists():
            pytest.skip("catalogs/indicators.yaml not found")
        store = IndicatorCatalogStore(path=real_path)
        ctrend = store.get_by_strategy("ctrend")
        assert len(ctrend) >= 10
