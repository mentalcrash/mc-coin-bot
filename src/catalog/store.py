"""Read-only YAML-based Data Catalog store.

catalogs/datasets.yaml에서 데이터셋 메타데이터를 로드:
- load(dataset_id): 단일 데이터셋 로드
- load_all(): 전체 데이터셋 로드
- get_source(source_id): 소스 메타데이터 로드
- get_by_type(DataType): 유형별 필터
- get_by_group(batch_group): 그룹별 필터
- 호환 API: get_batch_definitions, get_date_col, get_lag_days, build_precompute_map
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.catalog.models import (
    DataCatalog,
    DatasetEntry,
    DataType,
    EnrichmentScope,
    SourceMeta,
)

_DEFAULT_PATH = Path("catalogs/datasets.yaml")


class DataCatalogStore:
    """Read-only YAML 기반 데이터 카탈로그 저장소."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._catalog: DataCatalog | None = None
        self._datasets: dict[str, DatasetEntry] | None = None
        self._sources: dict[str, SourceMeta] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_loaded(self) -> DataCatalog:
        """Lazy load + cache."""
        if self._catalog is not None:
            return self._catalog

        if not self._path.exists():
            msg = f"Catalog YAML not found: {self._path}"
            raise FileNotFoundError(msg)

        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        self._catalog = DataCatalog(**raw)

        self._sources = {s.id: s for s in self._catalog.sources}
        self._datasets = {d.id: d for d in self._catalog.datasets}

        return self._catalog

    def load(self, dataset_id: str) -> DatasetEntry:
        """단일 데이터셋 로드."""
        self._ensure_loaded()
        assert self._datasets is not None
        if dataset_id not in self._datasets:
            msg = f"Dataset not found: {dataset_id}"
            raise KeyError(msg)
        return self._datasets[dataset_id]

    def load_all(self) -> list[DatasetEntry]:
        """전체 데이터셋 로드 (YAML 순서 유지)."""
        catalog = self._ensure_loaded()
        return list(catalog.datasets)

    def get_source(self, source_id: str) -> SourceMeta:
        """소스 메타데이터 로드."""
        self._ensure_loaded()
        assert self._sources is not None
        if source_id not in self._sources:
            msg = f"Source not found: {source_id}"
            raise KeyError(msg)
        return self._sources[source_id]

    def get_all_sources(self) -> list[SourceMeta]:
        """전체 소스 메타데이터 로드."""
        catalog = self._ensure_loaded()
        return list(catalog.sources)

    def get_by_type(self, data_type: DataType) -> list[DatasetEntry]:
        """데이터 유형별 필터."""
        catalog = self._ensure_loaded()
        return [d for d in catalog.datasets if d.data_type == data_type]

    def get_by_group(self, batch_group: str) -> list[DatasetEntry]:
        """배치 그룹별 필터."""
        catalog = self._ensure_loaded()
        return [d for d in catalog.datasets if d.batch_group == batch_group]

    # ─── 호환 API (Phase 2 연동용) ────────────────────────────────────

    def get_batch_definitions(self, group: str) -> list[tuple[str, str]]:
        """ONCHAIN_BATCH_DEFINITIONS 호환: (source_id, fetch_key) 리스트.

        Args:
            group: 배치 그룹 ("stablecoin", "tvl", "all" 등)

        Returns:
            (source_id, fetch_key) 튜플 리스트
        """
        catalog = self._ensure_loaded()
        if group == "all":
            return [
                (d.source_id, d.fetch_key or d.id)
                for d in catalog.datasets
                if d.data_type == DataType.ONCHAIN
            ]
        return [
            (d.source_id, d.fetch_key or d.id) for d in catalog.datasets if d.batch_group == group
        ]

    def get_date_col(self, source_id: str) -> str:
        """SOURCE_DATE_COLUMNS 호환: source의 date column 이름."""
        source = self.get_source(source_id)
        return source.date_column

    def get_lag_days(self, source_id: str, dataset_id: str | None = None) -> int:
        """Publication lag (일) — dataset 레벨 override 우선, fallback source 레벨.

        Args:
            source_id: 소스 ID
            dataset_id: 데이터셋 ID (optional, dataset-level lag 우선)

        Returns:
            Publication lag 일수
        """
        if dataset_id and self._datasets is not None:
            ds = self._datasets.get(dataset_id)
            if ds is not None and ds.lag_days is not None:
                return ds.lag_days
        elif dataset_id:
            self._ensure_loaded()
            assert self._datasets is not None
            ds = self._datasets.get(dataset_id)
            if ds is not None and ds.lag_days is not None:
                return ds.lag_days
        source = self.get_source(source_id)
        return source.lag_days

    def build_precompute_map(
        self, symbols: list[str]
    ) -> dict[str, list[tuple[str, str, list[str], dict[str, str]]]]:
        """_GLOBAL_SOURCES/_ASSET_SOURCES 호환: symbol→sources 매핑.

        Returns:
            {symbol: [(source_id, fetch_key, columns, rename_map), ...]}
        """
        catalog = self._ensure_loaded()

        # Enrichment 데이터셋 분류
        global_entries: list[DatasetEntry] = []
        asset_entries: dict[str, list[DatasetEntry]] = {}

        for ds in catalog.datasets:
            if ds.enrichment is None:
                continue
            if ds.enrichment.scope == EnrichmentScope.GLOBAL:
                global_entries.append(ds)
            elif ds.enrichment.scope == EnrichmentScope.ASSET:
                for asset in ds.enrichment.target_assets:
                    asset_entries.setdefault(asset, []).append(ds)

        result: dict[str, list[tuple[str, str, list[str], dict[str, str]]]] = {}
        for symbol in symbols:
            asset = symbol.split("/")[0].upper()
            sources: list[tuple[str, str, list[str], dict[str, str]]] = []

            for ds in global_entries:
                assert ds.enrichment is not None
                sources.append(
                    (
                        ds.source_id,
                        ds.fetch_key or ds.id,
                        list(ds.enrichment.columns),
                        dict(ds.enrichment.rename_map),
                    )
                )

            for ds in asset_entries.get(asset, []):
                assert ds.enrichment is not None
                sources.append(
                    (
                        ds.source_id,
                        ds.fetch_key or ds.id,
                        list(ds.enrichment.columns),
                        dict(ds.enrichment.rename_map),
                    )
                )

            result[symbol] = sources

        return result
