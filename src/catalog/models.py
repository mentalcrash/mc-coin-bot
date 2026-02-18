"""Data Catalog Pydantic models.

데이터셋 메타데이터를 구조화하는 모델:
- SourceMeta: 데이터 소스 정보 (API URL, date column, lag)
- EnrichmentConfig: OHLCV 병합 설정 (scope, columns, rename)
- DatasetEntry: 개별 데이터셋 메타데이터
- DataCatalog: 전체 카탈로그 (sources + datasets)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ─── Enums ───────────────────────────────────────────────────────────


class DataType(StrEnum):
    """데이터 유형."""

    OHLCV = "ohlcv"
    DERIVATIVES = "derivatives"
    ONCHAIN = "onchain"
    MACRO = "macro"
    OPTIONS = "options"
    DERIV_EXT = "deriv_ext"


class EnrichmentScope(StrEnum):
    """Enrichment 적용 범위."""

    GLOBAL = "global"
    ASSET = "asset"
    PER_ASSET = "per_asset"


# ─── Models ──────────────────────────────────────────────────────────


class SourceMeta(BaseModel):
    """데이터 소스 메타데이터."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Source 식별자 (e.g., defillama)")
    name: str = Field(description="표시 이름")
    api_url: str = Field(default="", description="API base URL")
    date_column: str = Field(default="date", description="날짜 컬럼명")
    lag_days: int = Field(default=0, description="Publication lag (일)")
    rate_limit_per_min: int = Field(default=0, description="분당 요청 제한")


class EnrichmentConfig(BaseModel):
    """OHLCV Enrichment 설정."""

    model_config = ConfigDict(frozen=True)

    scope: EnrichmentScope = Field(description="global (모든 심볼) 또는 asset (특정 심볼)")
    target_assets: list[str] = Field(
        default_factory=list, description="대상 에셋 (asset scope일 때)"
    )
    columns: list[str] = Field(default_factory=list, description="병합할 컬럼")
    rename_map: dict[str, str] = Field(default_factory=dict, description="컬럼 rename 매핑")


class DatasetEntry(BaseModel):
    """개별 데이터셋 메타데이터."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Dataset 식별자 (e.g., btc_metrics)")
    name: str = Field(description="표시 이름")
    data_type: DataType = Field(description="데이터 유형")
    source_id: str = Field(description="Source 참조 키")
    batch_group: str = Field(default="", description="배치 그룹 (e.g., stablecoin, tvl)")
    description: str = Field(default="", description="설명")
    resolution: str = Field(default="1d", description="데이터 해상도 (1m, 1h, 1d)")
    available_since: str = Field(default="", description="데이터 시작일")
    storage_path: str = Field(default="", description="Silver 저장 경로 패턴")
    columns: list[str] = Field(default_factory=list, description="주요 컬럼")
    strategy_hints: list[str] = Field(default_factory=list, description="전략 활용 힌트")
    fetch_key: str = Field(default="", description="route_fetch()에 전달할 name")
    enrichment: EnrichmentConfig | None = Field(default=None, description="OHLCV Enrichment 설정")
    silver_resample: str = Field(default="", description="Silver 리샘플 주기")
    history_limit_days: int = Field(default=0, description="API 히스토리 제한 (일)")
    lag_days: int | None = Field(default=None, description="Dataset-level publication lag override (일)")


class DataCatalog(BaseModel):
    """전체 데이터 카탈로그."""

    model_config = ConfigDict(frozen=True)

    sources: list[SourceMeta] = Field(default_factory=list)
    datasets: list[DatasetEntry] = Field(default_factory=list)
