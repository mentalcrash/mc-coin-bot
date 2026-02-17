"""On-chain data models (DeFiLlama stablecoins/TVL/DEX, Coin Metrics).

Pydantic V2 frozen models with Decimal for financial values.
Timestamps are Unix seconds (DeFiLlama convention, unlike Binance ms).

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Decimal for financial values
    - #10 Python Standards: Modern typing (X | None)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StablecoinSupplyRecord(BaseModel):
    """Stablecoin 전체 공급량 레코드 (DeFiLlama /stablecoincharts/all).

    Attributes:
        date: 날짜 (UTC)
        total_circulating_usd: 전체 유통량 (USD)
        source: 데이터 소스 (예: "defillama")
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    total_circulating_usd: Decimal = Field(..., ge=0)
    source: str = "defillama"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class StablecoinChainRecord(BaseModel):
    """체인별 Stablecoin 공급량 레코드 (DeFiLlama /stablecoincharts/{chain}).

    Attributes:
        date: 날짜 (UTC)
        chain: 체인 이름 (예: "Ethereum")
        total_circulating_usd: 체인 전체 유통량 (USD)
        total_minted_usd: 체인 전체 발행량 (USD, 선택)
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    chain: str
    total_circulating_usd: Decimal = Field(..., ge=0)
    total_minted_usd: Decimal | None = Field(default=None, ge=0)

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class StablecoinIndividualRecord(BaseModel):
    """개별 Stablecoin 유통량 레코드 (DeFiLlama /stablecoin/{id}).

    Attributes:
        date: 날짜 (UTC)
        stablecoin_id: DeFiLlama 스테이블코인 ID
        name: 스테이블코인 이름 (예: "USDT")
        circulating_usd: 유통량 (USD)
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    stablecoin_id: int
    name: str
    circulating_usd: Decimal = Field(..., ge=0)

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class TvlRecord(BaseModel):
    """체인별 TVL 레코드 (DeFiLlama /v2/historicalChainTvl).

    Attributes:
        date: 날짜 (UTC)
        chain: 체인 이름 ("all" = 전체 합산)
        tvl_usd: TVL (USD)
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    chain: str  # "all" for total, "Ethereum", "Tron" etc.
    tvl_usd: Decimal = Field(..., ge=0)

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class DexVolumeRecord(BaseModel):
    """DEX 거래량 레코드 (DeFiLlama /overview/dexs).

    Attributes:
        date: 날짜 (UTC)
        volume_usd: 일일 DEX 거래량 (USD)
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    date: datetime
    volume_usd: Decimal = Field(..., ge=0)
    source: str = "defillama"

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class CoinMetricsRecord(BaseModel):
    """Coin Metrics daily asset metric record.

    Note: fetcher는 wide format DataFrame을 직접 반환하며,
    이 모델은 long format 참조용입니다 (OnchainBatch.records Union 호환).

    Attributes:
        time: 날짜 (UTC)
        asset: 자산 심볼 (예: "btc", "eth")
        metric_name: 메트릭 이름 (예: "MVRV", "RealCap")
        value: 메트릭 값 (Decimal)
    """

    model_config = ConfigDict(frozen=True)

    time: datetime
    asset: str
    metric_name: str
    value: Decimal

    @field_validator("time", mode="before")
    @classmethod
    def parse_time(cls, v: str | int | float | datetime) -> datetime:
        """ISO 8601 / Unix seconds → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            # Coin Metrics: "2024-01-01T00:00:00.000000000Z"
            cleaned = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class FearGreedRecord(BaseModel):
    """Fear & Greed Index 레코드 (Alternative.me).

    Attributes:
        timestamp: 날짜 (UTC)
        value: 지수 값 (0=Extreme Fear ~ 100=Extreme Greed)
        classification: 분류 텍스트 (예: "Extreme Fear", "Greed")
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    value: int = Field(..., ge=0, le=100)
    classification: str
    source: str = "alternative_me"

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class BlockchainChartRecord(BaseModel):
    """Blockchain.com Chart 레코드 (BTC network health).

    Attributes:
        timestamp: 날짜 (UTC)
        chart_name: 차트 이름 (예: "hash-rate")
        value: 차트 값 (Decimal)
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    chart_name: str
    value: Decimal
    source: str = "blockchain_com"

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class EthSupplyRecord(BaseModel):
    """Etherscan ETH Supply 스냅샷 레코드 (ethsupply2 API).

    Attributes:
        timestamp: 수집 시각 (UTC)
        eth_supply: ETH 전체 공급량 (ETH 단위)
        eth2_staking: ETH2 스테이킹 잔액 (ETH 단위)
        burnt_fees: EIP-1559 소각량 (ETH 단위)
        withdrawn_total: 출금 총량 (ETH 단위)
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    eth_supply: Decimal = Field(..., ge=0)
    eth2_staking: Decimal = Field(..., ge=0)
    burnt_fees: Decimal = Field(..., ge=0)
    withdrawn_total: Decimal = Field(..., ge=0)
    source: str = "etherscan"

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class MempoolMiningRecord(BaseModel):
    """mempool.space BTC 마이닝 레코드 (hashrate + difficulty).

    Attributes:
        timestamp: 날짜 (UTC)
        avg_hashrate: 평균 해시레이트 (H/s)
        difficulty: 채굴 난이도
        block_height: 블록 높이 (difficulty 데이터에서만, 선택)
        adjustment: 난이도 조정 비율 (선택)
        source: 데이터 소스
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    avg_hashrate: Decimal = Field(..., ge=0)
    difficulty: Decimal = Field(..., ge=0)
    block_height: int | None = None
    adjustment: Decimal | None = None
    source: str = "mempool_space"

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | int | float | datetime) -> datetime:
        """Unix seconds / ISO string → UTC datetime."""
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class OnchainBatch(BaseModel):
    """On-chain 데이터 배치 (수집 결과 컨테이너).

    Attributes:
        source: 데이터 소스 (예: "defillama")
        data_type: 데이터 유형 (예: "stablecoin_total")
        records: 레코드 튜플
        fetched_at: 수집 시각 (UTC)
    """

    model_config = ConfigDict(frozen=True)

    source: str
    data_type: str
    records: tuple[
        StablecoinSupplyRecord
        | StablecoinChainRecord
        | StablecoinIndividualRecord
        | CoinMetricsRecord
        | TvlRecord
        | DexVolumeRecord
        | FearGreedRecord
        | BlockchainChartRecord
        | EthSupplyRecord
        | MempoolMiningRecord,
        ...,
    ]
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("fetched_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """fetched_at에 UTC timezone 적용."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @property
    def count(self) -> int:
        """레코드 수."""
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        """레코드가 비어있는지 확인."""
        return len(self.records) == 0
