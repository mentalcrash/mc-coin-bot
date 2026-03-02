# Dynamic Asset Surveillance Backtest System

**목적**: 라이브 Surveillance의 동적 에셋 선택/교체를 백테스트에서 재현하여,
고정 심볼 백테스트의 survivorship bias를 제거하고 라이브-백테스트 parity를 확보.

**갱신일**: 2026-03-01

---

## 1. 문제 정의

### 현재 상태

```
라이브 (설계 의도):
  MarketSurveillanceService → 7일 주기 스캔 → Pod 동적 에셋 교체
  → pinned_symbols: false → 유동성 상위 에셋 자동 할당

백테스트 (현재):
  OrchestratedRunner → 고정 심볼 → 전 기간 동일 에셋
  → Surveillance 시뮬레이션 없음
```

### 문제점

| 문제 | 설명 | 영향 |
|------|------|------|
| **Survivorship Bias** | 백테스트 심볼이 "현재 기준 상위"로 선택됨 | Sharpe 과대 추정 |
| **유니버스 불일치** | 라이브는 동적 교체, 백테스트는 고정 | 성과 비교 불가 |
| **검증 공백** | 동적 에셋 교체 전략의 효과를 사전 검증할 수 없음 | 라이브 배포 리스크 |
| **기회 비용** | 고정 심볼 → 유동성 로테이션 포착 불가 | alpha 손실 |

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────┐
│  Wide Universe Pool (1m Parquet, ~40 에셋)           │
│  BTC, ETH, SOL, DOGE, BNB, XRP, ADA, AVAX, ...     │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────▼───────────────┐
         │  BacktestSurveillance     │
         │  Simulator                │
         │                           │
         │  매 7일 (10,080 × 1m):    │
         │  1. rolling 7D volume 계산│
         │  2. 상위 N개 선정          │
         │  3. ScanResult 생성       │
         └───────────┬───────────────┘
                     │
         ┌───────────▼───────────────┐
         │  OrchestratedRunner       │
         │  (기존 경로 재사용)        │
         │                           │
         │  on_universe_update()     │
         │  → Pod.add_asset()        │
         │  → Pod.drop_asset()       │
         │  → 라우팅 테이블 갱신      │
         └───────────────────────────┘
```

---

## 3. 데이터 전략

### 3.1 필요 데이터: Wide Universe Pool

**"모든 에셋" 이 아닌 "후보군 풀" 1m 데이터.**

> Surveillance 필터(24h volume $50M+)를 통과할 가능성이 있었던 에셋만 수집.
> 2020년부터 현재까지 한 번이라도 24h 거래대금 상위 50위 안에 들었던 에셋이 대상.

```
Tier A (이미 보유): BTC, ETH, SOL, DOGE, BNB          → 5개
Tier B (상위 상주): XRP, ADA, AVAX, LINK, DOT, MATIC,
                   LTC, UNI, ATOM, NEAR, FIL, APT,
                   ARB, OP, SUI, SEI, TIA              → ~15개
Tier C (과거 상위): EOS, XLM, TRX, ALGO, FTM, SAND,
                   MANA, GALA, AXS, THETA, ICP, HBAR  → ~12개
Tier D (최근 부상): WIF, PEPE, BONK, JUP, RENDER,
                   INJ, STX, TON, WLD, ORDI            → ~10개
─────────────────────────────────────────────────
총 후보군: ~40~45개
```

### 3.2 데이터 수집 방법

기존 `mcbot ingest` 파이프라인으로 수집 가능:

```bash
# 에셋당 약 2~4GB (2020~2026, 1m Parquet)
# 전체: ~80~180GB
uv run mcbot ingest ohlcv --symbols XRP/USDT ADA/USDT ... --timeframe 1m --start 2020-01-01
```

### 3.3 상장 시점 처리

| 상황 | 처리 |
|------|------|
| **2020년 이전 상장** (BTC, ETH) | 2020-01-01부터 수집 |
| **2020년 이후 상장** (SUI, TIA) | 상장일부터 수집 |
| **상장폐지** (LUNA, FTT) | 데이터 복구 불가 → 제외 |

상장폐지 에셋을 제외하는 것은 약간의 survivorship bias를 남기지만,
현존 에셋 중 과거 상위→현재 하위인 에셋(EOS, XLM 등)을 포함하여 완화.

---

## 4. BacktestSurveillance Simulator 설계

### 4.1 핵심 로직

```python
class BacktestSurveillanceSimulator:
    """백테스트에서 라이브 Surveillance를 시뮬레이션.

    7일 rolling volume 기반으로 에셋 순위를 계산하고,
    라이브의 MarketSurveillanceService.scan()과 동일한
    ScanResult를 생성.
    """

    def __init__(
        self,
        config: SurveillanceConfig,
        volume_data: pd.DataFrame,  # index=datetime, columns=symbols, values=quote_volume
    ) -> None: ...

    def scan_at(self, timestamp: datetime) -> ScanResult:
        """특정 시점의 7일 rolling volume으로 에셋 순위 계산."""
        # 1. timestamp 기준 7일 이전~현재 volume 합산
        # 2. min_24h_volume_usd 필터 (7일 평균으로 환산)
        # 3. min_listing_age_days 필터
        # 4. 스테이블코인 제외
        # 5. max_total_assets 상한
        # 6. 이전 universe와 diff → added/dropped/retained
        # 7. ScanResult 반환
        ...
```

### 4.2 Volume 데이터 사전 계산

```python
def build_volume_matrix(parquet_dir: Path, symbols: list[str]) -> pd.DataFrame:
    """1m Parquet에서 7일 rolling quote volume matrix 생성.

    Returns:
        DataFrame: index=datetime(1m), columns=symbols,
                   values=rolling_7d_quote_volume (= sum(close * volume))
    """
    # 1m bar의 close * volume = quote_volume
    # 7일 = 10,080분 rolling sum
    ...
```

**성능 최적화**: 1m 데이터 전체를 rolling하면 느리므로,
1H 또는 1D 단위로 사전 집계 후 7일 rolling sum 계산.

### 4.3 OrchestratedRunner 통합

```python
# orchestrated_runner.py 변경

class OrchestratedRunner:
    def __init__(
        self,
        ...,
        surveillance_simulator: BacktestSurveillanceSimulator | None = None,
    ) -> None:
        self._surveillance_sim = surveillance_simulator

    async def _on_tf_bar(self, bar: BarEvent) -> None:
        # 기존 로직 ...

        # Surveillance 체크 (7일 주기)
        if self._surveillance_sim is not None:
            if self._should_scan(bar.bar_timestamp):
                scan_result = self._surveillance_sim.scan_at(bar.bar_timestamp)
                if scan_result.added or scan_result.dropped:
                    await self._orchestrator.on_universe_update(
                        scan_result, self._warmup_from_parquet
                    )
```

### 4.4 Warmup 데이터 주입

라이브에서는 REST API로 warmup bars를 fetch하지만,
백테스트에서는 **이미 로드된 Parquet에서 슬라이싱**:

```python
async def _warmup_from_parquet(
    self, symbol: str, timeframe: str, periods: int
) -> tuple[list[dict], list[datetime]]:
    """Parquet 데이터에서 현재 시점 이전 N개 bar 추출."""
    # 이미 메모리에 있는 wide universe 데이터에서 슬라이싱
    ...
```

---

## 5. Wide Universe 데이터 관리

### 5.1 데이터 로딩 전략

40+ 에셋의 1m 데이터를 전부 메모리에 올리면 ~50GB+ → 불가능.

**해결**: Lazy Loading + Parquet Partition

```
data/
  silver/
    ohlcv_1m/
      BTC_USDT/         # 이미 존재
        2024-01.parquet
        2024-02.parquet
        ...
      XRP_USDT/         # 신규 수집
        2024-01.parquet
        ...
```

| 데이터 종류 | 메모리 상주 | 접근 방식 |
|------------|-----------|----------|
| **운용 에셋** (현재 top N) | 1m 전체 | HistoricalDataFeed에 로드 |
| **후보군 에셋** (나머지) | Volume만 | 1D quote_volume 사전 집계 |
| **신규 진입 에셋** | Warmup 구간만 | Parquet에서 on-demand 슬라이싱 |

### 5.2 Volume Matrix 사전 집계

```
preprocessing 단계:
  1m Parquet (40 에셋) → 1D quote_volume summary → ~1MB CSV/Parquet
  이 1D summary만 메모리에 로드하여 7D rolling 계산
```

---

## 6. 검증 계획

### 6.1 Parity 검증

| 검증 항목 | 방법 |
|----------|------|
| **Surveillance 로직 일치** | `BacktestSurveillanceSimulator.scan_at()` vs `MarketSurveillanceService.scan()` 동일 필터 |
| **on_universe_update 경로** | 라이브와 동일 메서드 호출 (코드 공유) |
| **Pod add/drop 동작** | 기존 `test_orchestrator_surveillance.py` 확장 |
| **Warmup 데이터 정합성** | Parquet 슬라이싱 vs REST API fetch 결과 비교 |

### 6.2 Survivorship Bias 정량화

```
Test A: 고정 심볼 백테스트 (현재 방식)
  → 2024-2025, top 5 고정 (BTC/ETH/SOL/DOGE/BNB)

Test B: 동적 심볼 백테스트 (신규)
  → 2024-2025, 7일마다 top 5 교체

비교 지표:
  - Sharpe 차이 (A > B 이면 survivorship bias 존재)
  - MDD 차이
  - 에셋 교체 빈도 & turnover cost
```

### 6.3 멀티에셋 범용성 검증

```
전략 3개 × Wide Universe 풀 전체 → per-asset P4 결과:
  - "모든 상위 에셋에서 Sharpe > 0.5" 확인
  - 특정 에셋에서만 작동하면 동적 교체 시 성과 불안정
```

---

## 7. 구현 단계

### Phase 1: 데이터 수집 (선행 조건)

```
1-1. 후보군 에셋 리스트 확정 (~40개)
1-2. mcbot ingest로 1m OHLCV 수집 (에셋당 2~4GB)
1-3. 1D quote_volume summary 사전 집계
```

**소요**: 데이터 수집 ~2~3일 (API rate limit)

### Phase 2: BacktestSurveillanceSimulator 구현

```
2-1. BacktestSurveillanceSimulator 클래스 구현
     - build_volume_matrix(): 1D summary → 7D rolling
     - scan_at(): ScanResult 생성 (기존 SurveillanceConfig 재사용)
2-2. OrchestratedRunner 통합
     - _should_scan() 주기 체크
     - _warmup_from_parquet() Parquet 슬라이싱
2-3. Wide Universe 데이터 로더
     - Lazy Parquet loading
     - Volume matrix 캐싱
```

**소요**: ~3~4일

### Phase 3: 검증

```
3-1. 단위 테스트 (Simulator, Warmup, Volume Matrix)
3-2. Parity 검증 (고정 vs 동적 백테스트 비교)
3-3. 멀티에셋 범용성 검증 (전략 × 에셋 매트릭스)
3-4. Survivorship bias 정량화
```

**소요**: ~2~3일

### Phase 4: 라이브 Surveillance 활성화

```
4-1. orchestrator-live.yaml 설정 변경
     - pinned_symbols: false
     - surveillance.enabled: true
4-2. Paper 모드 1주간 관찰
4-3. 라이브 전환
```

**소요**: ~1주 (paper 관찰 포함)

---

## 8. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| **데이터 저장 용량** (~150GB) | 디스크 부담 | 월별 Parquet 파티션, 압축 |
| **메모리 부담** (40+ 에셋) | OOM 가능 | Lazy loading, volume matrix만 상주 |
| **상장폐지 에셋 미포함** | 잔여 survivorship bias | 현존 하락 에셋(EOS 등) 포함으로 완화 |
| **거래대금 ≠ 전략 적합성** | 상위 에셋이라도 전략이 안 맞을 수 있음 | 멀티에셋 범용성 검증(Phase 3-3)으로 사전 확인 |
| **교체 시 turnover cost** | 과도한 교체 → 비용 증가 | cooldown 기간, 최소 보유 기간 설정 |

---

## 9. 설정 예시

### 9.1 백테스트 설정

```yaml
# config/orchestrator-dynamic.yaml
orchestrator:
  pods:
    - pod_id: pod-tri-channel
      strategy: tri-channel-12h
      timeframe: "12H"
      symbols: []              # 초기 비어있음 → Surveillance가 채움
      pinned_symbols: false
      max_assets: 5

  surveillance:
    enabled: true
    scan_interval_hours: 168.0   # 7일
    min_24h_volume_usd: 50_000_000
    min_listing_age_days: 90
    max_total_assets: 10
    max_assets_per_pod: 5
```

### 9.2 CLI 사용 예시

```bash
# Wide Universe 백테스트 실행
uv run mcbot eda run config/orchestrator-dynamic.yaml \
  --wide-universe data/silver/ohlcv_1m/ \
  --start 2023-01-01 \
  --end 2025-12-31
```

---

## 10. 기대 효과

| 지표 | 고정 심볼 (현재) | 동적 심볼 (예상) |
|------|-----------------|-----------------|
| **Survivorship bias** | 존재 | 제거/최소화 |
| **유동성 로테이션 포착** | 불가 | 가능 |
| **백테스트-라이브 parity** | 불일치 | 일치 |
| **에셋 분산** | 4~5개 고정 | 10~20개 동적 |
| **검증 신뢰도** | 특정 에셋 의존 | 범용 검증 |

---

## 참고

- 기존 Surveillance 구현: `src/orchestrator/surveillance.py`
- 기존 on_universe_update: `src/orchestrator/orchestrator.py:1163`
- Pod 동적 에셋: `src/orchestrator/pod.py:657`
- LiveDataFeed add_symbol: `src/eda/live_data_feed.py:174`
- 라이브 Surveillance 루프: `src/eda/live_runner.py:2016`
