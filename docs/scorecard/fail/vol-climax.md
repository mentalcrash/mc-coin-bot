# 전략 스코어카드: Vol-Climax

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Vol-Climax (`vol-climax`) |
| **유형** | Mean Reversion (Volume spike capitulation/euphoria reversal) |
| **타임프레임** | 4H (annualization_factor=2190) |
| **상태** | `폐기 (Gate 1 FAIL — 4H 재검증 확정)` |
| **Best Asset** | N/A (전 에셋 Sharpe 음수, 4H 재검증) |
| **경제적 논거** | 극단적 거래량 스파이크(climax)는 집단 항복/과열 → 에너지 소진 → 단기 반전 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate |
|------|------|--------|------|-----|--------|------|---------|
| **1** | **SOL/USDT** | **0.48** | +0.0% | -121.1% | 8 | 0.10 | 28.6% |
| 2 | ETH/USDT | 0.01 | -3.6% | -11.0% | 27 | 1.03 | 48.1% |
| 3 | DOGE/USDT | -0.09 | -168.3% | -34.9% | 32 | 0.59 | 48.4% |
| 4 | BNB/USDT | -0.15 | -7803.3% | -102.1% | 39 | 0.06 | 53.8% |
| 5 | BTC/USDT | -0.31 | -70.1% | -6.8% | 18 | 0.68 | 33.3% |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.48 | > 1.0 | **FAIL** |
| CAGR | +0.0% | > 20% | **FAIL** |
| MDD | -121.1% | < 40% | **FAIL** |
| Trades | 8 | > 50 | **FAIL** |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 22/30점
G0B 코드검증 [PASS] C1-C7 전항목 PASS (2026-02-12 재검증). Warning 1건 (W3 Regime)
G1 백테스트  [FAIL] 4H 재검증 확정. 전 에셋 Sharpe 음수 (-0.05 ~ -1.73)
```

### Gate 1 상세

- **즉시 폐기 조건 해당**: BNB MDD 102.1% > 50%, SOL MDD 121.1% > 50% (즉시 폐기 #1)
- **전 에셋 Sharpe 음수 근접**: Best SOL 0.48, 나머지 4개 에셋 모두 음수 또는 0 근접
- **BNB CAGR -7803%**: 비정상 수치. annualization_factor 불일치 + HEDGE_ONLY 모드에서도 과대 레버리지
- **거래 수 부족**: SOL 8건, BTC 18건으로 통계적 유의성 없음
- **volume climax 1D 비적합**: 4H climax_threshold=2.5가 1D 데이터에서 과도하게 희박한 시그널 생성 (SOL 6년간 8건)
- **OBV divergence 무효**: 1D 데이터에서 obv_lookback=6(4H에서 1일)이 1D에서 6일로 해석 → divergence 감지 지연

**근본 원인**: Volume climax 전략은 높은 시간 해상도(4H/1H)에서 단기 급등/급락의 에너지 소진을 포착하는 전략. 1D 데이터에서는 intraday volume spike가 평활화되어 climax signal 정보 손실이 심각.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 22/30점 |
| 2026-02-10 | G0B | PASS | Critical 7항목 결함 0개 |
| 2026-02-11 | G1 | **FAIL** | 1D 왜곡: BNB MDD 102%, SOL MDD 121% |
| 2026-02-12 | G1 (4H 재검증) | **FAIL 확정** | 전 에셋 Sharpe 음수 (-0.05~-1.73). MDD: SOL -73.6%, DOGE -74.3%, ETH -50.9%. Volume climax 반전이 4H에서도 무효 |
