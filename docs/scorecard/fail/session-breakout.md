# 전략 스코어카드: Session Breakout

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Session Breakout (`session-breakout`) |
| **유형** | Structural / Session Decomposition |
| **타임프레임** | 1H |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | N/A (전 에셋 Sharpe 음수) |
| **경제적 논거** | Asian session (00-08 UTC) low-vol range를 EU/US 세션 open 시 breakout 포착 |

---

## 성과 요약 (6년, 2020-2025, 1H)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate | Alpha | Beta |
|------|------|--------|------|-----|--------|------|----------|-------|------|
| 1 | DOGE/USDT | -1.67 | -29.9% | -89.0% | 1649 | 0.67 | 36.8% | -6121.6% | 0.02 |
| 2 | SOL/USDT | -2.26 | -29.5% | -88.4% | 1457 | 0.63 | 37.9% | -4365.2% | 0.00 |
| 3 | ETH/USDT | -2.56 | -34.7% | -92.7% | 1676 | 0.61 | 35.2% | -2297.2% | -0.01 |
| 4 | BNB/USDT | -3.39 | -40.1% | -95.4% | 1615 | 0.58 | 34.0% | -6290.1% | 0.00 |
| 5 | BTC/USDT | -3.49 | -42.6% | -96.6% | 1671 | 0.52 | 31.7% | -1228.5% | 0.00 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 27/30점
G0B 코드검증 [PASS]
G1 백테스트  [FAIL] 전 에셋 Sharpe 음수 (-1.67 ~ -3.49), MDD 88~97%. 즉시 폐기.
G2 IS/OOS    [    ]
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
```

### Gate 1 상세

**판정**: **FAIL** (즉시 폐기 — 전 에셋 Sharpe 음수)

**즉시 폐기 사유**:

- 즉시 폐기 조건 #2 해당: **전 5개 에셋에서 Sharpe 음수** (-1.67 ~ -3.49)
- 즉시 폐기 조건 #1 해당: **전 5개 에셋에서 MDD > 50%** (88.4% ~ 96.6%)
- Profit Factor 전 에셋 < 1.0 (0.52 ~ 0.67)
- Win Rate 32~38% (단순 동전 던지기보다 열등)
- CAGR 전 에셋 음수 (-29.5% ~ -42.6%)

**근본 원인 분석**:

1. **과다 거래 + 낮은 Win Rate**: 연평균 250~280건 거래 x Win Rate 32~38% = 지속적 손실 누적. 비용 모델 (편도 ~0.11%) 적용 시 연 55~62건의 추가 비용 drag.

2. **Session Breakout Edge 부재 (크립토 시장)**: FX 시장의 Asian session breakout은 institutional flow의 시간대 분리에 기반. 크립토는 24/7 시장으로 명확한 session 분리가 없음. 00-08 UTC range가 EU/US open 시 의미 있는 accumulation zone이 아닐 가능성.

3. **Range Squeeze 필터 미작동**: range_pctl < 50 필터가 squeeze를 감지하지 못하거나, squeeze 후 breakout이 시스템적으로 반전(false breakout)되는 패턴.

4. **1H Timeframe + 높은 거래 빈도**: 1H에서 매일 breakout 시그널 발생 → 대부분 false breakout → 손실 누적. Stop-loss가 Asian range 반대쪽인데, 양방향 whipsaw로 양쪽 모두 피격.

5. **Vol-target 사이징 + 음의 기대값**: vol-target이 음의 기대값 전략에 적용되면 손실을 확대. 변동성이 낮을수록(vol_scalar 증가) 더 큰 포지션 → 더 큰 손실.

**CTREND 비교**:

| 지표 | CTREND Best (SOL) | Session-Breakout Best (DOGE) |
|------|-------------------|------------------------------|
| Sharpe | 2.05 | -1.67 |
| CAGR | +97.8% | -29.9% |
| MDD | -27.7% | -89.0% |
| Trades | 288 | 1649 |

**수정 불가 판단**: 전 에셋 Sharpe -1.67 이하, Win Rate < 40%는 파라미터 튜닝으로 개선 불가능한 수준. 기대값 자체가 음수인 전략 구조.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 27/30점. FX에서 검증된 session breakout 가설 |
| 2026-02-10 | G0B | PASS | 코드 품질 검증 통과 |
| 2026-02-10 | G1 | FAIL | 전 에셋 Sharpe 음수 (-1.67 ~ -3.49), MDD 88~97%. 크립토 24/7 시장에서 session decomposition edge 부재 |
