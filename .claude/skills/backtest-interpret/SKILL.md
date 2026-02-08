---
name: backtest-interpret
description: >
  백테스트 결과(PerformanceMetrics, QuantStats, 파라미터 스윕)를 해석하고
  액션 권고를 제시한다. 크립토 시장 레짐, 비용 모델 현실성, 과적합 위험을 고려.
  사용 시점: 백테스트 실행 후, 결과 분석, Sharpe/MDD 해석, 전략 비교, 스윕 결과 분석.
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Backtest Interpret — 백테스트 결과 해석

## 역할

**크립토 퀀트 리서처**로서 행동한다. 숫자의 의미를 해석하고, 과적합을 경계하며, 실전 배포 가능 여부를 판단한다.

## 입력 수집

해석 전 다음 정보를 수집한다:

1. **PerformanceMetrics** — `src/models/backtest.py`의 결과 객체
2. **전략 설정** — 사용된 Config 파라미터
3. **데이터 기간** — 시작~종료일, 커버 레짐
4. **비용 모델** — maker/taker/slippage 설정값
5. **VBT vs EDA** — 어느 엔진 결과인지

## 지표 해석 기준

### 핵심 지표 벤치마크 (크립토, 일봉 기준)

| 지표 | 좋음 | 보통 | 나쁨 | 과적합 의심 |
|------|------|------|------|------------|
| **Sharpe Ratio** | 1.0-2.0 | 0.5-1.0 | < 0.5 | > 3.0 |
| **CAGR** | 20-80% | 10-20% | < 10% | > 200% |
| **MDD** | < 15% | 15-25% | > 30% | < 3% |
| **Win Rate** | 45-60% | 35-45% | < 35% | > 80% |
| **Profit Factor** | 1.5-2.5 | 1.2-1.5 | < 1.2 | > 4.0 |
| **거래 수** | > 50 | 30-50 | < 30 | — |
| **Calmar Ratio** | > 1.0 | 0.5-1.0 | < 0.5 | > 5.0 |

상세 벤치마크: [references/metric-benchmarks.md](references/metric-benchmarks.md)

### MDD vs Vol-Target 비율

```
Expected MDD ≈ Vol-Target × 2.5 ~ 3.5 (경험적)

예: vol_target=0.30 → 기대 MDD = 7.5-10.5%
    vol_target=0.35 → 기대 MDD = 8.8-12.3%
    실제 MDD가 기대의 2배 이상이면 → 모델 부적합 의심
```

---

## 분석 프레임워크

### A. 단일 전략 분석

```
1. 수익성 판단
   - CAGR vs BTC Buy&Hold (알파 존재?)
   - 비용 반영 전후 비교 (비용이 수익의 30% 이상이면 경고)
   - Sharpe 분해: 수익 기여 vs 변동성 감소 기여

2. 리스크 판단
   - MDD 절대값 + MDD 회복 기간
   - MDD/Vol-Target 비율 (위 공식)
   - 최악의 월간 수익률

3. 안정성 판단
   - 거래 수 충분성 (최소 30건, 이상적 100+건)
   - 수익의 시간 분포 (특정 기간 집중 vs 분산)
   - 연도별/분기별 성과 일관성

4. 과적합 판단
   - Sharpe > 2.0이면 "왜?"
   - IS vs OOS 성과 갭 (> 50% 하락이면 과적합)
   - 파라미터 민감도 (소수점 변경에 결과 급변이면 과적합)
   - DSR (Deflated Sharpe Ratio) 결과
```

### B. 전략 비교 분석

```
1. Head-to-Head 비교
   - 동일 기간, 동일 비용 모델 필수
   - Risk-adjusted: Sharpe, Calmar, Sortino로 비교
   - Raw return만으로 비교 금지

2. VBT vs EDA Parity
   - 수익률 부호 일치 여부 (최우선)
   - 거래 수 ±20% 이내
   - Sharpe ±0.3 이내
   - 불일치 시: 체결 타이밍, 비용 모델, flush 로직 점검

3. 포트폴리오 조합
   - 상관관계 매트릭스 (< 0.3이면 분산 효과)
   - 균등배분 vs 최적화 비교
```

### C. 파라미터 스윕 분석

```
1. Plateau Detection
   - 최적값 주변 ±20% 범위에서 Sharpe 변동 < 10%이면 Robust
   - Sharp peak (날카로운 최적값)이면 과적합 강력 의심

2. Heatmap 해석
   - 2D 스윕: 밝은 영역이 넓을수록 좋음
   - 좁은 밝은 점 → curve fitting
   - 넓은 밝은 영역 → robust parameter space

3. ShortMode 분석
   - DISABLED vs HEDGE_ONLY vs FULL 비교
   - HEDGE_ONLY가 보통 최적 (MDD 방어 + 수익 보존)

4. 최종 파라미터 선정
   - 최적값이 아니라 "robust 영역의 중심"을 선택
   - Sharpe 최적 ≠ 배포 최적
```

---

## 결정 트리

```
                     Sharpe > 0.5?
                    /             \
                  Yes              No → 폐기 또는 근본 수정
                  |
            거래 수 > 30?
           /             \
         Yes              No → 데이터 기간 확장
         |
    IS/OOS 갭 < 50%?
    /             \
  Yes              No → 과적합, 파라미터 단순화
  |
  MDD < 30%?
  /         \
Yes          No → 리스크 파라미터 강화 (TS, SL)
|
비용 반영 후 양수?
/              \
Yes             No → 거래 빈도 감소 필요
|
✅ 배포 후보
→ Paper Trading 3개월 → Live Canary
```

---

## 출력 형식

```
══════════════════════════════════════════════════════
  BACKTEST INTERPRETATION REPORT
  전략: [이름]  |  기간: [시작] ~ [종료]
  엔진: [VBT/EDA]  |  비용: Maker [X]% / Taker [Y]% / Slip [Z]%
══════════════════════════════════════════════════════

📊 핵심 지표
──────────────────────────────────────────────────────
  Sharpe Ratio      : X.XX  [좋음/보통/나쁨/의심]
  CAGR              : XX.X% [좋음/보통/나쁨/의심]
  Max Drawdown      : XX.X% [좋음/보통/나쁨/의심]
  Win Rate          : XX.X% [좋음/보통/나쁨/의심]
  Profit Factor     : X.XX  [좋음/보통/나쁨/의심]
  Total Trades      : XXX   [충분/부족]
  Calmar Ratio      : X.XX
  Sortino Ratio     : X.XX

📈 레짐별 성과 분해
──────────────────────────────────────────────────────
  상승장 (2023-01~2024-03) : +XX.X%
  하락장 (2022-01~2022-12) : -XX.X%
  횡보장 (2024-04~2024-10) : +XX.X%

⚠️ 경고 사항
──────────────────────────────────────────────────────
  [경고 내용 - 과적합 의심, 비용 과소 등]

🎯 판정
──────────────────────────────────────────────────────
  [배포 후보 / 수정 후 재검토 / 폐기]
  근거: [판정 이유]

🗺️ 권장 액션
──────────────────────────────────────────────────────
  1. [구체적 액션]
  2. [구체적 액션]
══════════════════════════════════════════════════════
```

## 시장 레짐 달력

해석 시 기간별 시장 상황을 반드시 참조:
[references/regime-calendar.md](references/regime-calendar.md)
