# 전략 스코어카드: OU-MeanRev

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | OU-MeanRev (`ou-meanrev`) |
| **유형** | Mean Reversion (Ornstein-Uhlenbeck process) |
| **타임프레임** | 4H (annualization_factor=2190, 1D 데이터 백테스트) |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | SOL/USDT (Sharpe 0.07) |
| **경제적 논거** | OU process half-life가 짧으면 강한 mean-reversion → Z-score 진입, 길면 거래 중단 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate |
|------|------|--------|------|-----|--------|------|---------|
| **1** | **SOL/USDT** | **0.07** | +0.0% | -110.1% | 17 | 0.32 | 31.3% |
| 2 | BTC/USDT | 0.00 | -6.3% | -9.2% | 14 | 1.08 | 50.0% |
| 3 | ETH/USDT | -0.15 | -76.4% | -11.1% | 20 | 0.78 | 35.0% |
| 4 | DOGE/USDT | -0.18 | -9778.4% | -19669.1% | 3 | 0.10 | 50.0% |
| 5 | BNB/USDT | -0.19 | -99.2% | -13.8% | 20 | 0.79 | 35.0% |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.07 | > 1.0 | **FAIL** |
| CAGR | +0.0% | > 20% | **FAIL** |
| MDD | -110.1% | < 40% | **FAIL** |
| Trades | 17 | > 50 | **FAIL** |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 22/30점
G0B 코드검증 [PASS]
G1 백테스트  [FAIL] 전 에셋 Sharpe ~0, 4/5 음수. DOGE MDD -19669%. 즉시 폐기
```

### Gate 1 상세

- **즉시 폐기 조건 해당**:
  - DOGE MDD 19,669% > 50% (즉시 폐기 #1) — FULL short mode에서 밈코인 급등 노출
  - SOL MDD 110% > 50% (즉시 폐기 #1)
  - 4/5 에셋 Sharpe 음수 (즉시 폐기 #2 근접)
- **DOGE 파산 수준**: FULL short mode + DOGE 급등 = MDD -19,669%. VWAP-Disposition(MDD -622%)의 31.6배
- **전 에셋 수익 부재**: Best Asset SOL Sharpe 0.07, BTC 0.00 — edge 완전 부재
- **OU process 1D 부적합**: ou_window=120(4H에서 20일)이 1D에서 120일(4개월)로 해석 → half-life 추정 불안정, mean-reversion window가 극도로 길어져 MR edge 소실
- **FULL short mode 치명적**: DOGE/SOL 같은 고변동 에셋에서 OU z-score short 진입 → 추세 지속 시 파산

**근본 원인**:
1. **TF 불일치**: 4H 파라미터(ou_window=120 = 20일)가 1D에서 120일로 왜곡
2. **FULL short + 밈코인**: OU z-score 기반 short이 DOGE/SOL 급등에 노출 → 치명적 MDD
3. **MR edge 부재**: 크립토 1D는 추세 지속성이 강해 mean-reversion 전략이 구조적 불리

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 22/30점 |
| 2026-02-10 | G0B | PASS | Critical 7항목 결함 0개 |
| 2026-02-11 | G1 | **FAIL** | 즉시 폐기: DOGE MDD -19,669% (FULL short + 밈코인 급등), SOL MDD -110%. 전 에셋 Sharpe ~0, 4/5 음수. OU process 4H→1D TF 불일치 |
