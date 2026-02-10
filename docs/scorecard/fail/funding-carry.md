# 전략 스코어카드: Funding Carry

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Funding Carry (`funding-carry`) |
| **유형** | 캐리 (Funding rate premium) |
| **타임프레임** | 1D |
| **상태** | `폐기` |
| **Best Asset** | N/A |
| **경제적 논거** | 무기한 선물 펀딩비의 구조적 프리미엄을 캐리 수익으로 수취 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 25/30점
G1 백테스트  [FAIL] 데이터 부재로 검증 불가 → 폐기
```

### 폐기 사유

- `funding_rate` 컬럼이 Silver 데이터에 없음
- Binance Perpetuals API에서 펀딩비 히스토리 수집 파이프라인 미구축
- 데이터 인프라 투자 대비 기대 수익이 불확실하여 우선순위 하향 → 폐기 처리

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 25/30점 |
| 2026-02-10 | G1 | FAIL | funding_rate 데이터 부재, 인프라 투자 미진행 → 폐기 |
