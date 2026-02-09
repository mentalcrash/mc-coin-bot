# 전략 평가 표준 (Strategy Evaluation Standard)

> **목적**: 모든 전략을 동일한 기준으로 평가하기 위한 단일 참조 문서.
> Gate 0 ~ Gate 7까지의 통과 기준, 평가 방법, 의사결정 규칙을 정의한다.
>
> **핵심 원칙**: **단일 에셋 평가**. 멀티에셋 포트폴리오가 아닌, 전략별 최적 에셋 1개에 대해 평가한다.

---

## 평가 흐름 요약

```
Gate 0: 아이디어 검증 (코드 작성 전)
  ↓ PASS (18/30점)
Gate 1: 단일에셋 백테스트 → Best Asset 선정 + Best Timeframe 선정
  ↓ PASS (Sharpe > 1.0)
Gate 2: IS/OOS 검증 (Best Asset, 70/30)
  ↓ PASS (OOS Sharpe > 0.3, Decay < 50%)
Gate 3: 파라미터 안정성 (고원 + ±20% 안정)
  ↓ PASS
Gate 4: 심층 검증 (WFA + CPCV + PBO + DSR)
  ↓ PASS
Gate 5: EDA Parity (VBT vs EDA 수익 부호 일치)
  ↓ PASS
Gate 6: 모의 거래 (Paper Trading, 2주+)
  ↓ PASS
Gate 7: 실전 배포 + 모니터링
```

> 각 Gate는 순서대로 통과해야 다음으로 진행한다. FAIL 시 해당 Gate에서 멈추거나 폐기.

---

## 공통 규칙

### 평가 대상: 단일 에셋

- 모든 Gate는 **단일 에셋**에 대해 수행한다.
- Gate 1에서 Best Asset을 선정하고, 이후 Gate는 해당 에셋으로 진행한다.
- 2위, 3위 에셋도 **가능하면 함께 검증**하되, 판정은 Best Asset 기준이다.

### 타임프레임 선정

- 전략 특성에 맞는 타임프레임을 **Gate 1에서 선정**한다.
- 선정 기준: Sharpe가 가장 높은 (asset, timeframe) 조합.
- 후보 타임프레임: `1D`, `4h`, `1h` (전략에 따라 추가 가능).
- 선정된 타임프레임은 이후 모든 Gate에서 동일하게 사용한다.

### 백테스트 기간

- **기본 기간**: 2020-01-01 ~ 2025-12-31 (6년).
- **필수 요건**: 상승장(2020-2021), 하락장(2022), 횡보장(2023), 회복장(2024-2025) 레짐이 모두 포함.
- **초기 자본**: $100,000.

### 비용 모델

| 항목 | 기본값 | 설명 |
|------|-------|------|
| Maker Fee | 0.02% | 지정가 주문 |
| Taker Fee | 0.04% | 시장가 주문 |
| Slippage | 0.05% | 호가 미끄러짐 |
| Funding Rate (8h) | 0.01% | 무기한 선물 |
| Market Impact | 0.02% | 시장 충격 |
| **편도 합계** | **~0.11%** | |
| **왕복 합계** | **~0.22%** | |

---

## Gate 0: 아이디어 검증

> 코드를 한 줄도 작성하기 전에, 아이디어의 경제적 타당성을 평가한다.

### 평가 항목 (6개, 각 1~5점)

| 항목 | 기준 |
|------|------|
| **경제적 논거** | 5=행동편향/구조적 제약으로 설명 가능, 1=설명 불가 |
| **참신성** | 5=미공개, 1=SSRN 공개 3년+ |
| **데이터 확보성** | 5=Binance API 즉시 사용, 1=별도 수집 필요 |
| **구현 난이도** | 5=VectorBT 단순 구현, 1=HFT 인프라 필요 |
| **수용 용량** | 5=$1M+ 운용, 1=$100K 미만 |
| **레짐 의존성** | 5=모든 시장 환경, 1=특정 레짐만 |

### 통과 기준

| 판정 | 조건 |
|------|------|
| **PASS** | 합계 **>= 18/30** |
| **FAIL** | 합계 < 18/30 |

### 필수 기록

- 아이디어 출처 (논문/관찰/직관)
- 예상 거래 빈도 및 보유 기간
- 경제적 논거 1~2문장

---

## Gate 1: 단일에셋 백테스트

> 5개 에셋 x 6년 백테스트 후, **Best Asset + Best Timeframe**을 선정한다.

### 평가 방법

1. **에셋 후보**: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, DOGE/USDT
2. **타임프레임 후보**: 전략 기본 TF (1D) + 적합한 TF 추가 탐색
3. 각 (에셋, TF) 조합에 대해 `BacktestEngine.run()` 실행
4. **Best Asset**: Sharpe 최고인 에셋 선정
5. **2위, 3위**: 차선 에셋도 기록 (이후 Gate에서 참고)

### 통과 기준 (Best Asset 기준)

| 판정 | 조건 |
|------|------|
| **PASS** | Sharpe > **1.0** AND MDD < **40%** AND Trades > **50** |
| **WATCH** | 0.5 <= Sharpe <= 1.0, 또는 25% <= MDD <= 40% |
| **FAIL** | (총수익 < 0 AND Trades < 20) 또는 (Sharpe < 0.5 AND Trades < 20) |

### 보조 기준 (참고, 판정에 직접 영향 없음)

| 지표 | 권장 |
|------|------|
| Profit Factor | > 1.3 |
| Win Rate | > 45% (추세추종), > 55% (평균회귀) |
| Sortino Ratio | > 1.5 |
| Calmar Ratio | > 1.0 |

### 즉시 폐기 기준

| 조건 | 사유 |
|------|------|
| MDD > **50%** | 사실상 파산 위험 |
| 총수익 < 0 + Trades < 20 | 수익성 없음 + 통계적 무의미 |
| Sharpe < 0 (모든 에셋) | 구조적 결함 |

### Gate 1 산출물

- **선정 에셋**: 예) SOL/USDT (Sharpe 1.33)
- **선정 타임프레임**: 예) 1D
- **2위 에셋**: 예) ETH/USDT (Sharpe 1.05)
- **3위 에셋**: 예) BTC/USDT (Sharpe 0.92)

---

## Gate 2: IS/OOS 검증

> Best Asset에 대해 IS/OOS 70/30 분할 검증. 과적합 여부를 판단한다.

### 평가 방법

1. **데이터 분할**: 전체 기간의 앞 70%를 IS(학습), 뒤 30%를 OOS(검증)
2. **검증 대상**: Best Asset + 선정 TF
3. **추가 검증** (선택): 2위, 3위 에셋에 대해서도 동일 검증
4. CLI: `uv run python -m src.cli.backtest validate -m quick -s {strategy} --symbols {best_asset}`

### 통과 기준

| 지표 | 기준 | 설명 |
|------|------|------|
| **OOS Sharpe** | >= **0.3** | OOS에서도 양의 위험조정 수익 |
| **Sharpe Decay** | < **50%** | IS 대비 과도한 성과 하락 없음 |
| **OOS Total Return** | > **0%** | OOS에서 양수 수익 |

> **Sharpe Decay** = (IS Sharpe - OOS Sharpe) / IS Sharpe x 100%
> 50% 이상이면 과적합 의심. 100% 이상이면 OOS에서 손실.

### 판정

| 판정 | 조건 |
|------|------|
| **PASS** | OOS Sharpe >= 0.3 AND Decay < 50% |
| **FAIL** | OOS Sharpe < 0.3 또는 Decay >= 50% |

### 2위/3위 에셋 검증

- Best Asset이 PASS일 경우, 2위/3위 에셋도 동일 기준으로 검증한다.
- 2위/3위가 FAIL이어도 Best Asset PASS이면 Gate 2 통과.
- 결과는 스코어카드에 기록 (다각화 가능성 참고).

---

## Gate 3: 파라미터 안정성

> 파라미터 변동에 대한 전략의 로버스트성을 검증한다.

### 평가 방법

1. **Best Asset + 선정 TF**에 대해 파라미터 스윕 실행
2. 핵심 파라미터 각각에 대해 ±20% 범위로 그리드 서치
3. CLI: `uv run python -m src.cli.backtest sweep {config}`

### 통과 기준

| 항목 | 기준 | 설명 |
|------|------|------|
| **파라미터 고원** | 존재해야 함 | 넓은 범위에서 안정적 수익 (뾰족한 봉우리 X) |
| **±20% 안정성** | Sharpe 부호 유지 | 핵심 파라미터 ±20% 변경 시 양의 Sharpe 유지 |

### 판정

| 판정 | 조건 |
|------|------|
| **PASS** | 고원 존재 AND ±20% Sharpe 부호 유지 |
| **FAIL** | 뾰족한 봉우리만 존재, 또는 ±20%에서 Sharpe 부호 반전 |

### 추가 검증 (2위/3위 에셋)

- 가능하면 2위, 3위 에셋에서도 동일 파라미터 범위에서 고원이 존재하는지 확인.
- 에셋 간 최적 파라미터가 유사하면 로버스트성 높음.

---

## Gate 4: 심층 검증

> 통계적으로 유의미한 성과인지 검증한다. 다중 테스트 보정, 과적합 확률 추정.

### 평가 방법

| 검증 | 내용 | CLI |
|------|------|-----|
| **Walk-Forward Analysis** | 5-fold expanding window | `validate -m milestone` |
| **CPCV + PBO + DSR** | 조합적 교차검증 + 몬테카를로 | `validate -m final` |

### 통과 기준

| 지표 | 기준 | 설명 |
|------|------|------|
| **WFA OOS Sharpe** | >= **0.5** | Walk-Forward OOS 평균 |
| **WFA Sharpe Decay** | < **40%** | Walk-Forward 감쇠율 |
| **WFA Consistency** | >= **60%** | 양의 OOS Sharpe + Decay < 50% 비율 |
| **PBO** | < **40%** | 백테스트 과적합 확률 |
| **DSR** | > **0.95** | Deflated Sharpe (다중 테스트 보정) |
| **Monte Carlo p-value** | < **0.05** | 통계적 유의성 |

### 판정

| 판정 | 조건 |
|------|------|
| **PASS** | WFA + CPCV 모두 기준 충족 |
| **FAIL** | 하나라도 기준 미달 |

---

## Gate 5: EDA Parity

> VBT 백테스트와 EDA 이벤트 기반 백테스트의 결과가 일치하는지 검증한다.
> 실거래 전환 전 엔진 신뢰도를 확보하는 단계.

### 평가 방법

1. **Best Asset + 선정 TF**에 대해 EDA 백테스트 실행
2. CLI: `uv run python main.py eda run {config}`
3. VBT 결과와 비교

### 통과 기준

| 지표 | 기준 | 설명 |
|------|------|------|
| **수익 부호 일치** | 필수 | VBT 양수 → EDA 양수, VBT 음수 → EDA 음수 |
| **수익률 편차** | < **20%** | 절대 수익률 차이 허용 범위 |
| **거래 수 비율** | 0.5x ~ 2.0x | VBT 대비 EDA 거래 수가 극단적이지 않음 |

### 판정

| 판정 | 조건 |
|------|------|
| **PASS** | 수익 부호 일치 AND 편차 < 20% |
| **FAIL** | 수익 부호 불일치 또는 편차 >= 20% |

---

## Gate 6: 모의 거래 (Paper Trading)

> 실제 시장 데이터로 실시간 시그널을 생성하되, 체결은 시뮬레이션.

### 평가 방법

1. **Best Asset + 선정 TF**로 페이퍼 트레이딩 실행
2. **기간**: 최소 2주 (일봉 기준 10+ 거래일)
3. 백테스트 동일 기간과 비교

### 통과 기준

| 지표 | 기준 | 설명 |
|------|------|------|
| **백테스트-실전 일치도** | > **90%** | 시그널 방향 일치율 |
| **체결 편차** | < **10%** | 예상 체결가 vs 실제 체결가 |
| **시스템 안정성** | 무중단 운영 | 2주간 에러 없이 작동 |

### 판정

| 판정 | 조건 |
|------|------|
| **PASS** | 일치도 > 90% AND 체결 편차 < 10% AND 무중단 |
| **FAIL** | 기준 미달 |

---

## Gate 7: 실전 배포 + 모니터링

> 실제 자금으로 운용. 지속적 모니터링으로 전략 은퇴 시점을 판단.

### 배포 조건

- Gate 0~6 모두 PASS.
- 초기 자본은 검증 자본의 **10%** 이하로 시작 (점진적 증액).

### 모니터링 기준

| 지표 | 기준 | 액션 |
|------|------|------|
| **3개월 이동 Sharpe** | > **0.3** | 유지 |
| **3개월 이동 Sharpe** | 0 ~ 0.3 | 경고 (자본 축소 검토) |
| **3개월 이동 Sharpe** | < **0** | 은퇴 (운용 중단) |
| **실시간 MDD** | < 백테스트 MDD | 유지 |
| **실시간 MDD** | > 백테스트 MDD x 1.5 | 긴급 중단 |

### 은퇴 절차

1. 포지션 전량 청산.
2. 스코어카드 상태를 `은퇴`로 변경.
3. 은퇴 사유와 날짜 기록.
4. 코드는 유지 (참고용). 폐기와 다름.

---

## 판정 요약 테이블

| Gate | 평가 대상 | 핵심 기준 | PASS 조건 |
|------|----------|----------|----------|
| **0** | 아이디어 | 경제적 타당성 | >= 18/30점 |
| **1** | 5 에셋 x TF | Best Asset 선정 | Sharpe > 1.0, MDD < 40%, Trades > 50 |
| **2** | Best Asset | IS/OOS 70/30 | OOS Sharpe >= 0.3, Decay < 50% |
| **3** | Best Asset | 파라미터 스윕 | 고원 존재, ±20% 안정 |
| **4** | Best Asset | WFA + CPCV | WFE > 50%, PBO < 40%, DSR > 0.95 |
| **5** | Best Asset | VBT vs EDA | 수익 부호 일치, 편차 < 20% |
| **6** | Best Asset | 페이퍼 트레이딩 | 일치도 > 90%, 2주+ 무중단 |
| **7** | Best Asset | 실전 운용 | 3개월 Sharpe > 0.3 |

---

## 스코어카드

각 전략의 평가 결과는 스코어카드에 기록된다.

### 파일 구조

```
docs/scorecard/
├── template.md                # 스코어카드 템플릿
├── {strategy-name}.md         # 활성 전략 스코어카드
└── fail/
    └── {strategy-name}.md     # 폐기 전략 스코어카드
```

- **활성 전략**: `docs/scorecard/{name}.md`
- **폐기 전략**: `docs/scorecard/fail/{name}.md`
- **자동 생성**: `uv run python scripts/generate_scorecards.py`
- **템플릿**: [docs/scorecard/template.md](scorecard/template.md)

### 작성 규칙

1. 스코어카드는 **간결하게** 작성한다 (템플릿 참조).
2. Gate 0~1은 자동 생성. Gate 2 이후는 해당 Gate 완료 시 추가.
3. TSMOM은 수동 관리 (`MANUAL_SCORECARDS`에 등록).
4. 폐기 전략은 `fail/` 디렉토리로 이동하고, 폐기 사유와 일자를 기록.

### Gate 칼럼 표기법

| 약어 | 의미 |
|------|------|
| `P` | PASS |
| `W` | WATCH (Gate 1 한정) |
| `F` | FAIL |
| `—` | PENDING (해당 Gate 미도달) |

### README 테이블 예시

```
| # | 전략 | Best Asset | TF | Sharpe | G0 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|------|-----------|-----|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | TSMOM | SOL/USDT | 1D | 1.33  |  P |  P |  P |  — |  — |  — |  — |  — |
```

---

## CLI 명령 참조

| Gate | 명령 |
|------|------|
| Gate 1 | `uv run python -m src.cli.backtest run {config}` |
| Gate 2 | `uv run python -m src.cli.backtest validate -m quick -s {name}` |
| Gate 3 | `uv run python -m src.cli.backtest sweep {config}` |
| Gate 4 (WFA) | `uv run python -m src.cli.backtest validate -m milestone -s {name}` |
| Gate 4 (CPCV) | `uv run python -m src.cli.backtest validate -m final -s {name}` |
| Gate 5 | `uv run python main.py eda run {config}` |
| Gate 6 | `uv run python main.py eda run {config} --mode shadow` |
| 일괄 백테스트 | `uv run python scripts/bulk_backtest.py` |
| 일괄 검증 | `uv run python scripts/bulk_validate.py` |
| 스코어카드 생성 | `uv run python scripts/generate_scorecards.py` |
