# Strategy Evaluation Log

테스트한 전략들의 평가와 교훈을 기록합니다.
같은 실수를 반복하지 않고 더 나은 전략을 개발하기 위한 지식 베이스입니다.

## 문서 구조

| 파일 | 내용 | 상태 |
|------|------|------|
| [tsmom.md](tsmom.md) | VW-TSMOM 단일 에셋 (BTC 8년 검증) | ✅ 채택 |
| [bb-rsi.md](bb-rsi.md) | BB+RSI 평균회귀 전략 | ⚠️ 보류 |
| [cross-strategy.md](cross-strategy.md) | TSMOM + BB+RSI 교차 분석 | 분석 완료 |
| [multi-asset.md](multi-asset.md) | 멀티에셋 TSMOM (8-asset EW) | ✅ 채택 |
| [hedge-optimization.md](hedge-optimization.md) | 헤지 파라미터 최적화 (threshold × strength) | ✅ 완료 |
| [meta-lessons.md](meta-lessons.md) | 메타 교훈 + 폐기 전략 템플릿 | 지속 업데이트 |
| [data-inventory.md](data-inventory.md) | 멀티에셋 Silver 데이터 현황 | 지속 업데이트 |

---

## 최종 확정 설정

```python
# 8-asset EW TSMOM Portfolio (확정)
assets = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
          "DOGE/USDT", "LINK/USDT", "ADA/USDT", "AVAX/USDT"]
weight = 1/8  # Equal-Weight
timeframe = "1D"  # 일봉 (4h~12h 대비 Sharpe +1~12%)

TSMOMConfig(
    lookback=30,              # 30일 모멘텀 (30 × 1D = 30일)
    vol_window=30,            # 30일 변동성
    vol_target=0.35,          # 균형 설정
    annualization_factor=365, # 일봉 연환산
    short_mode=ShortMode.HEDGE_ONLY,
    hedge_threshold=-0.07,    # -7% 드로다운 시 헤지 활성화
    hedge_strength_ratio=0.3, # 롱의 30%로 방어적 숏 (Sharpe 2.33, CAGR +49.1%)
)
PortfolioManagerConfig(max_leverage_cap=2.0)
```

## 핵심 수치 요약

| 지표 | BTC TSMOM | T1 EW (4) | T1+T2 EW (8) | EW(8) vt=0.35 | **EW(8) 헤지 최적화** | BTC B&H |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Sharpe (6년) | 1.26 | 1.89 | 1.98 | 2.06 | **2.33** | - |
| Sharpe (8년) | 1.04 | - | - | - | - | 0.69 |
| CAGR (6년) | +38.5% | +43.8% | +40.7% | +48.8% | **+49.1%** | - |
| CAGR (8년) | +31.3% | - | - | - | - | +26.5% |
| MDD | -35.4% | -21.8% | -20.2% | -23.5% | **-20.7%** | -81.2% |
| AnnVol | 30.5% | 23.1% | 20.5% | 23.7% | **21.1%** | 65.7% |
| Calmar | 1.09 | 2.01 | 2.02 | 2.08 | **2.37** | 0.33 |
| Sortino | - | - | - | 3.03 | **3.21** | - |
| 2022 Sharpe | -0.78 | **+0.42** | -0.43 | - | - | - |

## 기대 성과 (6년 백테스트, 2020-2025)

| Sharpe | CAGR | MDD | AnnVol | Calmar | Sortino | 총수익 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2.33** | **+49.1%** | **-20.7%** | 21.1% | **2.37** | **3.21** | **+997%** |

## 검증 완료 항목

| 검증 항목 | 결과 | 방법 |
|-----------|------|------|
| 에셋 구성 | 8-asset EW | 4 vs 8 비교 (Sharpe +5%) |
| vol_target | 0.35 | 8단계 스윕 (0.15~0.50), 320 백테스트 |
| max_leverage_cap | 2.0x | 5단계 스윕 (1.0~3.0x), 효과 없음 확인 |
| 타임프레임 | 1D | 4개 TF × 6 lookback, 192 백테스트 |
| lookback | 30 (1D) | 30일 수평선이 4개 TF 모두 최적 |
| hedge_threshold | -0.07 | 8단계 스윕 (-0.05~-0.30), threshold 둔감 확인 |
| hedge_strength_ratio | 0.3 | 10단계 스윕 (0.1~1.0), 656 백테스트, Sharpe +13% |

## 향후 방향

- ✅ ~~TSMOM 멀티에셋 확장~~ — **완료** (8-asset EW, Sharpe 1.98)
- ✅ ~~포트폴리오 구성 최종 결정~~ — **8-asset 확정**
- ✅ ~~vol_target × leverage_cap 스윕~~ — **완료** (최적: vol_target=0.35, Sharpe 2.06)
- ✅ ~~타임프레임 분석~~ — **완료** (1D 확정, 4h~12h 모두 열등)
- ✅ ~~숏 헤지 최적화~~ — **완료** (strength=0.3, Sharpe 2.33, 656 백테스트)
- ⚠️ 50/50 고정 합성 — Sharpe 소폭 개선이나 CAGR 절반, 우선순위 낮음
- ❌ Regime Adaptive 동적 전환 — 과적합 리스크 높음, 보류

### 다음 단계 (우선순위)

1. **실거래 인프라 구축** — 멀티에셋 동시 시그널 생성 + OMS 확장

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-02-06 | 초기 문서 생성, VW-TSMOM 평가 추가 |
| 2026-02-06 | BB+RSI 평가, 교차 전략 분석, 8년 데이터 검증 결과 추가 |
| 2026-02-06 | 2017-2022 BTC 데이터 수집, 데이터 인벤토리 추가 |
| 2026-02-06 | Tier 1 멀티에셋 TSMOM 검증 (BTC+ETH+BNB+SOL, Sharpe 1.89) |
| 2026-02-06 | Tier 2 확장 (DOGE+LINK+ADA+AVAX), 8-asset EW Sharpe 1.98, 레버리지 평가 |
| 2026-02-06 | 8-asset 구성 확정, vol_target×leverage_cap 스윕 (40조합), 최적 vol_target=0.35 발견 |
| 2026-02-06 | 타임프레임 분석 (4h/8h/12h/1D × 6 lookback, 192 백테스트), 1D/30d 최적 확정 |
| 2026-02-06 | 문서 분할 — 단일 파일에서 7개 파일 구조로 재편 |
| 2026-02-06 | 헤지 파라미터 최적화 (80조합 × 8에셋 = 656 백테스트), strength 0.3 확정, Sharpe 2.06→2.33 |
