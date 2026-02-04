# 📊 VW-TSMOM 전략 저베타(Low Beta) 분석 및 개선 가이드

이 문서는 VW-TSMOM(Volatility-Weighted Time Series Momentum) 전략의 Beta 0.09 현상에 대한 정밀 진단 결과와 이를 해결하기 위한 기술적/수학적 솔루션을 정리합니다.

- **작성일:** 2026-02-04
- **대상 전략:** VW-TSMOM
- **핵심 문제:** Beta 0.09 (시장 상관관계 극저하 및 상승장 Upside 포착 실패)

---

## 1. Executive Summary

현재 전략의 **Beta 0.09**는 단순 파라미터 문제가 아닌, **과도한 리스크 제어 필터**와 **변동성 스케일링의 역설**이 결합된 구조적 문제입니다.

| 요인 | 상태 | 영향 | 심각도 |
| :--- | :--- | :--- | :--- |
| **Trend Filter** | 활성화 (MA 50) | Counter-trend 신호 100% 차단으로 인한 신호 누락 | 🔴 Critical |
| **Vol Targeting** | 40% 고정 | 고변동성 상승장에서 포지션 강제 축소 (Beta 억제) | 🔴 Critical |
| **Deadband Filter** | 0.2 | 미세한 추세 신호를 노이즈로 간주하여 필터링 | 🟡 Medium |
| **Leverage Cap** | 2.0x | 저변동성 구간에서의 공격적 Beta 확보 제한 | 🟡 Medium |

---

## 2. Diagnostic Logging System Design

전략의 'Black Box'를 열고 Beta 손실 구간을 추적하기 위한 진단 로깅 설계입니다.

### 2.1 Signal Pipeline Logging
매 캔들마다 다음 지표를 기록하여 **"왜 이 포지션이 이 크기인가?"**를 추적합니다.

```python
# 제안하는 진단 데이터 스키마 (src/strategy/tsmom/diagnostics.py)
@dataclass(frozen=True, slots=True)
class SignalDiagnosticRecord:
    timestamp: datetime
    symbol: str
    
    # Market & Signal
    raw_momentum: float      # 원시 모멘텀
    vol_scalar: float        # vol_target / realized_vol
    
    # Filter Decisions
    trend_regime: int        # 1(Up), -1(Down), 0(Neutral)
    is_suppressed: bool      # 필터에 의해 신호가 죽었는가?
    suppression_reason: str  # "trend_filter", "deadband", "vol_scaling"
    
    # Final Weights
    target_weight: float     # 최종 집행 비중
```

### 2.2 Beta Attribution 분석
백테스트 결과에서 각 필터가 Beta를 얼마나 갉아먹었는지 정량화합니다.

- **Potential Beta:** 필터 없이 모든 신호를 추종했을 때의 Beta
- **Lost Beta (Trend Filter):** 트렌드 필터로 인해 놓친 Beta
- **Lost Beta (Vol Scaling):** 변동성 조절로 인해 축소된 Beta

---

## 3. Hypothesis on Low Beta (0.09)

### 3.1 수학적 배경
$\beta = \rho \cdot \frac{\sigma_{strategy}}{\sigma_{market}}$ 에서 Beta가 낮다는 것은 상관관계($\rho$)가 낮거나 전략의 변동성($\sigma_{strategy}$)이 시장보다 너무 작음을 의미합니다.

### 3.2 핵심 원인 분석
1.  **Trend Filter의 Binary 특성:** MA 50 기반의 필터가 시장의 미세한 되돌림(Pullback) 시점에 신호를 0으로 만들어 버려, 추세 재개 시점의 Beta 확보를 방해합니다.
2.  **Vol Targeting의 역설:** 시장 급등 시 실현 변동성(Realized Vol)이 상승하면 `vol_scalar`가 급감하여 포지션 사이즈가 줄어듭니다. 이는 상승장에서의 Beta를 강제로 낮추는 결과를 초래합니다.

---

## 4. Parametric & Structural Optimization

### 4.1 파라미터 최적화 가이드

| 파라미터 | 추천 방향 | 기대 효과 |
| :--- | :--- | :--- |
| `lookback` | 30 → 12~20 | 짧은 주기로 시장 변화에 기민하게 반응 (Beta ↑) |
| `deadband` | 0.2 → 0.05 | 신호 문턱을 낮추어 더 많은 추세 참여 (Beta ↑) |
| `vol_target` | 0.4 → 0.6 | 전체적인 리스크 예산 확대로 포지션 크기 증대 |
| `leverage_cap`| 2.0x → 3.0x | 저변동성 구간에서의 Upside 캡 완화 |

### 4.2 구조적 개선: Asymmetric Vol Targeting
상승 신호(Long) 시에는 변동성 스케일링을 완화하여 시장 상승분을 더 많이 확보합니다.

```python
# 개선 로직 예시
if signal > 0: # Long
    # 실현 변동성이 높아도 포지션을 덜 줄임
    vol_scalar = max(base_vol_scalar, min_floor_for_long) 
```

---

## 5. Portfolio & Execution Logic

1.  **Position Floor 도입:** 모멘텀 신호가 확실할 경우, 변동성과 무관하게 유지할 최소 비중(Min Position Floor, 예: 30%)을 설정합니다.
2.  **Rebalance Threshold 최적화:** 현재 5%의 리밸런싱 문턱이 잦은 미세 조정을 막아주지만, 강한 추세 초입에서의 진입 지연을 유발할 수 있으므로 2-3%로 하향 조정을 검토합니다.

---

## 6. Action Plan (실행 로드맵)

1.  **[Step 1] Logging:** `src/strategy/tsmom/diagnostics.py` 모듈을 구현하여 백테스트 시 필터링 원인을 로그로 남깁니다.
2.  **[Step 2] Attribution:** 백테스트 엔진에 `Beta Attribution` 계산 로직을 추가하여 현재 필터들의 Beta 잠식률을 파악합니다.
3.  **[Step 3] Grid Search:** `lookback`과 `deadband`를 중심으로 Beta와 Sharpe Ratio의 조화 평균을 최적화하는 구간을 찾습니다.
4.  **[Step 4] Implementation:** `Asymmetric Vol Targeting` 및 `Soft Trend Filter`를 적용하여 구조적 문제를 해결합니다.

---

> **Note:** Beta 0.09는 시장 중립(Market Neutral) 전략에게는 훌륭한 수치이나, 추세 추종(Trend Following) 전략에게는 **"시장을 따라가지 못하고 있다"**는 강력한 경고 신호입니다. 위 로드맵에 따라 구조적 개선을 우선적으로 수행하십시오.
