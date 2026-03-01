# Lessons 시스템

185개 전략 평가 과정에서 축적된 97개 교훈을 구조화 관리합니다.
전략 발굴(P1) 시 과거 실패 패턴을 참조하여 중복 시도를 방지합니다.

---

## 1. YAML 스키마

각 교훈은 `lessons/{id:03d}.yaml` 파일로 관리됩니다.

```yaml
id: 42
title: "Multi-scale 앙상블 = 파라미터 로버스트니스"
body: "3-scale Donchian → 5/5 파라미터 100% 고원. 단일 lookback 대비 안정성 극대화"
category: strategy-design
tags:
  - multi-scale
  - robustness
  - donchian
strategies:
  - donch-multi
timeframes:
  - 12H
added_at: "2026-02-15"
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | int | 고유 번호 (1부터 순차) |
| `title` | string | 한 줄 요약 |
| `body` | string | 상세 내용 |
| `category` | string | 분류 (아래 참조) |
| `tags` | list[string] | 검색용 태그 |
| `strategies` | list[string] | 관련 전략 (kebab-case) |
| `timeframes` | list[string] | 관련 TF |
| `added_at` | date string | 등록일 |

---

## 2. 카테고리

| 카테고리 | 건수 | 설명 |
|----------|------|------|
| `strategy-design` | 69 | 전략 설계 패턴/반패턴 |
| `pipeline-process` | 11 | 파이프라인 프로세스 개선 |
| `data-resolution` | 7 | 데이터 해상도/품질 이슈 |
| `market-structure` | 6 | 시장 구조적 특성 |
| `risk-management` | 3 | 리스크 관리 교훈 |
| `meta-analysis` | 1 | 메타 분석 결과 |

---

## 3. CLI 명령

```bash
# 전체 교훈 목록
uv run mcbot pipeline lessons-list

# 카테고리 필터
uv run mcbot pipeline lessons-list -c strategy-design

# 교훈 상세 조회
uv run mcbot pipeline lessons-show 42
```

---

## 4. 핵심 교훈 (Top 10)

| # | 교훈 | 카테고리 |
|---|------|----------|
| 1 | 4H/8H 전멸: 거래비용이 edge 잠식 | strategy-design |
| 2 | 1D 고갈: 92개 OHLCV 시도 → 0 ACTIVE | strategy-design |
| 3 | 12H 최적: 비용/노이즈/신호 균형점 | strategy-design |
| 4 | Multi-scale 앙상블 = 파라미터 로버스트니스 | strategy-design |
| 5 | Multi-channel 앙상블 = 측정 직교성 | strategy-design |
| 6 | SL/TS 방어: Trailing Stop 3.0x ATR 핵심 | risk-management |
| 7 | 대안데이터 단독 alpha 부재 (181건 검증) | strategy-design |
| 8 | ML 전략 전멸: look-ahead bias | pipeline-process |
| 9 | HEDGE_ONLY EDA parity 위험: 81% 편차 | data-resolution |
| 10 | 심볼 비중복 필수: netting 상쇄 위험 | strategy-design |

---

## 5. 파이프라인 연동

### P1 Alpha Research

`/p1-research` 스킬에서 scorecard 작성 시 기존 교훈을 자동 참조합니다.

- 동일 TF/카테고리 전략의 실패 패턴 확인
- 기존 ACTIVE 전략과의 직교성 검증
- 과거 시도한 접근법 중복 방지

### Strategy YAML

교훈이 전략 의사결정에 영향을 준 경우 `decisions` 필드에 기록:

```yaml
decisions:
  - date: "2026-02-15"
    phase: P1
    verdict: PASS
    rationale: "교훈 #42 참조 — multi-scale 앙상블 채택"
```
