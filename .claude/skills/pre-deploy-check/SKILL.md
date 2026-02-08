---
name: pre-deploy-check
description: >
  라이브/페이퍼 트레이딩 배포 전 안전 점검. 거래소 연결, 리스크 파라미터,
  서킷 브레이커, 포지션 동기화, 운영 준비 상태를 검증한다.
  사용 시점: 라이브 배포 전, 페이퍼 트레이딩 시작 전, 라이브 인프라 코드 리뷰 시.
disable-model-invocation: true
---

# Pre-Deploy Check — 라이브 배포 전 안전 점검

## 역할

**운영 안전 엔지니어**로서 행동한다.
판단 기준: **"이 코드가 내 돈을 운용해도 되는가?"**

이 체크리스트를 사용자와 함께 **하나씩** 확인한다.
자동 실행이 아닌 **대화형 점검**이다.

---

## 배포 티어

| 티어 | 환경 | 자금 | 필수 통과 |
|------|------|------|----------|
| **T0: Paper** | 시뮬레이션 | $0 | Tier 1-3 |
| **T1: Testnet** | 거래소 테스트넷 | $0 | Tier 1-4 |
| **T2: Canary** | 메인넷, 최소 자금 | < $100 | Tier 1-6 |
| **T3: Production** | 메인넷, 운용 자금 | 설정 금액 | 전체 통과 |

상세 티어 요구사항: [references/deployment-tiers.md](references/deployment-tiers.md)

---

## 점검 항목

### Tier 1: 코드 품질 (모든 티어)

```bash
# 1.1 린트 + 타입체크
uv run ruff check .
uv run pyright src/

# 1.2 테스트 전체 통과
uv run pytest --cov=src

# 1.3 보안 스캔
grep -rn "api_key.*=.*['\"]" --include="*.py" src/
grep -rn "api_secret.*=.*['\"]" --include="*.py" src/
# 결과 0건이어야 함
```

- [ ] ruff 0 error
- [ ] pyright 0 error
- [ ] pytest 전체 통과
- [ ] 하드코딩된 시크릿 없음
- [ ] `.env` 파일 `.gitignore`에 포함

### Tier 2: 백테스트 검증 (모든 티어)

- [ ] 최소 2년 이상 백테스트 기간
- [ ] IS/OOS 분리 검증 통과
- [ ] WFA 또는 CPCV 검증 통과
- [ ] 비용 모델 현실적 (Maker 0.02%+, Taker 0.04%+, Slippage 0.03%+)
- [ ] MDD < 30% (vol-target 대비 합리적)
- [ ] 거래 수 > 30 (통계 유의성)

### Tier 3: 리스크 파라미터 (모든 티어)

확인 대상 파일: `src/models/eda.py`, `config/default.yaml`

- [ ] `system_stop_loss` 설정됨 (기본 10%)
- [ ] `max_leverage_cap` 설정됨 (권장 2.0x 이하)
- [ ] `trailing_stop_atr_multiplier` 설정됨 (권장 3.0x)
- [ ] `rebalance_threshold` 설정됨 (권장 10%)
- [ ] Circuit Breaker 활성화 (RM에서 시스템 DD 초과 시 전량 청산)

**검증 방법:**
```bash
# EDA 백테스트로 리스크 파라미터 작동 확인
uv run python main.py eda run tsmom BTC/USDT --start 2022-01-01 --end 2022-12-31
# 2022 약세장에서 system_stop_loss 발동 여부 확인
```

### Tier 4: 거래소 연결 (Testnet+)

- [ ] API 키 권한 확인:
  - [ ] Read: 잔고, 포지션, 주문 조회
  - [ ] Trade: 주문 생성/취소
  - [ ] **NO** Withdraw (출금 권한 절대 부여 금지)
- [ ] IP 화이트리스트 설정
- [ ] `load_markets()` 성공
- [ ] `fetch_balance()` 성공
- [ ] Rate limit 준수 코드 확인 (Binance: 1200 req/min)
- [ ] WebSocket 연결 안정성 (5분 이상 유지)

Binance 특화 안전 패턴: [references/exchange-safety.md](references/exchange-safety.md)

### Tier 5: 주문 안전성 (Canary+)

- [ ] `client_order_id` 고유성 보장 (UUID 또는 타임스탬프 기반)
- [ ] `amount_to_precision()` 적용 (거래소 소수점 규격)
- [ ] `price_to_precision()` 적용 (해당 시)
- [ ] Decimal 연산 (float 금지)
- [ ] 최소 주문 금액 체크 (Binance: ~$5)
- [ ] 최대 주문 크기 제한 (일 거래량의 1% 이하)
- [ ] 멱등성: 동일 `client_order_id` 재전송 시 중복 체결 없음

**테스트:**
```python
# Testnet에서 실제 주문 cycle 테스트
# 1. 최소 주문 생성
# 2. 주문 상태 조회
# 3. 주문 취소
# 4. 포지션 확인
```

### Tier 6: 포지션 동기화 (Canary+)

- [ ] 시작 시 거래소 포지션 조회 → 로컬 상태와 비교
- [ ] 불일치 시 경고 + 수동 개입 요구 (자동 청산 금지)
- [ ] 재시작 시 이전 포지션 복구 (영속적 상태 또는 거래소 동기화)
- [ ] 네트워크 장애 시 포지션 안전 (열린 포지션 + 하드 SL)

### Tier 7: 운영 인프라 (Production)

- [ ] **로깅**: loguru 설정, 로그 파일 로테이션
- [ ] **알림**: Discord/Telegram 웹훅 (체결, 에러, 일일 요약)
- [ ] **Graceful Shutdown**: SIGINT/SIGTERM 시 열린 주문 취소 + 상태 저장
- [ ] **프로세스 감시**: systemd / supervisor / Docker restart policy
- [ ] **모니터링**: 에러율, 체결 지연, 수익률 대시보드

### Tier 8: 롤백 계획 (Production)

- [ ] **비상 정지 절차** 문서화:
  1. 봇 프로세스 중지
  2. 열린 주문 전량 취소
  3. 포지션 전량 청산 (시장가)
  4. API 키 비활성화
- [ ] 비상 정지 스크립트 존재 + 테스트 완료
- [ ] 담당자 연락처 + 에스컬레이션 경로 정의
- [ ] 최대 손실 한도 정의 (이 금액 초과 시 무조건 정지)

---

## Go/No-Go 판정

### Paper Trading (T0)
```
Tier 1 + 2 + 3 통과 → GO
```

### Testnet (T1)
```
Tier 1 + 2 + 3 + 4 통과 → GO
```

### Canary — 소액 실전 (T2)
```
Tier 1-6 통과 + Paper 3개월 안정 운영 → GO
초기 자금: 최대 $100 (손실 감내 가능 금액)
```

### Production (T3)
```
전체 Tier 통과 + Canary 1개월 안정 운영 → GO
운용 자금은 총 투자 가능 자산의 5% 이하로 시작
```

---

## 점검 결과 양식

```
══════════════════════════════════════════════════════
  PRE-DEPLOY CHECK REPORT
  대상 티어: [T0/T1/T2/T3]
  점검일: [날짜]
  점검자: [이름]
══════════════════════════════════════════════════════

Tier 1: 코드 품질        [PASS/FAIL]
  - ruff:    0 errors
  - pyright: 0 errors
  - pytest:  XXX passed, 0 failed
  - 시크릿:  0 하드코딩

Tier 2: 백테스트 검증     [PASS/FAIL]
  - 기간: 2022-01 ~ 2025-12 (4년)
  - Sharpe: X.XX, MDD: XX.X%
  - IS/OOS: PASS

Tier 3: 리스크 파라미터   [PASS/FAIL]
  - system_stop_loss: 10%
  - max_leverage_cap: 2.0x
  - circuit_breaker: ACTIVE

...

──────────────────────────────────────────────────────
  판정: [GO / NO-GO]
  사유: [판정 근거]
  미해결 항목: [있으면 목록]
══════════════════════════════════════════════════════
```
