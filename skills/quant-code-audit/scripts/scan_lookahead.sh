#!/bin/bash
# scan_lookahead.sh — 퀀트 코드 자동 탐지 스크립트
# 사용법: bash scan_lookahead.sh <project_root>
#
# Look-ahead bias, 데이터 누수, 리스크 관리 누락 등을 grep으로 자동 탐지.
# 결과는 의심 항목이며, 반드시 수동 검토가 필요.

PROJECT_ROOT="${1:-.}"

YELLOW='\033[1;33m'
RED='\033[1;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

found_issues=0

print_header() {
    echo ""
    echo "── $1 ──"
    echo ""
}

run_scan() {
    local label="$1"
    local severity="$2"
    local pattern="$3"

    local results
    results=$(grep -rn --include="*.py" -E "$pattern" "$PROJECT_ROOT" 2>/dev/null | grep -v "__pycache__" || true)

    if [ -n "$results" ]; then
        local count
        count=$(echo "$results" | wc -l | tr -d ' ')
        found_issues=$((found_issues + count))

        case "$severity" in
            CRITICAL) echo -e "${RED}🔴 [${severity}] ${label} (${count}건)${NC}" ;;
            HIGH)     echo -e "${YELLOW}🟠 [${severity}] ${label} (${count}건)${NC}" ;;
            MEDIUM)   echo -e "${CYAN}🟡 [${severity}] ${label} (${count}건)${NC}" ;;
            *)        echo -e "${GREEN}🔵 [${severity}] ${label} (${count}건)${NC}" ;;
        esac

        echo "$results" | head -10
        if [ "$count" -gt 10 ]; then
            echo "  ... 외 $((count - 10))건"
        fi
        echo ""
    fi
}

echo "══════════════════════════════════════════════════════"
echo "  QUANT CODE AUTO-SCAN"
echo "  대상: ${PROJECT_ROOT}"
echo "  날짜: $(date '+%Y-%m-%d %H:%M')"
echo "══════════════════════════════════════════════════════"

# ═══════════════════════════════════════
# 1단계: LOOK-AHEAD BIAS
# ═══════════════════════════════════════
print_header "1단계: Look-Ahead Bias 탐지"

run_scan "shift(-N): 미래 값 참조" \
    "CRITICAL" \
    'shift\(-[0-9]'

run_scan "pct_change(-N): 미래 수익률" \
    "CRITICAL" \
    'pct_change\(-[0-9]'

run_scan "iloc[i+N]: 미래 행 접근" \
    "CRITICAL" \
    'iloc\[.*\+.*[0-9]'

# ═══════════════════════════════════════
# 2단계: DATA LEAKAGE
# ═══════════════════════════════════════
print_header "2단계: Data Leakage 탐지"

run_scan "scaler.fit(): 미래 통계 유입 가능성" \
    "HIGH" \
    '\.fit\(|\.fit_transform\('

run_scan ".mean()/.std() (전체 데이터 통계 사용 가능성)" \
    "HIGH" \
    '\.(mean|std)\(\)'

run_scan "train_test_split (시계열 무작위 분할 위험)" \
    "HIGH" \
    'train_test_split'

# ═══════════════════════════════════════
# 3단계: 실행 현실성
# ═══════════════════════════════════════
print_header "3단계: 실행 현실성 탐지"

run_scan "slippage/commission/fee = 0 설정" \
    "HIGH" \
    '(slippage|commission|fee) *= *0'

run_scan "시그널 봉 close에서 체결" \
    "MEDIUM" \
    'fill_price.*close|entry_price.*=.*close'

# ═══════════════════════════════════════
# 4단계: 리스크 관리
# ═══════════════════════════════════════
print_header "4단계: 리스크 관리 탐지"

run_scan "주석 처리된 stop_loss / 리스크 코드" \
    "HIGH" \
    '^[[:space:]]*#.*(stop_loss|max_leverage|risk_limit)'

run_scan "stop_loss 설정 존재 확인" \
    "INFO" \
    'stop_loss|stop_price|take_profit'

run_scan "max_leverage / max_position 설정 확인" \
    "INFO" \
    'max_leverage|max_position|position_limit'

# ═══════════════════════════════════════
# 5단계: 코드 품질
# ═══════════════════════════════════════
print_header "5단계: 코드 품질 탐지"

run_scan "API 키 하드코딩 (보안 위험)" \
    "CRITICAL" \
    '(api_key|api_secret|password|secret_key) *= *[\"'"'"']'

run_scan "bare except: (에러 무시)" \
    "MEDIUM" \
    'except *:'

run_scan "0 나눗셈 위험 변수" \
    "MEDIUM" \
    '/ *(realized_vol|atr|volatility|std_dev)'

# ═══════════════════════════════════════
# 결과 요약
# ═══════════════════════════════════════
echo "══════════════════════════════════════════════════════"
if [ "$found_issues" -gt 0 ]; then
    echo -e "${YELLOW}  총 ${found_issues}건의 의심 항목 탐지${NC}"
    echo "  ⚠️  자동 탐지 결과이며, 수동 검토가 반드시 필요합니다."
    echo "  오탐(false positive)이 포함될 수 있습니다."
else
    echo -e "${GREEN}  자동 탐지 항목 없음${NC}"
    echo "  ✅ 하지만 자동 탐지로 잡히지 않는 논리 오류가 있을 수 있습니다."
fi
echo "══════════════════════════════════════════════════════"

exit 0
