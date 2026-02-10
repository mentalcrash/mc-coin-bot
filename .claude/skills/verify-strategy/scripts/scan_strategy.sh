#!/bin/bash
# scan_strategy.sh — Gate 0B 전략 코드 자동 프리스캔
#
# 사용법: bash .claude/skills/verify-strategy/scripts/scan_strategy.sh <strategy_dir>
# 예시:   bash .claude/skills/verify-strategy/scripts/scan_strategy.sh src/strategy/range_squeeze
#
# C1-C7 관련 의심 패턴을 자동 탐지. 결과는 수동 검토의 시작점.

set -euo pipefail

STRATEGY_DIR="${1:?Usage: scan_strategy.sh <strategy_directory>}"

if [ ! -d "$STRATEGY_DIR" ]; then
    echo "ERROR: Directory not found: $STRATEGY_DIR"
    exit 1
fi

RED='\033[1;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

found_critical=0
found_warning=0

header() {
    echo ""
    echo -e "${CYAN}── $1 ──${NC}"
    echo ""
}

scan() {
    local id="$1"
    local label="$2"
    local severity="$3"
    local pattern="$4"
    local target="${5:-$STRATEGY_DIR}"

    local results
    results=$(grep -rn --include="*.py" -E "$pattern" "$target" 2>/dev/null | grep -v "__pycache__\|# noqa\|# type:" || true)

    if [ -n "$results" ]; then
        local count
        count=$(echo "$results" | wc -l | tr -d ' ')

        case "$severity" in
            CRITICAL)
                echo -e "${RED}[$id] CRITICAL: ${label} (${count}건)${NC}"
                found_critical=$((found_critical + count))
                ;;
            HIGH)
                echo -e "${YELLOW}[$id] HIGH: ${label} (${count}건)${NC}"
                found_critical=$((found_critical + count))
                ;;
            WARNING)
                echo -e "${CYAN}[$id] WARNING: ${label} (${count}건)${NC}"
                found_warning=$((found_warning + count))
                ;;
        esac

        echo "$results" | head -5
        if [ "$count" -gt 5 ]; then
            echo "  ... (+$((count - 5))건)"
        fi
        echo ""
    fi
}

# ================================================================
echo "============================================================"
echo "  GATE 0B: STRATEGY AUTO-SCAN"
echo "  Target: ${STRATEGY_DIR}"
echo "  Date:   $(date '+%Y-%m-%d %H:%M')"
echo "============================================================"

# ================================================================
# 구조 확인
# ================================================================
header "Structure Check"

for f in config.py preprocessor.py signal.py strategy.py __init__.py; do
    if [ -f "$STRATEGY_DIR/$f" ]; then
        echo -e "  ${GREEN}[OK]${NC} $f"
    else
        echo -e "  ${YELLOW}[MISSING]${NC} $f"
    fi
done
echo ""

# @register 데코레이터 확인
if grep -q '@register' "$STRATEGY_DIR/strategy.py" 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} @register decorator found"
else
    echo -e "  ${YELLOW}[MISSING]${NC} @register decorator not found"
fi

# warmup_periods() 확인
if grep -q 'warmup_periods' "$STRATEGY_DIR/config.py" 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} warmup_periods() defined"
else
    echo -e "  ${YELLOW}[MISSING]${NC} warmup_periods() not defined"
fi

# ================================================================
# C1: Look-Ahead Bias
# ================================================================
header "C1: Look-Ahead Bias"

scan "C1-a" "shift(-N): 미래 값 참조" \
    "CRITICAL" \
    'shift\(-[0-9]'

scan "C1-b" "pct_change(-N): 미래 수익률" \
    "CRITICAL" \
    'pct_change\(-[0-9]'

scan "C1-c" "iloc[i+N]: 미래 행 접근" \
    "CRITICAL" \
    'iloc\[.*\+.*[1-9]'

scan "C1-d" ".min()/.max()/.mean()/.std() (rolling/expanding 없이)" \
    "HIGH" \
    '\.(min|max|mean|std)\(\)' \
    "$STRATEGY_DIR/preprocessor.py"

# ================================================================
# C2: Data Leakage
# ================================================================
header "C2: Data Leakage"

scan "C2-a" "scaler.fit() / fit_transform()" \
    "HIGH" \
    '\.fit\(|\.fit_transform\('

scan "C2-b" "train_test_split (무작위 분할)" \
    "HIGH" \
    'train_test_split'

# ================================================================
# C4: Signal Vectorization
# ================================================================
header "C4: Signal Vectorization"

scan "C4-a" "for loop over DataFrame" \
    "HIGH" \
    'for.*range.*len|iterrows|itertuples'

scan "C4-b" "append in loop (non-vectorized)" \
    "WARNING" \
    '\.append\(' \
    "$STRATEGY_DIR/signal.py"

# ================================================================
# C5: Position Sizing
# ================================================================
header "C5: Position Sizing"

scan "C5-a" "vol division (0 나눗셈 위험)" \
    "WARNING" \
    '/ *(realized_vol|vol|volatility|std_dev)'

# min_volatility 사용 확인
if grep -q 'min_volatility\|min_vol\|np\.maximum\|np\.clip' "$STRATEGY_DIR/preprocessor.py" 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} min_volatility / clip protection found in preprocessor.py"
else
    echo -e "  ${YELLOW}[CHECK]${NC} No min_volatility protection found in preprocessor.py — verify manually"
fi
echo ""

# annualization factor 확인
if grep -rq '252' "$STRATEGY_DIR/" 2>/dev/null; then
    echo -e "  ${YELLOW}[CHECK]${NC} Found '252' (stock annualization) — verify if crypto should use 365"
    grep -rn '252' "$STRATEGY_DIR/" 2>/dev/null | head -3
fi

# ================================================================
# C6: Cost Model
# ================================================================
header "C6: Cost Model"

scan "C6-a" "fee/slippage/commission = 0" \
    "HIGH" \
    '(fee|slippage|commission) *= *0'

scan "C6-b" "자체 PnL 계산 (전략은 시그널만)" \
    "WARNING" \
    'pnl|profit|equity.*=.*cash'

# ================================================================
# C7: Entry/Exit Logic
# ================================================================
header "C7: Entry/Exit Logic"

# ShortMode 처리 확인
if grep -q 'ShortMode\|short_mode' "$STRATEGY_DIR/signal.py" 2>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} ShortMode handling found in signal.py"
    # 3가지 모드 모두 처리되는지 확인
    for mode in DISABLED HEDGE_ONLY FULL; do
        if grep -q "$mode" "$STRATEGY_DIR/signal.py" 2>/dev/null; then
            echo -e "    ${GREEN}[OK]${NC} $mode branch"
        else
            echo -e "    ${YELLOW}[CHECK]${NC} $mode branch not found — verify"
        fi
    done
else
    echo -e "  ${YELLOW}[CHECK]${NC} ShortMode handling not found in signal.py"
fi
echo ""

# ================================================================
# 보안 (빠른 체크)
# ================================================================
header "Security"

scan "SEC" "API 키 하드코딩" \
    "CRITICAL" \
    "(api_key|api_secret|password|secret_key|token) *= *[\"']"

# ================================================================
# 요약
# ================================================================
echo "============================================================"
if [ "$found_critical" -gt 0 ]; then
    echo -e "  ${RED}Critical/High 의심: ${found_critical}건 — 수동 검증 필수${NC}"
fi
if [ "$found_warning" -gt 0 ]; then
    echo -e "  ${YELLOW}Warning 의심: ${found_warning}건${NC}"
fi
if [ "$found_critical" -eq 0 ] && [ "$found_warning" -eq 0 ]; then
    echo -e "  ${GREEN}자동 탐지 항목 없음${NC}"
fi
echo ""
echo "  이 결과는 자동 프리스캔이며, 오탐(false positive) 포함 가능."
echo "  반드시 SKILL.md의 C1-C7 수동 검증을 수행하세요."
echo "============================================================"

exit 0
