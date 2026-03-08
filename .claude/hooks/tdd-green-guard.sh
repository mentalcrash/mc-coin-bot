#!/bin/bash
# TDD Green Phase Guard — tests/ 파일 접근 차단
# /tdd-green skill 활성화 시에만 실행됨 (skill frontmatter hook)
#
# stdin: JSON with tool_name, tool_input
# exit 0: 허용
# exit 2: 차단 (stderr → Claude에 피드백)

INPUT=$(cat)

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input // empty')

# 1) Read/Edit/Write — file_path 체크
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
if [[ -n "$FILE_PATH" && "$FILE_PATH" == *"tests/"* ]]; then
    echo "BLOCKED: tdd-green phase에서 tests/ 파일 접근 금지 ($TOOL_NAME: $FILE_PATH)" >&2
    exit 2
fi

# 2) Grep — path 체크
GREP_PATH=$(echo "$INPUT" | jq -r '.tool_input.path // empty')
if [[ "$TOOL_NAME" == "Grep" && -n "$GREP_PATH" && "$GREP_PATH" == *"tests/"* ]]; then
    echo "BLOCKED: tdd-green phase에서 tests/ 검색 금지 (Grep: $GREP_PATH)" >&2
    exit 2
fi

# 3) Glob — path 체크
GLOB_PATH=$(echo "$INPUT" | jq -r '.tool_input.path // empty')
if [[ "$TOOL_NAME" == "Glob" && -n "$GLOB_PATH" && "$GLOB_PATH" == *"tests/"* ]]; then
    echo "BLOCKED: tdd-green phase에서 tests/ 검색 금지 (Glob: $GLOB_PATH)" >&2
    exit 2
fi

# 4) Glob — pattern에 tests/ 포함
GLOB_PATTERN=$(echo "$INPUT" | jq -r '.tool_input.pattern // empty')
if [[ "$TOOL_NAME" == "Glob" && "$GLOB_PATTERN" == *"tests/"* ]]; then
    echo "BLOCKED: tdd-green phase에서 tests/ 패턴 검색 금지 (Glob: $GLOB_PATTERN)" >&2
    exit 2
fi

# 5) Bash — cat/head/tail/less 등으로 tests/ 파일 읽기 차단
if [[ "$TOOL_NAME" == "Bash" ]]; then
    COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
    # tests/ 파일을 직접 읽는 명령 차단 (pytest 실행은 허용)
    if [[ "$COMMAND" == *"tests/"* && "$COMMAND" != *"pytest"* && "$COMMAND" != *"uv run pytest"* ]]; then
        echo "BLOCKED: tdd-green phase에서 tests/ 파일 직접 접근 금지 (Bash: $COMMAND)" >&2
        exit 2
    fi
fi

exit 0
