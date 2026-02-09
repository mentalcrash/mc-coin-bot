"""Docker HEALTHCHECK script for MC Coin Bot.

PID 1 프로세스 생존 여부로 컨테이너 상태를 판별합니다.
HTTP 서버 없이 동작하는 트레이딩 봇에 적합한 방식입니다.

Exit codes:
    0: healthy
    1: unhealthy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# PID 1 프로세스 생존 확인
_PID_1 = 1


def check_pid1_alive() -> bool:
    """PID 1 프로세스가 살아있는지 확인."""
    try:
        os.kill(_PID_1, 0)
    except OSError:
        return False
    return True


def check_log_freshness(log_dir: str = "logs", max_stale_seconds: int = 600) -> bool:
    """로그 디렉토리에 최근 파일이 있는지 확인 (선택).

    Args:
        log_dir: 로그 디렉토리 경로
        max_stale_seconds: 이 시간(초) 이상 로그가 없으면 stale

    Returns:
        로그가 신선하면 True, stale이면 False
    """
    import time

    log_path = Path(log_dir)
    if not log_path.exists():
        return True  # 로그 디렉토리 없으면 패스 (아직 생성 전일 수 있음)

    log_files = list(log_path.glob("*.json")) + list(log_path.glob("*.log"))
    if not log_files:
        return True  # 로그 파일 없으면 패스

    newest_mtime = max(f.stat().st_mtime for f in log_files)
    age = time.time() - newest_mtime
    return age < max_stale_seconds


def main() -> None:
    """Health check 실행."""
    if not check_pid1_alive():
        print("UNHEALTHY: PID 1 process not found")
        sys.exit(1)

    if not check_log_freshness():
        print("UNHEALTHY: logs are stale (>600s)")
        sys.exit(1)

    print("HEALTHY")
    sys.exit(0)


if __name__ == "__main__":
    main()
