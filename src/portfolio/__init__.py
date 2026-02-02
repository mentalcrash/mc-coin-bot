"""Portfolio Management Module.

이 모듈은 전략-독립적인 포트폴리오 집행 설정을 제공합니다.
백테스팅과 실전 트레이딩에서 공유할 수 있는 설정 구조입니다.

Rules Applied:
    - #01 Project Structure: src/portfolio/ 모듈
    - #11 Pydantic Modeling: frozen=True, validation
"""

from src.portfolio.config import PortfolioManagerConfig

__all__ = ["PortfolioManagerConfig"]
