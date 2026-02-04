"""Portfolio Management Module.

이 모듈은 포트폴리오 도메인 객체와 집행 설정을 제공합니다.
백테스팅과 실전 트레이딩에서 공유할 수 있는 구조입니다.

Exports:
    - Portfolio: 포트폴리오 도메인 객체 (initial_capital + config)
    - PortfolioManagerConfig: 전략-독립적인 집행 설정
    - CostModel: 거래 비용 모델

Rules Applied:
    - #01 Project Structure: src/portfolio/ 모듈
    - #11 Pydantic Modeling: frozen=True, validation
"""

from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.portfolio.portfolio import Portfolio

__all__ = ["CostModel", "Portfolio", "PortfolioManagerConfig"]
