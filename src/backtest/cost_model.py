"""Cost Model re-export for backward compatibility.

NOTE: CostModel은 이제 src/portfolio/cost_model.py에 정의됩니다.
이 모듈은 기존 임포트와의 하위 호환성을 위해 유지됩니다.

Canonical location: src.portfolio.cost_model
"""

# Re-export from canonical location
from src.portfolio.cost_model import CostModel

__all__ = ["CostModel"]
