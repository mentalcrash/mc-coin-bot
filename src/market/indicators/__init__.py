"""Common technical indicator functions.

전략 모듈 전반에서 공유되는 순수 지표 함수 라이브러리.
모든 함수는 stateless, vectorized이며 ``pd.Series`` in → ``pd.Series`` out.

Usage::

    from src.market.indicators import atr, rsi, realized_volatility, log_returns
"""

from src.market.indicators.channels import (
    bollinger_bands as bollinger_bands,
    donchian_channel as donchian_channel,
    keltner_channels as keltner_channels,
)
from src.market.indicators.composite import (
    bb_position as bb_position,
    count_consecutive as count_consecutive,
    drawdown as drawdown,
    ema_cross as ema_cross,
    fractal_dimension as fractal_dimension,
    hurst_exponent as hurst_exponent,
    mean_reversion_score as mean_reversion_score,
    price_acceleration as price_acceleration,
    rolling_zscore as rolling_zscore,
    rsi_divergence as rsi_divergence,
    sma_cross as sma_cross,
    squeeze_detect as squeeze_detect,
    trend_strength as trend_strength,
)
from src.market.indicators.derivatives import (
    basis_spread as basis_spread,
    funding_rate_ma as funding_rate_ma,
    funding_zscore as funding_zscore,
    liquidation_intensity as liquidation_intensity,
    ls_ratio_zscore as ls_ratio_zscore,
    oi_momentum as oi_momentum,
    oi_price_divergence as oi_price_divergence,
)
from src.market.indicators.oscillators import (
    cci as cci,
    macd as macd,
    momentum as momentum,
    roc as roc,
    rsi as rsi,
    stochastic as stochastic,
    williams_r as williams_r,
)
from src.market.indicators.returns import (
    log_returns as log_returns,
    rolling_return as rolling_return,
    simple_returns as simple_returns,
)
from src.market.indicators.trend import (
    adx as adx,
    atr as atr,
    efficiency_ratio as efficiency_ratio,
    ema as ema,
    kama as kama,
    sma as sma,
)
from src.market.indicators.volatility import (
    garman_klass_volatility as garman_klass_volatility,
    parkinson_volatility as parkinson_volatility,
    realized_volatility as realized_volatility,
    vol_percentile_rank as vol_percentile_rank,
    vol_regime as vol_regime,
    volatility_of_volatility as volatility_of_volatility,
    volatility_scalar as volatility_scalar,
    yang_zhang_volatility as yang_zhang_volatility,
)
from src.market.indicators.volume import (
    chaikin_money_flow as chaikin_money_flow,
    obv as obv,
    volume_macd as volume_macd,
    volume_weighted_returns as volume_weighted_returns,
)

__all__ = [
    "adx",
    "atr",
    "basis_spread",
    "bb_position",
    "bollinger_bands",
    "cci",
    "chaikin_money_flow",
    "count_consecutive",
    "donchian_channel",
    "drawdown",
    "efficiency_ratio",
    "ema",
    "ema_cross",
    "fractal_dimension",
    "funding_rate_ma",
    "funding_zscore",
    "garman_klass_volatility",
    "hurst_exponent",
    "kama",
    "keltner_channels",
    "liquidation_intensity",
    "log_returns",
    "ls_ratio_zscore",
    "macd",
    "mean_reversion_score",
    "momentum",
    "obv",
    "oi_momentum",
    "oi_price_divergence",
    "parkinson_volatility",
    "price_acceleration",
    "realized_volatility",
    "roc",
    "rolling_return",
    "rolling_zscore",
    "rsi",
    "rsi_divergence",
    "simple_returns",
    "sma",
    "sma_cross",
    "squeeze_detect",
    "stochastic",
    "trend_strength",
    "vol_percentile_rank",
    "vol_regime",
    "volatility_of_volatility",
    "volatility_scalar",
    "volume_macd",
    "volume_weighted_returns",
    "williams_r",
    "yang_zhang_volatility",
]
