"""Strategy module for trading strategies.

이 모듈은 트레이딩 전략의 기반 클래스와 타입 정의를 제공합니다.
모든 전략은 BaseStrategy를 상속받아 구현됩니다.

Registry Pattern:
    전략은 @register() 데코레이터로 등록되며, get_strategy()로 조회합니다.
    이를 통해 CLI와 전략 간의 결합도를 제거하여 OCP를 준수합니다.

Example:
    >>> from src.strategy import BaseStrategy, StrategySignals, Direction
    >>> from src.strategy import get_strategy, list_strategies
    >>>
    >>> # 이름으로 전략 조회
    >>> strategy_class = get_strategy("tsmom")
    >>> strategy = strategy_class()
    >>>
    >>> # 등록된 전략 목록
    >>> print(list_strategies())  # ['adaptive-breakout', 'tsmom']
"""

# pyright: reportUnusedImport=false

# 전략 자동 등록 (import 시 @register 데코레이터 실행)
# NOTE: 각 전략 모듈이 임포트될 때 Registry에 등록됨
import src.strategy.ac_regime  # 전략 등록 side effect
import src.strategy.accel_conv  # 전략 등록 side effect
import src.strategy.accel_skew  # 전략 등록 side effect
import src.strategy.accel_vol_trend  # 전략 등록 side effect
import src.strategy.adaptive_fr_carry  # 전략 등록 side effect
import src.strategy.adx_regime  # 전략 등록 side effect
import src.strategy.anchor_mom  # 전략 등록 side effect
import src.strategy.anti_corr_mom  # 전략 등록 side effect
import src.strategy.aroc_mom  # 전략 등록 side effect
import src.strategy.asym_semivar_mr  # 전략 등록 side effect
import src.strategy.asym_vol_resp  # 전략 등록 side effect
import src.strategy.atf_3h  # 전략 등록 side effect
import src.strategy.autocorr_mom  # 전략 등록 side effect
import src.strategy.bb_rsi  # 전략 등록 side effect
import src.strategy.breakout  # 전략 등록 side effect
import src.strategy.btc_lead  # 전략 등록 side effect
import src.strategy.candle_conv_mom  # 전략 등록 side effect
import src.strategy.candle_reject  # 전략 등록 side effect
import src.strategy.cap_wick_rev  # 전략 등록 side effect
import src.strategy.carry_cond_mom  # 전략 등록 side effect
import src.strategy.carry_mom_convergence  # 전략 등록 side effect
import src.strategy.carry_sent  # 전략 등록 side effect
import src.strategy.cascade_mom  # 전략 등록 side effect
import src.strategy.cft_2h  # 전략 등록 side effect
import src.strategy.cgo_mom  # 전략 등록 side effect
import src.strategy.cmf_persist  # 전략 등록 side effect
import src.strategy.complexity_trend  # 전략 등록 side effect
import src.strategy.conviction_trend_composite  # 전략 등록 side effect
import src.strategy.copula_pairs  # 전략 등록 side effect
import src.strategy.ctrend  # 전략 등록 side effect
import src.strategy.ctrend_x  # 전략 등록 side effect
import src.strategy.dex_mom  # 전략 등록 side effect
import src.strategy.dir_vol_trend  # 전략 등록 side effect
import src.strategy.disp_breakout  # 전략 등록 side effect
import src.strategy.dist_mom  # 전략 등록 side effect
import src.strategy.donchian  # 전략 등록 side effect
import src.strategy.donchian_ensemble  # 전략 등록 side effect
import src.strategy.dual_vol  # 전략 등록 side effect
import src.strategy.dvr_mom  # 전략 등록 side effect
import src.strategy.eff_brk  # 전략 등록 side effect
import src.strategy.ema_cross_base  # 전략 등록 side effect
import src.strategy.ema_multi_cross  # 전략 등록 side effect
import src.strategy.ema_ribbon_mom  # 전략 등록 side effect
import src.strategy.enhanced_tsmom  # 전략 등록 side effect
import src.strategy.ens_regime_dual  # 전략 등록 side effect
import src.strategy.ensemble  # 전략 등록 side effect
import src.strategy.entropy_carry_mom  # 전략 등록 side effect
import src.strategy.entropy_switch  # 전략 등록 side effect
import src.strategy.ex_flow_mom  # 전략 등록 side effect
import src.strategy.fear_divergence  # 전략 등록 side effect
import src.strategy.fg_asym_mom  # 전략 등록 side effect
import src.strategy.fg_delta  # 전략 등록 side effect
import src.strategy.fg_ema_cycle  # 전략 등록 side effect
import src.strategy.fg_persist_break  # 전략 등록 side effect
import src.strategy.flow_imbalance  # 전략 등록 side effect
import src.strategy.fr_carry_vol  # 전략 등록 side effect
import src.strategy.fr_cond_mom  # 전략 등록 side effect
import src.strategy.fr_pred  # 전략 등록 side effect
import src.strategy.fr_press_trend  # 전략 등록 side effect
import src.strategy.fr_quality_mom  # 전략 등록 side effect
import src.strategy.fr_stab_conf  # 전략 등록 side effect
import src.strategy.fractal_mom  # 전략 등록 side effect
import src.strategy.fund_div_mom  # 전략 등록 side effect
import src.strategy.funding_carry  # 전략 등록 side effect
import src.strategy.gbtrend  # 전략 등록 side effect
import src.strategy.gk_breakout  # 전략 등록 side effect
import src.strategy.gk_range_mom  # 전략 등록 side effect
import src.strategy.har_vol  # 전략 등록 side effect
import src.strategy.hd_mom_rev  # 전략 등록 side effect
import src.strategy.hmm_regime  # 전략 등록 side effect
import src.strategy.hour_season  # 전략 등록 side effect
import src.strategy.hurst_adaptive  # 전략 등록 side effect
import src.strategy.jump_drift_mom  # 전략 등록 side effect
import src.strategy.kalman_trend  # 전략 등록 side effect
import src.strategy.kama  # 전략 등록 side effect
import src.strategy.kelt_eff_trend  # 전략 등록 side effect
import src.strategy.liq_cascade_rev  # 전략 등록 side effect
import src.strategy.liq_conf_trend  # 전략 등록 side effect
import src.strategy.liq_momentum  # 전략 등록 side effect
import src.strategy.macro_liq_trend  # 전략 등록 side effect
import src.strategy.max_min  # 전략 등록 side effect
import src.strategy.mcr_mom  # 전략 등록 side effect
import src.strategy.mh_roc  # 전략 등록 side effect
import src.strategy.mhm  # 전략 등록 side effect
import src.strategy.ml_deriv_regime  # 전략 등록 side effect
import src.strategy.mom_accel  # 전략 등록 side effect
import src.strategy.mom_mr_blend  # 전략 등록 side effect
import src.strategy.ms_vol_ratio  # 전략 등록 side effect
import src.strategy.mtf_macd  # 전략 등록 side effect
import src.strategy.multi_domain_score  # 전략 등록 side effect
import src.strategy.multi_factor  # 전략 등록 side effect
import src.strategy.nvt_cycle  # 전략 등록 side effect
import src.strategy.obv_accel  # 전략 등록 side effect
import src.strategy.oi_diverge  # 전략 등록 side effect
import src.strategy.onchain_accum  # 전략 등록 side effect
import src.strategy.onchain_bias_4h  # 전략 등록 side effect
import src.strategy.ou_meanrev  # 전략 등록 side effect
import src.strategy.perm_entropy_mom  # 전략 등록 side effect
import src.strategy.qd_mom  # 전략 등록 side effect
import src.strategy.range_squeeze  # 전략 등록 side effect
import src.strategy.regime_adaptive_mom  # 전략 등록 side effect
import src.strategy.regime_mf_mr  # 전략 등록 side effect
import src.strategy.regime_tsmom  # 전략 등록 side effect
import src.strategy.residual_mom  # 전략 등록 side effect
import src.strategy.ret_persist  # 전략 등록 side effect
import src.strategy.rp_vol_regime  # 전략 등록 side effect
import src.strategy.rs_btc  # 전략 등록 side effect
import src.strategy.rv_jump_cont  # 전략 등록 side effect
import src.strategy.scaled_mom  # 전략 등록 side effect
import src.strategy.session_breakout  # 전략 등록 side effect
import src.strategy.skew_mom  # 전략 등록 side effect
import src.strategy.stab_comp  # 전략 등록 side effect
import src.strategy.stab_mom_trend  # 전략 등록 side effect
import src.strategy.stoch_mom  # 전략 등록 side effect
import src.strategy.trend_eff_score  # 전략 등록 side effect
import src.strategy.trend_persist  # 전략 등록 side effect
import src.strategy.trend_quality_mom  # 전략 등록 side effect
import src.strategy.tsmom  # 전략 등록 side effect
import src.strategy.ttm_squeeze  # 전략 등록 side effect
import src.strategy.up_vol_mom  # 전략 등록 side effect
import src.strategy.vardecomp_mom  # 전략 등록 side effect
import src.strategy.vmacd_mom  # 전략 등록 side effect
import src.strategy.vmsm  # 전략 등록 side effect
import src.strategy.vol_adaptive  # 전략 등록 side effect
import src.strategy.vol_asym_trend  # 전략 등록 side effect
import src.strategy.vol_climax  # 전략 등록 side effect
import src.strategy.vol_compress_brk  # 전략 등록 side effect
import src.strategy.vol_confirm_mom  # 전략 등록 side effect
import src.strategy.vol_impulse_mom  # 전략 등록 side effect
import src.strategy.vol_ratio_trend  # 전략 등록 side effect
import src.strategy.vol_regime  # 전략 등록 side effect
import src.strategy.vol_squeeze_brk  # 전략 등록 side effect
import src.strategy.vol_squeeze_deriv  # 전략 등록 side effect
import src.strategy.vol_struct_ml  # 전략 등록 side effect
import src.strategy.vol_structure  # 전략 등록 side effect
import src.strategy.vol_surface_mom  # 전략 등록 side effect
import src.strategy.vol_term_ml  # 전략 등록 side effect
import src.strategy.vov_mom  # 전략 등록 side effect
import src.strategy.vpin_flow  # 전략 등록 side effect
import src.strategy.vr_regime  # 전략 등록 side effect
import src.strategy.vrp_trend  # 전략 등록 side effect
import src.strategy.vw_tsmom  # 전략 등록 side effect
import src.strategy.vwap_disposition  # 전략 등록 side effect
import src.strategy.vwap_trend_cross  # 전략 등록 side effect
import src.strategy.xsmom  # 전략 등록 side effect
from src.strategy.base import BaseStrategy
from src.strategy.registry import (
    get_strategy,
    is_registered,
    list_strategies,
    register,
)
from src.strategy.types import (
    DEFAULT_OHLCV_COLUMNS,
    Direction,
    SignalType,
    StrategySignals,
)

__all__ = [
    "DEFAULT_OHLCV_COLUMNS",
    # Base & Types
    "BaseStrategy",
    "Direction",
    "SignalType",
    "StrategySignals",
    # Registry
    "get_strategy",
    "is_registered",
    "list_strategies",
    "register",
]
