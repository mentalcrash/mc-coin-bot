"""AssetSelector — Pod 내 에셋 선별 FSM.

성과 기반 복합 점수로 에셋의 참여 여부(multiplier 0.0~1.0)를 관리합니다.

핵심 원칙:
    - WHO(참여 여부) vs HOW MUCH(비중) 분리
    - Hysteresis + Confirmation + Cooldown으로 whipsaw 방지
    - Gradual ramp (즉시 0/1 전환 금지)
    - min_active_assets 안전망

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - #23 Exception Handling: 경계 조건 방어
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.orchestrator.models import AssetLifecycleState

if TYPE_CHECKING:
    from src.orchestrator.config import AssetSelectorConfig

# ── Constants ─────────────────────────────────────────────────────

_NEUTRAL_SCORE = 0.5
_MIN_DATA_BARS = 10
_MIN_SHARPE_SAMPLES = 2
_EPSILON = 1e-12


# ── AssetState (per-symbol internal state) ────────────────────────


class _AssetState:
    """단일 에셋의 FSM 상태."""

    __slots__ = (
        "confirmation_count",
        "cooldown_bars",
        "cooldown_cycles",
        "multiplier",
        "permanently_excluded",
        "ramp_position",
        "score",
        "state",
    )

    def __init__(self) -> None:
        self.state = AssetLifecycleState.ACTIVE
        self.multiplier: float = 1.0
        self.score: float = _NEUTRAL_SCORE
        self.confirmation_count: int = 0
        self.cooldown_bars: int = 0
        self.cooldown_cycles: int = 0  # cooldown 진입 횟수
        self.ramp_position: int = 0  # 0=ramp 미사용, 1~ramp_steps=ramp 진행 중
        self.permanently_excluded: bool = False


# ── AssetSelector ────────────────────────────────────────────────


class AssetSelector:
    """Pod 내 에셋 선별 FSM.

    각 에셋의 성과를 cross-sectional ranking으로 평가하고,
    성과 미달 에셋을 점진적으로 제외/재진입합니다.

    Args:
        config: AssetSelectorConfig
        symbols: Pod 내 거래 심볼 목록
    """

    def __init__(
        self,
        config: AssetSelectorConfig,
        symbols: tuple[str, ...],
    ) -> None:
        self._config = config
        self._symbols = symbols
        self._states: dict[str, _AssetState] = {s: _AssetState() for s in symbols}

    # ── Properties ────────────────────────────────────────────────

    @property
    def multipliers(self) -> dict[str, float]:
        """에셋별 participation multiplier (0.0~1.0)."""
        return {s: st.multiplier for s, st in self._states.items()}

    @property
    def active_symbols(self) -> list[str]:
        """multiplier > 0인 에셋 목록."""
        return [s for s, st in self._states.items() if st.multiplier > _EPSILON]

    @property
    def all_excluded(self) -> bool:
        """모든 에셋이 제외되었는지 여부."""
        return all(st.multiplier < _EPSILON for st in self._states.values())

    @property
    def asset_states(self) -> dict[str, AssetLifecycleState]:
        """에셋별 FSM 상태."""
        return {s: st.state for s, st in self._states.items()}

    @property
    def scores(self) -> dict[str, float]:
        """에셋별 최근 점수."""
        return {s: st.score for s, st in self._states.items()}

    # ── Core ──────────────────────────────────────────────────────

    def on_bar(
        self,
        returns: dict[str, list[float]],
        close_prices: dict[str, float],
    ) -> None:
        """매 bar마다 호출: 점수 계산 → FSM 전이.

        Args:
            returns: 에셋별 수익률 히스토리 {symbol: [r1, r2, ...]}
            close_prices: 에셋별 최신 close price
        """
        if not self._config.enabled:
            return

        # 1. 복합 점수 계산
        scores = self._compute_scores(returns)
        for symbol, score in scores.items():
            if symbol in self._states:
                self._states[symbol].score = score

        # 2. Hard exclusion 체크
        self._check_hard_exclusion(returns)

        # 2b. Absolute threshold 체크 (cross-sectional과 독립)
        self._check_absolute_thresholds(returns)

        # 3. FSM 전이
        for symbol in self._symbols:
            if symbol not in self._states:
                continue
            self._transition(symbol)

    def _compute_scores(
        self,
        returns: dict[str, list[float]],
    ) -> dict[str, float]:
        """Cross-sectional ranking 기반 복합 점수 계산.

        score = w_sharpe * sharpe_rank + w_return * return_rank + w_dd * (1 - dd_rank)
        """
        cfg = self._config
        sharpe_vals: dict[str, float] = {}
        return_vals: dict[str, float] = {}
        dd_vals: dict[str, float] = {}

        for symbol in self._symbols:
            rets = returns.get(symbol, [])
            if len(rets) < _MIN_DATA_BARS:
                sharpe_vals[symbol] = 0.0
                return_vals[symbol] = 0.0
                dd_vals[symbol] = 0.0
                continue

            sharpe_vals[symbol] = self._compute_sharpe(
                rets[-cfg.sharpe_lookback:]
            )
            return_vals[symbol] = self._compute_cumulative_return(
                rets[-cfg.return_lookback:]
            )
            dd_vals[symbol] = self._compute_max_drawdown(
                rets[-cfg.sharpe_lookback:]
            )

        # Cross-sectional rank (0~1)
        sharpe_ranks = _cross_sectional_rank(sharpe_vals)
        return_ranks = _cross_sectional_rank(return_vals)
        dd_ranks = _cross_sectional_rank(dd_vals)

        scores: dict[str, float] = {}
        for symbol in self._symbols:
            rets = returns.get(symbol, [])
            if len(rets) < _MIN_DATA_BARS:
                scores[symbol] = _NEUTRAL_SCORE
            else:
                scores[symbol] = (
                    cfg.sharpe_weight * sharpe_ranks.get(symbol, 0.5)
                    + cfg.return_weight * return_ranks.get(symbol, 0.5)
                    + cfg.drawdown_weight * (1.0 - dd_ranks.get(symbol, 0.5))
                )

        return scores

    def _check_hard_exclusion(
        self,
        returns: dict[str, list[float]],
    ) -> None:
        """Hard exclusion: Sharpe < threshold AND DD > threshold → 즉시 COOLDOWN."""
        cfg = self._config
        for symbol in self._symbols:
            st = self._states.get(symbol)
            if st is None or st.state == AssetLifecycleState.COOLDOWN:
                continue

            rets = returns.get(symbol, [])
            if len(rets) < _MIN_DATA_BARS:
                continue

            sharpe = self._compute_sharpe(rets[-cfg.sharpe_lookback:])
            dd = self._compute_max_drawdown(rets[-cfg.sharpe_lookback:])

            if (
                sharpe < cfg.hard_exclude_sharpe
                and dd > cfg.hard_exclude_drawdown
                and self._can_exclude(symbol)
            ):
                    st.state = AssetLifecycleState.COOLDOWN
                    st.multiplier = 0.0
                    st.cooldown_bars = 0
                    st.confirmation_count = 0
                    st.ramp_position = 0
                    st.cooldown_cycles += 1
                    self._check_permanent_exclusion(symbol, st)
                    logger.info(
                        "AssetSelector: {} hard excluded (Sharpe={:.2f}, DD={:.1%}, cycle {})",
                        symbol,
                        sharpe,
                        dd,
                        st.cooldown_cycles,
                    )

    def _transition(self, symbol: str) -> None:
        """단일 에셋의 FSM 전이를 수행합니다."""
        st = self._states[symbol]
        if st.permanently_excluded:
            return  # 영구 제외 에셋은 전이 불가
        cfg = self._config

        if st.state == AssetLifecycleState.ACTIVE:
            self._transition_active(symbol, st, cfg)
        elif st.state == AssetLifecycleState.UNDERPERFORMING:
            self._transition_underperforming(symbol, st, cfg)
        elif st.state == AssetLifecycleState.COOLDOWN:
            self._transition_cooldown(symbol, st, cfg)
        elif st.state == AssetLifecycleState.RE_ENTRY:
            self._transition_re_entry(symbol, st, cfg)

    def _transition_active(
        self,
        symbol: str,
        st: _AssetState,
        cfg: AssetSelectorConfig,
    ) -> None:
        """ACTIVE → UNDERPERFORMING (confirmation 충족 시)."""
        if st.score < cfg.exclude_score_threshold:
            st.confirmation_count += 1
            if st.confirmation_count >= cfg.exclude_confirmation_bars:
                if self._can_exclude(symbol):
                    st.state = AssetLifecycleState.UNDERPERFORMING
                    st.ramp_position = 0
                    st.confirmation_count = 0
                    self._step_ramp_down(st, cfg)
                    logger.info("AssetSelector: {} → UNDERPERFORMING", symbol)
                else:
                    st.confirmation_count = 0
        else:
            st.confirmation_count = 0

    def _transition_underperforming(
        self,
        symbol: str,
        st: _AssetState,
        cfg: AssetSelectorConfig,
    ) -> None:
        """UNDERPERFORMING → COOLDOWN (ramp 완료) 또는 RE_ENTRY (회복)."""
        # 회복 체크: score 상승 시 ramp 도중 RE_ENTRY
        if st.score > cfg.include_score_threshold:
            st.confirmation_count += 1
            if st.confirmation_count >= cfg.include_confirmation_bars:
                st.state = AssetLifecycleState.RE_ENTRY
                st.confirmation_count = 0
                self._step_ramp_up(st, cfg)
                logger.info("AssetSelector: {} UNDERPERFORMING → RE_ENTRY", symbol)
                return
        else:
            st.confirmation_count = 0

        # Ramp down 진행
        self._step_ramp_down(st, cfg)

        # Ramp 완료 (multiplier == 0) → COOLDOWN
        if st.multiplier < _EPSILON:
            st.state = AssetLifecycleState.COOLDOWN
            st.multiplier = 0.0
            st.cooldown_bars = 0
            st.confirmation_count = 0
            st.cooldown_cycles += 1
            self._check_permanent_exclusion(symbol, st)
            logger.info("AssetSelector: {} → COOLDOWN (cycle {})", symbol, st.cooldown_cycles)

    def _transition_cooldown(
        self,
        symbol: str,
        st: _AssetState,
        cfg: AssetSelectorConfig,
    ) -> None:
        """COOLDOWN → RE_ENTRY (cooldown 경과 + score 충족)."""
        st.cooldown_bars += 1

        if st.cooldown_bars < cfg.min_exclusion_bars:
            return  # 아직 cooldown 중

        if st.score > cfg.include_score_threshold:
            st.confirmation_count += 1
            if st.confirmation_count >= cfg.include_confirmation_bars:
                st.state = AssetLifecycleState.RE_ENTRY
                st.ramp_position = 0
                st.confirmation_count = 0
                self._step_ramp_up(st, cfg)
                logger.info("AssetSelector: {} COOLDOWN → RE_ENTRY", symbol)
        else:
            st.confirmation_count = 0

    def _transition_re_entry(
        self,
        symbol: str,
        st: _AssetState,
        cfg: AssetSelectorConfig,
    ) -> None:
        """RE_ENTRY → ACTIVE (ramp 완료) 또는 UNDERPERFORMING (재악화)."""
        # 재악화 체크
        if st.score < cfg.exclude_score_threshold:
            st.confirmation_count += 1
            if st.confirmation_count >= cfg.exclude_confirmation_bars:
                if self._can_exclude(symbol):
                    st.state = AssetLifecycleState.UNDERPERFORMING
                    st.confirmation_count = 0
                    self._step_ramp_down(st, cfg)
                    logger.info("AssetSelector: {} RE_ENTRY → UNDERPERFORMING", symbol)
                    return
                st.confirmation_count = 0
        else:
            st.confirmation_count = 0

        # Ramp up 진행
        self._step_ramp_up(st, cfg)

        # Ramp 완료 (multiplier == 1.0) → ACTIVE
        if st.multiplier >= 1.0 - _EPSILON:
            st.state = AssetLifecycleState.ACTIVE
            st.multiplier = 1.0
            st.ramp_position = 0
            st.confirmation_count = 0
            logger.info("AssetSelector: {} → ACTIVE", symbol)

    # ── Ramp Helpers ─────────────────────────────────────────────

    def _step_ramp_down(self, st: _AssetState, cfg: AssetSelectorConfig) -> None:
        """Ramp down: multiplier를 한 단계 감소."""
        if st.ramp_position >= cfg.ramp_steps:
            st.multiplier = 0.0
            return
        st.ramp_position += 1
        st.multiplier = max(0.0, 1.0 - st.ramp_position / cfg.ramp_steps)

    def _step_ramp_up(self, st: _AssetState, cfg: AssetSelectorConfig) -> None:
        """Ramp up: multiplier를 한 단계 증가."""
        if st.ramp_position >= cfg.ramp_steps:
            st.multiplier = 1.0
            return
        st.ramp_position += 1
        st.multiplier = min(1.0, st.ramp_position / cfg.ramp_steps)

    # ── Safety ───────────────────────────────────────────────────

    def _can_exclude(self, symbol: str) -> bool:
        """제외 가능 여부: min_active_assets 안전망."""
        active_count = sum(
            1
            for s, st in self._states.items()
            if s != symbol and st.multiplier > _EPSILON
        )
        return active_count >= self._config.min_active_assets

    # ── Absolute Thresholds ─────────────────────────────────────

    def _check_absolute_thresholds(
        self,
        returns: dict[str, list[float]],
    ) -> None:
        """Absolute threshold: Sharpe / DD 절대 기준으로 COOLDOWN 진입.

        cross-sectional rank와 독립적으로 개별 에셋 성과를 판단합니다.
        약세장에서 모든 에셋이 cross-sectionally 비슷해도 절대 기준으로 제외 가능.
        """
        cfg = self._config
        min_sharpe = cfg.absolute_min_sharpe
        max_dd = cfg.absolute_max_drawdown

        # 두 절대 기준 모두 비활성 → skip
        if min_sharpe is None and max_dd is None:
            return

        for symbol in self._symbols:
            st = self._states.get(symbol)
            if st is None or st.state == AssetLifecycleState.COOLDOWN:
                continue
            if st.permanently_excluded:
                continue

            rets = returns.get(symbol, [])
            if len(rets) < _MIN_DATA_BARS:
                continue

            sharpe = self._compute_sharpe(rets[-cfg.sharpe_lookback:])
            dd = self._compute_max_drawdown(rets[-cfg.sharpe_lookback:])

            sharpe_fail = min_sharpe is not None and sharpe < min_sharpe
            dd_fail = max_dd is not None and dd > max_dd

            if (sharpe_fail or dd_fail) and self._can_exclude(symbol):
                st.state = AssetLifecycleState.COOLDOWN
                st.multiplier = 0.0
                st.cooldown_bars = 0
                st.confirmation_count = 0
                st.ramp_position = 0
                st.cooldown_cycles += 1
                self._check_permanent_exclusion(symbol, st)
                logger.info(
                    "AssetSelector: {} absolute threshold excluded (Sharpe={:.2f}, DD={:.1%})",
                    symbol,
                    sharpe,
                    dd,
                )

    def _check_permanent_exclusion(self, symbol: str, st: _AssetState) -> None:
        """max_cooldown_cycles 초과 시 영구 제외."""
        max_cycles = self._config.max_cooldown_cycles
        if max_cycles is not None and st.cooldown_cycles >= max_cycles:
            st.permanently_excluded = True
            logger.warning(
                "AssetSelector: {} permanently excluded (cooldown cycles {}>={})",
                symbol,
                st.cooldown_cycles,
                max_cycles,
            )

    # ── Metrics Helpers ──────────────────────────────────────────

    @staticmethod
    def _compute_sharpe(returns: list[float]) -> float:
        """Sharpe ratio (rf=0, annualized)."""
        n = len(returns)
        if n < _MIN_SHARPE_SAMPLES:
            return 0.0
        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / (n - 1)
        vol = var**0.5
        if vol < _EPSILON:
            return 0.0
        return (mean * 365**0.5) / vol

    @staticmethod
    def _compute_cumulative_return(returns: list[float]) -> float:
        """Cumulative return: prod(1+r) - 1."""
        cum = 1.0
        for r in returns:
            cum *= 1.0 + r
        return cum - 1.0

    @staticmethod
    def _compute_max_drawdown(returns: list[float]) -> float:
        """Max drawdown (양수 표현)."""
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= 1.0 + r
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize selector state for persistence."""
        states_data: dict[str, dict[str, Any]] = {}
        for symbol, st in self._states.items():
            states_data[symbol] = {
                "state": st.state.value,
                "multiplier": st.multiplier,
                "score": st.score,
                "confirmation_count": st.confirmation_count,
                "cooldown_bars": st.cooldown_bars,
                "cooldown_cycles": st.cooldown_cycles,
                "ramp_position": st.ramp_position,
                "permanently_excluded": st.permanently_excluded,
            }
        return {"states": states_data}

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore selector state from persisted dict."""
        states_data = data.get("states")
        if not isinstance(states_data, dict):
            return

        for symbol, st_data in states_data.items():
            if symbol not in self._states or not isinstance(st_data, dict):
                continue

            st = self._states[symbol]
            state_val = st_data.get("state")
            if isinstance(state_val, str):
                with contextlib.suppress(ValueError):
                    st.state = AssetLifecycleState(state_val)

            mult = st_data.get("multiplier")
            if isinstance(mult, int | float):
                st.multiplier = float(mult)

            score = st_data.get("score")
            if isinstance(score, int | float):
                st.score = float(score)

            conf = st_data.get("confirmation_count")
            if isinstance(conf, int):
                st.confirmation_count = conf

            cd = st_data.get("cooldown_bars")
            if isinstance(cd, int):
                st.cooldown_bars = cd

            ramp = st_data.get("ramp_position")
            if isinstance(ramp, int):
                st.ramp_position = ramp

            cycles = st_data.get("cooldown_cycles")
            if isinstance(cycles, int):
                st.cooldown_cycles = cycles

            perm = st_data.get("permanently_excluded")
            if isinstance(perm, bool):
                st.permanently_excluded = perm


# ── Utility ──────────────────────────────────────────────────────


def _cross_sectional_rank(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional percentile rank (0~1).

    동순위는 평균 rank 사용. 단일 에셋 → 0.5.
    """
    n = len(values)
    if n <= 1:
        return dict.fromkeys(values, 0.5)

    sorted_items = sorted(values.items(), key=lambda x: x[1])
    ranks: dict[str, float] = {}

    i = 0
    while i < n:
        j = i
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[sorted_items[k][0]] = avg_rank / (n - 1) if n > 1 else 0.5
        i = j

    return ranks
