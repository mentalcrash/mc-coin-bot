"""PositionReconciler — 거래소 vs PM 포지션 교차 검증.

주기적으로 거래소의 실제 포지션과 PM의 내부 상태를 비교하여
불일치를 감지합니다. Optional auto-correction 모드로 PM 상태를
거래소 기준으로 보정할 수 있습니다.

Usage:
    reconciler = PositionReconciler()
    await reconciler.initial_check(pm, futures_client, symbols)
    # periodic loop (LiveRunner에서 관리)
    await reconciler.periodic_check(pm, futures_client, symbols)
    # auto-correction 활성화 (LIVE 모드 전용)
    reconciler = PositionReconciler(auto_correct=True)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.models.types import Direction

if TYPE_CHECKING:
    from src.eda.portfolio_manager import EDAPortfolioManager, Position
    from src.exchange.binance_futures_client import BinanceFuturesClient

# Position drift 임계값 (2%): 이 이상 차이나면 CRITICAL
_DRIFT_THRESHOLD = 0.02

# Balance drift 임계값
_BALANCE_DRIFT_THRESHOLD = 0.05  # 5%: CRITICAL
_BALANCE_WARN_THRESHOLD = 0.02  # 2%: WARNING

# Auto-correction 임계값: drift가 이 이상이면 PM 상태를 거래소 기준으로 보정
_AUTO_CORRECT_THRESHOLD = 0.10  # 10%


class PositionReconciler:
    """거래소 vs PM 포지션 교차 검증.

    Args:
        auto_correct: True 시 drift > 10% 인 포지션의 PM size를 거래소 기준으로 보정.
                      기본 False (safety-first: 경고만 발행).
    """

    def __init__(self, *, auto_correct: bool = False) -> None:
        self._auto_correct = auto_correct
        self._corrections_applied: int = 0

    @property
    def auto_correct_enabled(self) -> bool:
        """Auto-correction 활성화 여부."""
        return self._auto_correct

    @property
    def corrections_applied(self) -> int:
        """적용된 auto-correction 횟수."""
        return self._corrections_applied

    async def initial_check(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> list[str]:
        """시작 시 초기 포지션 검증.

        Args:
            pm: PortfolioManager
            futures_client: Futures client
            symbols: 트레이딩 심볼 리스트

        Returns:
            불일치 심볼 리스트 (비어 있으면 정상)
        """
        logger.info("PositionReconciler: Running initial check...")
        drifts = await self._compare(pm, futures_client, symbols)

        if drifts:
            logger.critical(
                "PositionReconciler: {} position drifts detected at startup! Symbols: {}",
                len(drifts),
                drifts,
            )
        else:
            logger.info("PositionReconciler: All positions match")

        return drifts

    async def periodic_check(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> list[str]:
        """주기적 포지션 검증.

        Args:
            pm: PortfolioManager
            futures_client: Futures client
            symbols: 트레이딩 심볼 리스트

        Returns:
            불일치 심볼 리스트
        """
        try:
            drifts = await self._compare(pm, futures_client, symbols)
        except Exception:
            logger.exception("PositionReconciler: Periodic check failed")
            return []
        else:
            if drifts:
                logger.warning(
                    "PositionReconciler: {} drift(s) detected: {}",
                    len(drifts),
                    drifts,
                )
            return drifts

    async def check_balance(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
    ) -> float | None:
        """PM equity vs 거래소 잔고 비교.

        Args:
            pm: PortfolioManager
            futures_client: Futures client

        Returns:
            거래소 equity (float) 또는 실패 시 None
        """
        try:
            balance = await futures_client.fetch_balance()
            usdt_info = balance.get("USDT", {})
            exchange_equity = float(
                usdt_info.get("total", 0) if isinstance(usdt_info, dict) else 0
            )
        except Exception:
            logger.exception("PositionReconciler: Failed to fetch balance")
            return None

        pm_equity = pm.total_equity
        if pm_equity <= 0:
            return exchange_equity

        drift = abs(pm_equity - exchange_equity) / pm_equity

        if drift > _BALANCE_DRIFT_THRESHOLD:
            logger.critical(
                "PositionReconciler: BALANCE DRIFT {:.1%} — PM=${:.0f} vs Exchange=${:.0f}",
                drift,
                pm_equity,
                exchange_equity,
            )
        elif drift > _BALANCE_WARN_THRESHOLD:
            logger.warning(
                "PositionReconciler: balance drift {:.1%} — PM=${:.0f} vs Exchange=${:.0f}",
                drift,
                pm_equity,
                exchange_equity,
            )

        return exchange_equity

    async def _compare(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> list[str]:
        """거래소 vs PM 포지션 비교.

        auto_correct 활성화 시 drift > _AUTO_CORRECT_THRESHOLD 인 포지션의
        PM size를 거래소 기준으로 보정합니다.

        Returns:
            불일치가 발견된 심볼 리스트
        """
        from src.exchange.binance_futures_client import BinanceFuturesClient

        futures_symbols = [BinanceFuturesClient.to_futures_symbol(s) for s in symbols]
        exchange_positions = await futures_client.fetch_positions(futures_symbols)

        # Hedge Mode: 심볼당 LONG/SHORT 분리 매핑
        exchange_map: dict[str, dict[str, float]] = {}
        for pos in exchange_positions:
            sym = str(pos.get("symbol", ""))
            spot_sym = sym.split(":")[0] if ":" in sym else sym
            contracts = abs(float(pos.get("contracts", 0)))
            side = str(pos.get("side", "")).lower()

            if spot_sym not in exchange_map:
                exchange_map[spot_sym] = {"long_size": 0.0, "short_size": 0.0}

            if side == "long":
                exchange_map[spot_sym]["long_size"] = contracts
            elif side == "short":
                exchange_map[spot_sym]["short_size"] = contracts

        drifts: list[str] = []

        for symbol in symbols:
            pm_pos = pm.positions.get(symbol)
            ex_info = exchange_map.get(symbol, {"long_size": 0.0, "short_size": 0.0})

            pm_size = pm_pos.size if pm_pos and pm_pos.is_open else 0.0
            pm_dir = pm_pos.direction if pm_pos and pm_pos.is_open else Direction.NEUTRAL

            if self._check_symbol_drift(symbol, pm_size, pm_dir, ex_info):
                drifts.append(symbol)

                # Auto-correction: PM size를 거래소 기준으로 보정
                if self._auto_correct and pm_pos is not None:
                    ex_size = self._get_exchange_size(pm_dir, ex_info)
                    self._apply_correction(pm_pos, ex_size, pm_dir, ex_info)

        return drifts

    def _get_exchange_size(
        self, pm_dir: Direction, ex_info: dict[str, float]
    ) -> float:
        """PM 방향에 맞는 거래소 포지션 크기 반환."""
        if pm_dir == Direction.LONG:
            return ex_info["long_size"]
        if pm_dir == Direction.SHORT:
            return ex_info["short_size"]
        return ex_info["long_size"] + ex_info["short_size"]

    def _apply_correction(
        self,
        pm_pos: Position,
        ex_size: float,
        pm_dir: Direction,
        ex_info: dict[str, float],
    ) -> None:
        """PM 포지션 크기를 거래소 기준으로 보정.

        drift가 _AUTO_CORRECT_THRESHOLD 이상일 때만 보정합니다.
        """
        pm_size = pm_pos.size
        max_size = max(pm_size, ex_size)
        if max_size <= 0:
            return

        drift_ratio = abs(pm_size - ex_size) / max_size
        if drift_ratio < _AUTO_CORRECT_THRESHOLD:
            return

        old_size = pm_pos.size
        pm_pos.size = ex_size

        # 포지션이 0이 되었으면 NEUTRAL로 설정
        if ex_size <= 0:
            pm_pos.direction = Direction.NEUTRAL
            pm_pos.avg_entry_price = 0.0

        self._corrections_applied += 1
        logger.warning(
            "PositionReconciler AUTO-CORRECT: {} size {:.6f} → {:.6f} (drift {:.1%})",
            pm_pos.symbol,
            old_size,
            ex_size,
            drift_ratio,
        )

    @staticmethod
    def _check_symbol_drift(
        symbol: str,
        pm_size: float,
        pm_dir: Direction,
        ex_info: dict[str, float],
    ) -> bool:
        """단일 심볼의 포지션 불일치 검사.

        Returns:
            True면 drift 감지됨
        """

        # PM 방향에 맞는 거래소 사이즈 선택
        if pm_dir == Direction.LONG:
            ex_size = ex_info["long_size"]
        elif pm_dir == Direction.SHORT:
            ex_size = ex_info["short_size"]
        else:
            ex_size = ex_info["long_size"] + ex_info["short_size"]

        has_drift = False

        # 수량 불일치 (상대 비교)
        if pm_size > 0 or ex_size > 0:
            max_size = max(pm_size, ex_size)
            if max_size > 0:
                drift_ratio = abs(pm_size - ex_size) / max_size
                if drift_ratio > _DRIFT_THRESHOLD:
                    dir_label = pm_dir.value if pm_dir != Direction.NEUTRAL else "neutral"
                    logger.warning(
                        "PositionReconciler: {} size drift — PM={:.6f} ({}), Exchange={:.6f} ({:.1f}%)",
                        symbol,
                        pm_size,
                        dir_label,
                        ex_size,
                        drift_ratio * 100,
                    )
                    has_drift = True

        # 한쪽만 포지션 보유
        if (pm_size > 0) != (ex_size > 0):
            who_has = "PM" if pm_size > 0 else "Exchange"
            logger.warning(
                "PositionReconciler: {} — only {} has position (PM={:.6f}, Exchange={:.6f})",
                symbol,
                who_has,
                pm_size,
                ex_size,
            )
            has_drift = True

        # 거래소에 반대방향 포지션 존재 감지
        if pm_dir == Direction.LONG and ex_info["short_size"] > 0:
            logger.warning(
                "PositionReconciler: {} — PM is LONG but exchange has SHORT ({:.6f})",
                symbol,
                ex_info["short_size"],
            )
            has_drift = True
        elif pm_dir == Direction.SHORT and ex_info["long_size"] > 0:
            logger.warning(
                "PositionReconciler: {} — PM is SHORT but exchange has LONG ({:.6f})",
                symbol,
                ex_info["long_size"],
            )
            has_drift = True

        return has_drift
