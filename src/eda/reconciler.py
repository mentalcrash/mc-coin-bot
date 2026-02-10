"""PositionReconciler — 거래소 vs PM 포지션 교차 검증.

주기적으로 거래소의 실제 포지션과 PM의 내부 상태를 비교하여
불일치를 감지합니다. 자동 수정 없음 (safety-first), drift 경고만 발행.

Usage:
    reconciler = PositionReconciler()
    await reconciler.initial_check(pm, futures_client, symbols)
    # periodic loop (LiveRunner에서 관리)
    await reconciler.periodic_check(pm, futures_client, symbols)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.exchange.binance_futures_client import BinanceFuturesClient

# Drift 임계값 (5%): 이 이상 차이나면 CRITICAL
_DRIFT_THRESHOLD = 0.05


class PositionReconciler:
    """거래소 vs PM 포지션 교차 검증."""

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

    async def _compare(
        self,
        pm: EDAPortfolioManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
    ) -> list[str]:
        """거래소 vs PM 포지션 비교.

        Returns:
            불일치가 발견된 심볼 리스트
        """
        from src.exchange.binance_futures_client import BinanceFuturesClient

        futures_symbols = [BinanceFuturesClient.to_futures_symbol(s) for s in symbols]
        exchange_positions = await futures_client.fetch_positions(futures_symbols)

        # 거래소 포지션을 symbol → {size, side} 맵으로 변환
        exchange_map: dict[str, dict[str, float | str]] = {}
        for pos in exchange_positions:
            sym = str(pos.get("symbol", ""))
            # Futures symbol → Spot symbol 역변환 (BTC/USDT:USDT → BTC/USDT)
            spot_sym = sym.split(":")[0] if ":" in sym else sym
            contracts = abs(float(pos.get("contracts", 0)))
            side = str(pos.get("side", "")).lower()
            exchange_map[spot_sym] = {"size": contracts, "side": side}

        drifts: list[str] = []

        for symbol in symbols:
            pm_pos = pm.positions.get(symbol)
            ex_info = exchange_map.get(symbol)

            pm_size = pm_pos.size if pm_pos and pm_pos.is_open else 0.0
            ex_size = float(ex_info["size"]) if ex_info else 0.0

            # 방향 비교 (PM Direction vs exchange side)
            pm_side = ""
            if pm_pos and pm_pos.is_open:
                from src.models.types import Direction

                pm_side = "long" if pm_pos.direction == Direction.LONG else "short"

            ex_side = str(ex_info["side"]) if ex_info else ""

            # 불일치 감지
            has_drift = False

            # 방향 불일치
            if pm_size > 0 and ex_size > 0 and pm_side != ex_side:
                logger.warning(
                    "PositionReconciler: {} direction mismatch — PM={}, Exchange={}",
                    symbol,
                    pm_side,
                    ex_side,
                )
                has_drift = True

            # 수량 불일치 (상대 비교)
            if pm_size > 0 or ex_size > 0:
                max_size = max(pm_size, ex_size)
                if max_size > 0:
                    drift_ratio = abs(pm_size - ex_size) / max_size
                    if drift_ratio > _DRIFT_THRESHOLD:
                        logger.warning(
                            "PositionReconciler: {} size drift — PM={:.6f}, Exchange={:.6f} ({:.1f}%)",
                            symbol,
                            pm_size,
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

            if has_drift:
                drifts.append(symbol)

        return drifts
