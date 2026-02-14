"""reconciler_formatters 테스트 — DriftDetail + embed 구조 검증."""

from __future__ import annotations

from src.notification.reconciler_formatters import (
    DriftDetail,
    format_balance_drift_embed,
    format_position_drift_embed,
)

# Discord 색상 코드
_COLOR_ORANGE = 0xE67E22
_COLOR_RED = 0xED4245
_COLOR_YELLOW = 0xFFFF00


class TestDriftDetail:
    """DriftDetail dataclass 검증."""

    def test_frozen(self) -> None:
        """frozen=True 확인."""
        detail = DriftDetail(
            symbol="BTC/USDT",
            pm_size=0.01,
            pm_side="LONG",
            exchange_size=0.015,
            exchange_side="LONG",
            drift_pct=33.3,
        )
        assert detail.symbol == "BTC/USDT"
        assert detail.is_orphan is False
        assert detail.auto_corrected is False

    def test_defaults(self) -> None:
        """기본값 확인."""
        detail = DriftDetail(
            symbol="ETH/USDT",
            pm_size=0.0,
            pm_side="FLAT",
            exchange_size=1.0,
            exchange_side="LONG",
            drift_pct=100.0,
            is_orphan=True,
        )
        assert detail.is_orphan is True
        assert detail.auto_corrected is False


class TestFormatPositionDriftEmbed:
    """format_position_drift_embed() 테스트."""

    def test_single_normal_drift(self) -> None:
        """일반 drift → ORANGE 색상."""
        drifts = [
            DriftDetail(
                symbol="BTC/USDT",
                pm_size=0.01,
                pm_side="LONG",
                exchange_size=0.012,
                exchange_side="LONG",
                drift_pct=5.0,
            ),
        ]
        embed = format_position_drift_embed(drifts)

        assert embed["color"] == _COLOR_ORANGE
        assert "1 symbol" in embed["title"]
        assert len(embed["fields"]) == 1
        assert "BTC/USDT" in embed["fields"][0]["name"]
        assert "5.0%" in embed["fields"][0]["value"]
        assert "timestamp" in embed

    def test_orphan_drift_red(self) -> None:
        """Orphan drift → RED 색상."""
        drifts = [
            DriftDetail(
                symbol="ETH/USDT",
                pm_size=0.0,
                pm_side="FLAT",
                exchange_size=1.0,
                exchange_side="LONG",
                drift_pct=100.0,
                is_orphan=True,
            ),
        ]
        embed = format_position_drift_embed(drifts)

        assert embed["color"] == _COLOR_RED
        assert "ORPHAN" in embed["fields"][0]["name"]

    def test_critical_drift_red(self) -> None:
        """10%+ drift → RED 색상."""
        drifts = [
            DriftDetail(
                symbol="BTC/USDT",
                pm_size=0.01,
                pm_side="LONG",
                exchange_size=0.02,
                exchange_side="LONG",
                drift_pct=50.0,
            ),
        ]
        embed = format_position_drift_embed(drifts)
        assert embed["color"] == _COLOR_RED

    def test_multiple_symbols(self) -> None:
        """멀티 심볼 → 각 심볼별 field."""
        drifts = [
            DriftDetail(
                symbol="BTC/USDT",
                pm_size=0.01,
                pm_side="LONG",
                exchange_size=0.012,
                exchange_side="LONG",
                drift_pct=5.0,
            ),
            DriftDetail(
                symbol="ETH/USDT",
                pm_size=1.0,
                pm_side="SHORT",
                exchange_size=1.2,
                exchange_side="SHORT",
                drift_pct=8.0,
            ),
        ]
        embed = format_position_drift_embed(drifts)

        assert "2 symbols" in embed["title"]
        assert len(embed["fields"]) == 2

    def test_auto_corrected_label(self) -> None:
        """Auto-corrected → 'Auto-corrected' label."""
        drifts = [
            DriftDetail(
                symbol="BTC/USDT",
                pm_size=0.01,
                pm_side="LONG",
                exchange_size=0.015,
                exchange_side="LONG",
                drift_pct=33.3,
                auto_corrected=True,
            ),
        ]
        embed = format_position_drift_embed(drifts)
        assert "Auto-corrected" in embed["fields"][0]["value"]


class TestFormatBalanceDriftEmbed:
    """format_balance_drift_embed() 테스트."""

    def test_warning_level(self) -> None:
        """2~5% drift → YELLOW 색상, WARNING."""
        embed = format_balance_drift_embed(
            pm_equity=10000.0,
            exchange_equity=10300.0,
            drift_pct=3.0,
        )

        assert embed["color"] == _COLOR_YELLOW
        assert "WARNING" in embed["title"]
        assert len(embed["fields"]) == 3
        assert "3.0%" in embed["fields"][2]["value"]

    def test_critical_level(self) -> None:
        """5%+ drift → RED 색상, CRITICAL."""
        embed = format_balance_drift_embed(
            pm_equity=10000.0,
            exchange_equity=8000.0,
            drift_pct=20.0,
        )

        assert embed["color"] == _COLOR_RED
        assert "CRITICAL" in embed["title"]

    def test_has_footer_and_timestamp(self) -> None:
        """Footer + timestamp 존재."""
        embed = format_balance_drift_embed(
            pm_equity=10000.0,
            exchange_equity=10200.0,
            drift_pct=2.0,
        )

        assert embed["footer"]["text"] == "MC-Coin-Bot"
        assert "timestamp" in embed
