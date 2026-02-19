"""Event -> Discord Embed 변환 테스트."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.notification.formatters import (
    format_balance_embed,
    format_circuit_breaker_embed,
    format_daily_report_embed,
    format_enhanced_daily_report_embed,
    format_fill_embed,
    format_fill_with_position_embed,
    format_position_embed,
    format_risk_alert_embed,
    format_weekly_report_embed,
)

if TYPE_CHECKING:
    from src.core.events import (
        BalanceUpdateEvent,
        CircuitBreakerEvent,
        FillEvent,
        PositionUpdateEvent,
        RiskAlertEvent,
    )

_COLOR_GREEN = 0x57F287
_COLOR_RED = 0xED4245
_COLOR_BLUE = 0x3498DB
_COLOR_YELLOW = 0xFFFF00
_COLOR_ORANGE = 0xE67E22


class TestFormatFillEmbed:
    def test_buy_fill(self, sample_fill: FillEvent) -> None:
        embed = format_fill_embed(sample_fill)
        assert "BUY" in embed["title"]
        assert "BTC/USDT" in embed["title"]
        assert embed["color"] == _COLOR_GREEN
        assert len(embed["fields"]) == 4
        assert embed["footer"]["text"] == "MC-Coin-Bot"
        assert "timestamp" in embed

    def test_sell_fill(self, sample_fill_sell: FillEvent) -> None:
        embed = format_fill_embed(sample_fill_sell)
        assert "SELL" in embed["title"]
        assert "ETH/USDT" in embed["title"]
        assert embed["color"] == _COLOR_RED

    def test_price_field(self, sample_fill: FillEvent) -> None:
        embed = format_fill_embed(sample_fill)
        price_field = embed["fields"][0]
        assert price_field["name"] == "Price"
        assert "$50,000" in price_field["value"]
        assert price_field["inline"] is True

    def test_value_field(self, sample_fill: FillEvent) -> None:
        embed = format_fill_embed(sample_fill)
        value_field = embed["fields"][3]
        assert value_field["name"] == "Value"
        assert "$5,000" in value_field["value"]


class TestFormatCircuitBreakerEmbed:
    def test_embed_structure(self, sample_circuit_breaker: CircuitBreakerEvent) -> None:
        embed = format_circuit_breaker_embed(sample_circuit_breaker)
        assert embed["title"] == "CIRCUIT BREAKER TRIGGERED"
        assert embed["color"] == _COLOR_ORANGE
        assert embed["description"] == "System drawdown exceeded 10%"
        assert embed["fields"][0]["value"] == "Yes"

    def test_close_all_false(self) -> None:
        from src.core.events import CircuitBreakerEvent

        event = CircuitBreakerEvent(reason="Test", close_all_positions=False)
        embed = format_circuit_breaker_embed(event)
        assert embed["fields"][0]["value"] == "No"


class TestFormatRiskAlertEmbed:
    def test_warning(self, sample_risk_alert_warning: RiskAlertEvent) -> None:
        embed = format_risk_alert_embed(sample_risk_alert_warning)
        assert embed["title"] == "RISK WARNING"
        assert embed["color"] == _COLOR_YELLOW
        assert "7.5%" in embed["description"]

    def test_critical(self, sample_risk_alert_critical: RiskAlertEvent) -> None:
        embed = format_risk_alert_embed(sample_risk_alert_critical)
        assert embed["title"] == "RISK CRITICAL"
        assert embed["color"] == _COLOR_ORANGE


class TestFormatBalanceEmbed:
    def test_embed_structure(self, sample_balance_update: BalanceUpdateEvent) -> None:
        embed = format_balance_embed(sample_balance_update)
        assert embed["title"] == "Balance Update"
        assert embed["color"] == _COLOR_BLUE
        assert len(embed["fields"]) == 3
        assert "$10,500" in embed["fields"][0]["value"]
        assert "$8,000" in embed["fields"][1]["value"]
        assert "$2,500" in embed["fields"][2]["value"]


class TestFormatFillWithPositionEmbed:
    def test_combined_embed(
        self, sample_fill: FillEvent, sample_position_update: PositionUpdateEvent
    ) -> None:
        embed = format_fill_with_position_embed(sample_fill, sample_position_update)
        assert "BUY" in embed["title"]
        assert "BTC/USDT" in embed["title"]
        assert embed["color"] == _COLOR_GREEN
        assert len(embed["fields"]) == 7
        field_names = [f["name"] for f in embed["fields"]]
        assert field_names == [
            "Price",
            "Qty",
            "Fee",
            "Value",
            "Position",
            "Avg Entry",
            "Realized PnL",
        ]

    def test_combined_sell(
        self, sample_fill_sell: FillEvent, sample_position_update: PositionUpdateEvent
    ) -> None:
        embed = format_fill_with_position_embed(sample_fill_sell, sample_position_update)
        assert "SELL" in embed["title"]
        assert embed["color"] == _COLOR_RED

    def test_position_field_value(
        self, sample_fill: FillEvent, sample_position_update: PositionUpdateEvent
    ) -> None:
        embed = format_fill_with_position_embed(sample_fill, sample_position_update)
        pos_field = embed["fields"][4]
        assert pos_field["name"] == "Position"
        assert "LONG" in pos_field["value"]
        avg_field = embed["fields"][5]
        assert "$50,000" in avg_field["value"]
        pnl_field = embed["fields"][6]
        assert "$+100.00" in pnl_field["value"]


class TestFormatPositionEmbed:
    def test_long_profit(self, sample_position_update: PositionUpdateEvent) -> None:
        embed = format_position_embed(sample_position_update)
        assert "BTC/USDT" in embed["title"]
        assert embed["color"] == _COLOR_GREEN  # positive PnL
        assert len(embed["fields"]) == 5
        assert embed["fields"][0]["value"] == "LONG"
        assert "250.00" in embed["fields"][3]["value"]

    def test_loss_color(self) -> None:
        from src.core.events import PositionUpdateEvent
        from src.models.types import Direction

        event = PositionUpdateEvent(
            symbol="ETH/USDT",
            direction=Direction.SHORT,
            size=1.0,
            avg_entry_price=3000.0,
            unrealized_pnl=-150.0,
        )
        embed = format_position_embed(event)
        assert embed["color"] == _COLOR_RED
        assert embed["fields"][0]["value"] == "SHORT"


class TestFormatDailyReportEmbed:
    def test_embed_structure(self) -> None:
        from unittest.mock import MagicMock

        metrics = MagicMock()
        metrics.max_drawdown = 5.2
        metrics.sharpe_ratio = 1.35
        embed = format_daily_report_embed(
            metrics, open_positions=3, total_equity=10500.0, trades_today=[]
        )
        assert embed["title"] == "Daily Report"
        assert embed["color"] == _COLOR_BLUE
        assert len(embed["fields"]) == 6
        field_names = [f["name"] for f in embed["fields"]]
        assert "Today's Trades" in field_names
        assert "Sharpe Ratio" in field_names

    def test_has_timestamp(self) -> None:
        from unittest.mock import MagicMock

        metrics = MagicMock()
        metrics.max_drawdown = 5.0
        metrics.sharpe_ratio = 1.0
        embed = format_daily_report_embed(
            metrics, open_positions=0, total_equity=10000.0, trades_today=[]
        )
        assert "timestamp" in embed
        assert len(embed["timestamp"]) > 0


class TestFormatWeeklyReportEmbed:
    def test_embed_structure(self) -> None:
        from unittest.mock import MagicMock

        metrics = MagicMock()
        metrics.max_drawdown = 3.1
        metrics.sharpe_ratio = 1.5
        embed = format_weekly_report_embed(metrics, trades_week=[])
        assert embed["title"] == "Weekly Report"
        assert embed["color"] == _COLOR_BLUE
        assert len(embed["fields"]) == 6
        field_names = [f["name"] for f in embed["fields"]]
        assert "Weekly Trades" in field_names
        assert "Best Trade" in field_names

    def test_has_timestamp(self) -> None:
        from unittest.mock import MagicMock

        metrics = MagicMock()
        metrics.max_drawdown = 3.0
        metrics.sharpe_ratio = 1.0
        embed = format_weekly_report_embed(metrics, trades_week=[])
        assert "timestamp" in embed
        assert len(embed["timestamp"]) > 0


# ─── Enhanced Daily Report Embed 테스트 ──────────────────


class TestFormatEnhancedDailyReportEmbed:
    def _make_metrics(self) -> Any:
        from unittest.mock import MagicMock

        m = MagicMock()
        m.max_drawdown = 5.2
        m.sharpe_ratio = 1.35
        return m

    def test_enhanced_daily_basic_structure(self) -> None:
        """기본 6 fields + 추가 sections."""
        from unittest.mock import MagicMock

        system_health = MagicMock()
        system_health.uptime_seconds = 86400.0
        system_health.is_circuit_breaker_active = False
        system_health.total_symbols = 8
        system_health.stale_symbol_count = 0

        strategy_health = MagicMock()
        strategy_health.alpha_decay_detected = False
        strategy_health.strategy_breakdown = ()
        strategy_health.open_positions = ()

        embed = format_enhanced_daily_report_embed(
            metrics=self._make_metrics(),
            open_positions=3,
            total_equity=10500.0,
            trades_today=[],
            system_health=system_health,
            strategy_health=strategy_health,
        )
        assert embed["title"] == "Daily Report"
        assert embed["color"] == _COLOR_BLUE
        # 6 기본 + 3 system status = 9
        assert len(embed["fields"]) >= 9
        field_names = [f["name"] for f in embed["fields"]]
        assert "Today's PnL" in field_names
        assert "Uptime" in field_names

    def test_enhanced_daily_strategy_breakdown(self) -> None:
        """전략별 섹션 포맷."""
        from unittest.mock import MagicMock

        from src.notification.health_models import StrategyPerformanceSnapshot

        sp = StrategyPerformanceSnapshot(
            strategy_name="ctrend",
            rolling_sharpe=1.2,
            win_rate=0.55,
            total_pnl=500.0,
            trade_count=10,
            status="HEALTHY",
        )
        strategy_health = MagicMock()
        strategy_health.alpha_decay_detected = False
        strategy_health.strategy_breakdown = (sp,)
        strategy_health.open_positions = ()

        embed = format_enhanced_daily_report_embed(
            metrics=self._make_metrics(),
            open_positions=0,
            total_equity=10000.0,
            trades_today=[],
            strategy_health=strategy_health,
        )
        field_names = [f["name"] for f in embed["fields"]]
        assert "Strategy Breakdown (30d)" in field_names
        breakdown_field = next(
            f for f in embed["fields"] if f["name"] == "Strategy Breakdown (30d)"
        )
        assert "ctrend" in breakdown_field["value"]

    def test_enhanced_daily_regime_section(self) -> None:
        """Market Regime 필드."""
        from unittest.mock import MagicMock

        regime = MagicMock()
        regime.regime_label = "Extreme Greed"
        regime.regime_score = 0.65
        regime.symbols = ()

        embed = format_enhanced_daily_report_embed(
            metrics=self._make_metrics(),
            open_positions=0,
            total_equity=10000.0,
            trades_today=[],
            regime_report=regime,
        )
        field_names = [f["name"] for f in embed["fields"]]
        assert "Market Regime" in field_names

    def test_enhanced_daily_alpha_decay_color(self) -> None:
        """alpha_decay 시 RED 색상."""
        from unittest.mock import MagicMock

        strategy_health = MagicMock()
        strategy_health.alpha_decay_detected = True
        strategy_health.rolling_sharpe_30d = 0.3
        strategy_health.strategy_breakdown = ()
        strategy_health.open_positions = ()

        embed = format_enhanced_daily_report_embed(
            metrics=self._make_metrics(),
            open_positions=0,
            total_equity=10000.0,
            trades_today=[],
            strategy_health=strategy_health,
        )
        assert embed["color"] == _COLOR_RED
        field_names = [f["name"] for f in embed["fields"]]
        assert "Alpha Decay Warning" in field_names

    def test_enhanced_daily_all_none(self) -> None:
        """모든 health 데이터 None → 기본 daily와 동일."""
        embed = format_enhanced_daily_report_embed(
            metrics=self._make_metrics(),
            open_positions=0,
            total_equity=10000.0,
            trades_today=[],
        )
        assert embed["title"] == "Daily Report"
        assert embed["color"] == _COLOR_BLUE
        assert len(embed["fields"]) == 6
