"""Health formatters + regime score 테스트."""

from __future__ import annotations

from datetime import UTC, datetime

from src.notification.health_formatters import (
    _COLOR_BLUE,
    _COLOR_GREEN,
    _COLOR_RED,
    _COLOR_YELLOW,
    _heartbeat_color,
    format_heartbeat_embed,
    format_strategy_health_embed,
    format_uptime,
)
from src.notification.health_models import (
    PositionStatus,
    StrategyHealthSnapshot,
    SystemHealthSnapshot,
)

# ─── Fixtures ──────────────────────────────────────────────


def _make_health_snapshot(**overrides: object) -> SystemHealthSnapshot:
    defaults: dict[str, object] = {
        "timestamp": datetime(2026, 2, 14, tzinfo=UTC),
        "uptime_seconds": 86400.0 + 3600.0 * 14,
        "total_equity": 52341.0,
        "available_cash": 30000.0,
        "aggregate_leverage": 0.35,
        "open_position_count": 3,
        "total_symbols": 5,
        "current_drawdown": 0.023,
        "peak_equity": 53500.0,
        "is_circuit_breaker_active": False,
        "today_pnl": 127.0,
        "today_trades": 2,
        "stale_symbol_count": 0,
        "bars_emitted": 1200,
        "events_dropped": 0,
        "max_queue_depth": 12,
        "is_notification_degraded": False,
    }
    defaults.update(overrides)
    return SystemHealthSnapshot(**defaults)  # type: ignore[arg-type]


# ─── Heartbeat color 테스트 ────────────────────────────────


class TestHeartbeatColor:
    def test_green_normal(self) -> None:
        snap = _make_health_snapshot()
        assert _heartbeat_color(snap) == _COLOR_GREEN

    def test_red_circuit_breaker(self) -> None:
        snap = _make_health_snapshot(is_circuit_breaker_active=True)
        assert _heartbeat_color(snap) == _COLOR_RED

    def test_red_high_drawdown(self) -> None:
        snap = _make_health_snapshot(current_drawdown=0.09)
        assert _heartbeat_color(snap) == _COLOR_RED

    def test_red_all_stale(self) -> None:
        snap = _make_health_snapshot(stale_symbol_count=5, total_symbols=5)
        assert _heartbeat_color(snap) == _COLOR_RED

    def test_red_notification_degraded(self) -> None:
        snap = _make_health_snapshot(is_notification_degraded=True)
        assert _heartbeat_color(snap) == _COLOR_RED

    def test_red_safety_stop_failures(self) -> None:
        snap = _make_health_snapshot(safety_stop_failures=5)
        assert _heartbeat_color(snap) == _COLOR_RED

    def test_yellow_moderate_drawdown(self) -> None:
        snap = _make_health_snapshot(current_drawdown=0.06)
        assert _heartbeat_color(snap) == _COLOR_YELLOW

    def test_yellow_some_stale(self) -> None:
        snap = _make_health_snapshot(stale_symbol_count=1)
        assert _heartbeat_color(snap) == _COLOR_YELLOW

    def test_yellow_high_queue_depth(self) -> None:
        snap = _make_health_snapshot(max_queue_depth=60)
        assert _heartbeat_color(snap) == _COLOR_YELLOW


# ─── Uptime 포맷 테스트 ──────────────────────────────────


class TestFormatUptime:
    def test_minutes_only(self) -> None:
        assert format_uptime(300) == "5m"

    def test_hours_and_minutes(self) -> None:
        assert format_uptime(3720) == "1h 2m"

    def test_days_hours_minutes(self) -> None:
        result = format_uptime(86400 + 3600 * 14)
        assert result == "1d 14h 0m"


# ─── Heartbeat embed 테스트 ───────────────────────────────


class TestFormatHeartbeatEmbed:
    def test_basic_structure(self) -> None:
        snap = _make_health_snapshot()
        embed = format_heartbeat_embed(snap)
        assert embed["title"] == "System Heartbeat — OK"
        assert embed["color"] == _COLOR_GREEN
        assert len(embed["fields"]) == 10
        assert embed["footer"]["text"] == "MC-Coin-Bot"

    def test_alert_status(self) -> None:
        snap = _make_health_snapshot(is_circuit_breaker_active=True)
        embed = format_heartbeat_embed(snap)
        assert "ALERT" in embed["title"]


# ─── Strategy health embed 테스트 ────────────────────────


class TestFormatStrategyHealthEmbed:
    def test_basic_structure(self) -> None:
        pos = PositionStatus(
            symbol="SOL/USDT",
            direction="LONG",
            unrealized_pnl=120.5,
            size=10.0,
            current_weight=0.12,
        )
        snap = StrategyHealthSnapshot(
            timestamp=datetime(2026, 2, 14, tzinfo=UTC),
            rolling_sharpe_30d=1.38,
            win_rate_recent=0.65,
            profit_factor=1.8,
            total_closed_trades=142,
            open_positions=(pos,),
            is_circuit_breaker_active=False,
            alpha_decay_detected=False,
        )
        embed = format_strategy_health_embed(snap)
        assert embed["title"] == "Strategy Health Report"
        assert embed["color"] == _COLOR_BLUE
        assert any("1.38" in str(f.get("value", "")) for f in embed["fields"])

    def test_alpha_decay_red(self) -> None:
        snap = StrategyHealthSnapshot(
            timestamp=datetime(2026, 2, 14, tzinfo=UTC),
            rolling_sharpe_30d=0.5,
            win_rate_recent=0.5,
            profit_factor=1.1,
            total_closed_trades=50,
            open_positions=(),
            is_circuit_breaker_active=False,
            alpha_decay_detected=True,
        )
        embed = format_strategy_health_embed(snap)
        assert embed["color"] == _COLOR_RED
        sharpe_field = embed["fields"][0]
        assert "DECAY" in str(sharpe_field["value"])

    def test_profit_factor_inf_shows_na(self) -> None:
        """profit_factor=inf → 'N/A' 표시."""
        snap = StrategyHealthSnapshot(
            timestamp=datetime(2026, 2, 14, tzinfo=UTC),
            rolling_sharpe_30d=1.0,
            win_rate_recent=0.7,
            profit_factor=float("inf"),
            total_closed_trades=10,
            open_positions=(),
            is_circuit_breaker_active=False,
            alpha_decay_detected=False,
        )
        embed = format_strategy_health_embed(snap)
        pf_field = next(f for f in embed["fields"] if f["name"] == "Profit Factor")
        assert pf_field["value"] == "N/A"

    def test_no_positions(self) -> None:
        snap = StrategyHealthSnapshot(
            timestamp=datetime(2026, 2, 14, tzinfo=UTC),
            rolling_sharpe_30d=1.0,
            win_rate_recent=0.6,
            profit_factor=1.5,
            total_closed_trades=100,
            open_positions=(),
            is_circuit_breaker_active=False,
            alpha_decay_detected=False,
        )
        embed = format_strategy_health_embed(snap)
        pos_field = next(f for f in embed["fields"] if "Open Positions" in f["name"])
        assert pos_field["value"] == "None"
