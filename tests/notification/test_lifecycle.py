"""Tests for src/notification/lifecycle.py — Bot Lifecycle embed formatters."""

from __future__ import annotations

from src.notification.lifecycle import (
    format_crash_embed,
    format_shutdown_embed,
    format_startup_embed,
)


class TestFormatStartupEmbed:
    """format_startup_embed() 단위 테스트."""

    def test_basic_fields(self) -> None:
        embed = format_startup_embed(
            mode="paper",
            strategy_name="TSMOM",
            symbols=["BTC/USDT", "ETH/USDT"],
            capital=10000.0,
            timeframe="1D",
        )
        assert embed["title"] == "MC Coin Bot Started"
        assert embed["color"] == 0x57F287  # GREEN
        assert "footer" in embed
        assert "timestamp" in embed

        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["Mode"] == "PAPER"
        assert fields["Strategy"] == "TSMOM"
        assert fields["Timeframe"] == "1D"
        assert "$10,000" in fields["Capital"]
        assert "BTC/USDT" in fields["Symbols"]
        assert "ETH/USDT" in fields["Symbols"]

    def test_empty_symbols(self) -> None:
        embed = format_startup_embed(
            mode="live",
            strategy_name="Test",
            symbols=[],
            capital=5000.0,
            timeframe="4h",
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["Symbols"] == "N/A"

    def test_startup_with_pod_summaries(self) -> None:
        pods = [
            {"pod_id": "anchor-mom", "state": "production", "capital_fraction": 0.5},
            {"pod_id": "ctrend", "state": "incubation", "capital_fraction": 0.3},
        ]
        embed = format_startup_embed(
            mode="paper",
            strategy_name="Orchestrator (2 pods)",
            symbols=["BTC/USDT"],
            capital=100000.0,
            timeframe="1D",
            pod_summaries=pods,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "Pods" in fields
        assert "anchor-mom" in fields["Pods"]
        assert "production" in fields["Pods"]
        assert "50.0%" in fields["Pods"]

    def test_startup_without_pod_summaries(self) -> None:
        embed = format_startup_embed(
            mode="paper",
            strategy_name="TSMOM",
            symbols=["BTC/USDT"],
            capital=10000.0,
            timeframe="1D",
        )
        field_names = {f["name"] for f in embed["fields"]}
        assert "Pods" not in field_names


class TestFormatShutdownEmbed:
    """format_shutdown_embed() 단위 테스트."""

    def test_basic_fields(self) -> None:
        embed = format_shutdown_embed(
            reason="SIGTERM (graceful)",
            uptime_seconds=7200.0,
            final_equity=12000.0,
            initial_capital=10000.0,
            realized_pnl=300.0,
            unrealized_pnl=200.0,
            open_positions=2,
        )
        assert embed["title"] == "MC Coin Bot Stopped"
        assert embed["color"] == 0xFFFF00  # YELLOW
        assert "timestamp" in embed

        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["Reason"] == "SIGTERM (graceful)"
        assert "2h" in fields["Uptime"]
        assert "$12,000" in fields["Final Equity"]
        assert fields["Open Positions"] == "2"

    def test_zero_uptime(self) -> None:
        embed = format_shutdown_embed(
            reason="SIGINT",
            uptime_seconds=0.0,
            final_equity=10000.0,
            initial_capital=10000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            open_positions=0,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "0m" in fields["Uptime"]

    def test_shutdown_pnl_breakdown(self) -> None:
        """Realized + Unrealized + Total + % 검증."""
        embed = format_shutdown_embed(
            reason="Graceful shutdown",
            uptime_seconds=3600.0,
            final_equity=10500.0,
            initial_capital=10000.0,
            realized_pnl=300.0,
            unrealized_pnl=200.0,
            open_positions=1,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        # Total PnL = 300 + 200 = 500, % = 500/10000 * 100 = 5.00%
        assert "$+500.00" in fields["Today PnL"]
        assert "+5.00%" in fields["Today PnL"]
        assert "$+300.00" in fields["Realized"]
        assert "$+200.00" in fields["Unrealized"]

    def test_shutdown_with_pod_summaries(self) -> None:
        pods = [
            {"pod_id": "ctrend", "state": "production", "capital_fraction": 0.6},
        ]
        embed = format_shutdown_embed(
            reason="Graceful shutdown",
            uptime_seconds=3600.0,
            final_equity=10000.0,
            initial_capital=10000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            open_positions=0,
            pod_summaries=pods,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "Pods" in fields
        assert "ctrend" in fields["Pods"]

    def test_shutdown_zero_initial_capital(self) -> None:
        """Division by zero 방어."""
        embed = format_shutdown_embed(
            reason="Graceful shutdown",
            uptime_seconds=60.0,
            final_equity=0.0,
            initial_capital=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            open_positions=0,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "+0.00%" in fields["Today PnL"]


class TestFormatCrashEmbed:
    """format_crash_embed() 단위 테스트."""

    def test_basic_fields(self) -> None:
        embed = format_crash_embed(
            error_type="RuntimeError",
            error_message="Something went wrong",
            uptime_seconds=3600.0,
        )
        assert embed["title"] == "MC Coin Bot CRASHED"
        assert embed["color"] == 0xED4245  # RED
        assert "timestamp" in embed

        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["Error Type"] == "RuntimeError"
        assert "1h" in fields["Uptime"]
        assert fields["Error"] == "Something went wrong"

    def test_long_error_message_truncated(self) -> None:
        long_msg = "x" * 300
        embed = format_crash_embed(
            error_type="ValueError",
            error_message=long_msg,
            uptime_seconds=60.0,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        error_value = fields["Error"]
        assert len(error_value) <= 204  # 200 + "..."
        assert error_value.endswith("...")

    def test_short_error_not_truncated(self) -> None:
        embed = format_crash_embed(
            error_type="OSError",
            error_message="short",
            uptime_seconds=10.0,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["Error"] == "short"

    def test_crash_with_equity_info(self) -> None:
        """equity/positions/unrealized 필드 존재."""
        embed = format_crash_embed(
            error_type="ConnectionError",
            error_message="lost connection",
            uptime_seconds=120.0,
            final_equity=9500.0,
            open_positions=2,
            unrealized_pnl=-150.5,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "$9,500" in fields["Final Equity"]
        assert fields["Open Positions"] == "2"
        assert "$-150.50" in fields["Unrealized PnL"]

    def test_crash_without_equity_info(self) -> None:
        """None → 필드 미포함 (하위 호환)."""
        embed = format_crash_embed(
            error_type="RuntimeError",
            error_message="oops",
            uptime_seconds=60.0,
        )
        field_names = {f["name"] for f in embed["fields"]}
        assert "Final Equity" not in field_names
        assert "Open Positions" not in field_names
        assert "Unrealized PnL" not in field_names
