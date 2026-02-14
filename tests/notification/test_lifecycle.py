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
        assert embed["title"] == "Bot Started"
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


class TestFormatShutdownEmbed:
    """format_shutdown_embed() 단위 테스트."""

    def test_basic_fields(self) -> None:
        embed = format_shutdown_embed(
            reason="SIGTERM (graceful)",
            uptime_seconds=7200.0,
            final_equity=12000.0,
            today_pnl=500.0,
            open_positions=2,
        )
        assert embed["title"] == "Bot Stopped"
        assert embed["color"] == 0xFFFF00  # YELLOW
        assert "timestamp" in embed

        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["Reason"] == "SIGTERM (graceful)"
        assert "2h" in fields["Uptime"]
        assert "$12,000" in fields["Final Equity"]
        assert "$+500" in fields["Today PnL"]
        assert fields["Open Positions"] == "2"

    def test_zero_uptime(self) -> None:
        embed = format_shutdown_embed(
            reason="SIGINT",
            uptime_seconds=0.0,
            final_equity=10000.0,
            today_pnl=0.0,
            open_positions=0,
        )
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert "0m" in fields["Uptime"]


class TestFormatCrashEmbed:
    """format_crash_embed() 단위 테스트."""

    def test_basic_fields(self) -> None:
        embed = format_crash_embed(
            error_type="RuntimeError",
            error_message="Something went wrong",
            uptime_seconds=3600.0,
        )
        assert embed["title"] == "Bot CRASHED"
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
