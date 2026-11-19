"""
Unit tests for Assignment 11 — Rate Limiter, Audit Log, Monitoring.

Run after implementing the three modules:
    pytest tests/public/test_assignment_units.py -q
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest


class TestRateLimiter:
    """Tests for RateLimitPlugin."""

    @pytest.fixture
    def plugin(self):
        from assignment.rate_limiter import RateLimitPlugin
        return RateLimitPlugin(max_requests=1, window_seconds=60)

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock()
        ctx.user_id = "user-123"
        return ctx

    @pytest.mark.asyncio
    async def test_same_user_blocked_on_second_request(self, plugin, mock_context):
        """Same user blocked at time=1 when max_requests=1, window=60."""
        with patch("assignment.rate_limiter.time.monotonic", side_effect=[0, 1]):
            # First request at t=0 - should allow
            result1 = await plugin.on_user_message_callback(
                invocation_context=mock_context, user_message="hello"
            )
            assert result1 is None

            # Second request at t=1 - should block
            result2 = await plugin.on_user_message_callback(
                invocation_context=mock_context, user_message="hello again"
            )
            assert result2 is not None
            assert "Rate limit exceeded" in result2.parts[0].text
            assert plugin.blocked_count == 1

    @pytest.mark.asyncio
    async def test_different_user_allowed(self, plugin, mock_context):
        """Different user allowed at time=1."""
        with patch("assignment.rate_limiter.time.monotonic", side_effect=[0, 1]):
            # First user at t=0
            await plugin.on_user_message_callback(
                invocation_context=mock_context, user_message="hello"
            )

            # Different user at t=1
            other_context = MagicMock()
            other_context.user_id = "user-456"
            result = await plugin.on_user_message_callback(
                invocation_context=other_context, user_message="hello"
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_same_user_allowed_after_window(self, plugin, mock_context):
        """Same user allowed at time=61 (window expired)."""
        with patch("assignment.rate_limiter.time.monotonic", side_effect=[0, 61]):
            # First request at t=0
            await plugin.on_user_message_callback(
                invocation_context=mock_context, user_message="hello"
            )

            # Second request at t=61 - window expired, should allow
            result = await plugin.on_user_message_callback(
                invocation_context=mock_context, user_message="hello again"
            )
            assert result is None


class TestAuditLog:
    """Tests for AuditLogPlugin."""

    @pytest.fixture
    def plugin(self):
        from assignment.audit_log import AuditLogPlugin
        return AuditLogPlugin()

    def test_record_input_and_output_correlation(self, plugin):
        """Input and output records correlate via request_id."""
        request_id = "req-123"
        user_id = "user-123"

        # Record input
        plugin.record_input(user_id=user_id, text="Hello", request_id=request_id)

        # Record output
        plugin.record_output(
            user_id=user_id, text="Hi there!", blocked=False, layer="input_guard", request_id=request_id
        )

        # Verify log entry
        assert len(plugin.logs) == 1
        log = plugin.logs[0]
        assert log["request_id"] == request_id
        assert log["user_id"] == user_id
        assert log["input"] == "Hello"
        assert log["output"] == "Hi there!"
        assert log["blocked"] is False
        assert log["layer"] == "input_guard"
        assert log["latency_ms"] >= 0
        assert "timestamp" in log

    def test_record_output_without_input_uses_user_id_fallback(self, plugin):
        """record_output works even without prior record_input (user_id fallback)."""
        user_id = "user-456"
        plugin.record_output(
            user_id=user_id, text="Response", blocked=True, layer="output_guard"
        )

        assert len(plugin.logs) == 1
        log = plugin.logs[0]
        assert log["user_id"] == user_id
        assert log["request_id"] == user_id  # fallback
        assert log["latency_ms"] >= 0

    def test_export_json_creates_file_and_dirs(self, plugin, tmp_path):
        """export_json writes UTF-8 JSON and creates parent directories."""
        plugin.record_input(user_id="u1", text="in", request_id="r1")
        plugin.record_output(user_id="u1", text="out", request_id="r1")

        filepath = tmp_path / "subdir" / "audit.json"
        plugin.export_json(str(filepath))

        assert filepath.exists()
        import json
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1


class TestMonitoringAlert:
    """Tests for MonitoringAlert."""

    @pytest.fixture
    def monitor(self):
        from assignment.monitoring import MonitoringAlert
        return MonitoringAlert(
            block_rate_threshold=0.5,
            rate_limit_hit_threshold=5,
            judge_fail_rate_threshold=0.3,
        )

    def test_block_rate_alert(self, monitor):
        """Alert fires when block_rate exceeds threshold."""
        monitor.total_requests = 10
        monitor.blocked_requests = 6  # 60% block rate > 50%

        alerts = monitor.check_metrics()
        assert len(alerts) == 1
        assert alerts[0].metric == "block_rate"
        assert alerts[0].value == 0.6
        assert alerts[0].threshold == 0.5

    def test_rate_limit_hits_alert(self, monitor):
        """Alert fires when rate_limit_hits exceeds threshold."""
        monitor.rate_limit_hits = 6  # > 5

        alerts = monitor.check_metrics()
        assert len(alerts) == 1
        assert alerts[0].metric == "rate_limit_hits"
        assert alerts[0].value == 6
        assert alerts[0].threshold == 5

    def test_judge_fail_rate_alert(self, monitor):
        """Alert fires when judge_fail_rate exceeds threshold."""
        monitor.judge_checks = 10
        monitor.judge_fails = 4  # 40% > 30%

        alerts = monitor.check_metrics()
        assert len(alerts) == 1
        assert alerts[0].metric == "judge_fail_rate"
        assert alerts[0].value == 0.4
        assert alerts[0].threshold == 0.3

    def test_no_alerts_when_below_thresholds(self, monitor):
        """No alerts when all metrics are below thresholds."""
        monitor.total_requests = 10
        monitor.blocked_requests = 3  # 30% < 50%
        monitor.rate_limit_hits = 3  # < 5
        monitor.judge_checks = 10
        monitor.judge_fails = 2  # 20% < 30%

        alerts = monitor.check_metrics()
        assert len(alerts) == 0

    def test_check_metrics_replaces_alerts_not_accumulate(self, monitor):
        """Repeated check_metrics replaces alerts, doesn't accumulate duplicates."""
        monitor.total_requests = 10
        monitor.blocked_requests = 6

        # First call
        alerts1 = monitor.check_metrics()
        assert len(alerts1) == 1

        # Second call with same metrics
        alerts2 = monitor.check_metrics()
        assert len(alerts2) == 1
        assert len(monitor.alerts) == 1  # Not 2

        # Third call with different metrics - should replace
        monitor.blocked_requests = 3
        alerts3 = monitor.check_metrics()
        assert len(alerts3) == 0
        assert len(monitor.alerts) == 0

    def test_export_json_writes_snapshot(self, monitor, tmp_path):
        """export_json writes snapshot to JSON."""
        monitor.total_requests = 10
        monitor.blocked_requests = 5
        monitor.check_metrics()

        filepath = tmp_path / "metrics.json"
        monitor.export_json(str(filepath))

        assert filepath.exists()
        import json
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["total_requests"] == 10
        assert data["blocked_requests"] == 5
        assert "alerts" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])