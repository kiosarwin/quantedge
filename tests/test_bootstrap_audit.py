from types import SimpleNamespace

from src.reporting.bootstrap import build_bootstrap_audit_report, format_bootstrap_audit_lines


def _trade(symbol: str, direction: str, pnl_usd: float, pnl_pct: float, session: str, opened_at: float, closed_at: float, tp1_hit: bool = False):
    return SimpleNamespace(
        symbol=symbol,
        direction=direction,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        session=session,
        opened_at=opened_at,
        closed_at=closed_at,
        tp1_hit=tp1_hit,
        reason="tp2",
        strategy_sleeve="reversal" if direction == "short" else "trend_following",
        asset=symbol.split("/")[0],
        timeframe="1h",
        entry_reason="bootstrap sample",
        lifecycle_status="RESEARCH",
    )


def test_bootstrap_audit_report_includes_direction_mix_session_matrix_and_edge_state():
    trades = [
        _trade("BTC/USDT:USDT", "long", 1.25, 1.5, "asia", 100.0, 200.0, tp1_hit=True),
        _trade("ETH/USDT:USDT", "short", -0.75, -0.9, "london", 150.0, 260.0),
        _trade("BCH/USDT:USDT", "short", 0.50, 0.6, "london", 180.0, 300.0, tp1_hit=True),
    ]

    edge_memory = SimpleNamespace(
        total_samples=lambda: 12,
        summary_rows=lambda limit=3: [
            {
                "pair": "BCH/USDT:USDT",
                "edge_type": "trending_expansion|neutral|short",
                "sample_size": 6,
                "overall_wr": 0.667,
                "recent_wr": 0.5,
                "regime_coverage": "2/4",
            }
        ],
    )
    lifecycle_report = {
        "counts": {"RESEARCH": 1, "PAPER_VALIDATION": 1, "SMALL_LIVE": 0, "ACTIVE": 0, "DEGRADED": 0, "DISABLED": 0},
        "recommendation": "PAPER_ONLY",
    }

    report = build_bootstrap_audit_report(trades, lifecycle_report=lifecycle_report, edge_memory=edge_memory)
    lines = format_bootstrap_audit_lines(report)

    joined = "\n".join(lines)
    assert "Phase:" in joined
    assert "Mix: `LONG`" in joined or "Mix: LONG" in joined
    assert "Session matrix open→close" in joined
    assert "Edge memory: `12` samples" in joined
    assert "BCH/USDT:USDT" in joined
    assert "Cohorts:" in joined


def test_bootstrap_audit_report_marks_small_sample_bootstrap():
    trades = [
        _trade("BTC/USDT:USDT", "long", 1.0, 1.0, "asia", 100.0, 110.0),
    ]
    report = build_bootstrap_audit_report(trades, lifecycle_report={"counts": {}, "recommendation": "PAPER_ONLY"})
    lines = format_bootstrap_audit_lines(report)

    assert any("BOOTSTRAP" in line for line in lines)
    assert any("N `1`" in line for line in lines)
