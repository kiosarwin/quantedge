from src.learning.learner import TradeRecord
from src.reporting.trading_brain import (
    TradingBrainJournal,
    build_trading_brain_report,
    format_closed_trade_markdown,
)


def _trade(symbol="BTC/USDT:USDT", pnl=1.0, sleeve="trend_following", sector="l1"):
    return TradeRecord(
        symbol=symbol,
        direction="long",
        entry_price=100.0,
        exit_price=101.0,
        pnl_usd=pnl,
        pnl_pct=pnl,
        reason="tp1",
        scores={
            "score": 68.0,
            "setup_type": "trend_continuation",
            "strategy_sleeve": sleeve,
            "sector": sector,
            "market_risk_on_state": "risk_on",
            "market_rotation_state": "alts_leading",
            "sector_rotation_state": "rotating_in",
        },
        opened_at=1_700_000_000.0,
        closed_at=1_700_000_600.0,
        regime="trending_expansion",
        strategy_sleeve=sleeve,
        sector=sector,
        market_risk_on_state="risk_on",
        market_rotation_state="alts_leading",
        market_btc_trend="up",
        market_eth_btc_trend="up",
        market_context_confidence=0.8,
    )


def test_trading_brain_report_is_diagnostic_summary():
    report = build_trading_brain_report(
        [_trade(pnl=2.0), _trade(symbol="ETH/USDT:USDT", pnl=-1.0)],
        min_trades=1,
    )

    assert report["trades"] == 2
    assert report["sample_ready"] is True
    assert report["attribution"]["by_sleeve"]
    assert report["top_strengths"]


def test_closed_trade_markdown_contains_context():
    text = format_closed_trade_markdown(_trade(), trade_count=3)

    assert "BTC/USDT:USDT LONG Closed" in text
    assert "Market Context" in text
    assert "rotating_in" in text
    assert "diagnostic" not in text.lower()


def test_trading_brain_journal_writes_local_artifacts(tmp_path):
    cfg = {
        "trading_brain": {
            "enabled": True,
            "output_dir": str(tmp_path / "brain"),
            "min_trades": 1,
        }
    }
    journal = TradingBrainJournal(cfg)
    trades = [_trade()]

    report = journal.record_closed_trade(trades[0], trades)

    assert report["trades"] == 1
    assert (tmp_path / "brain" / "analysis" / "latest-edge-report.md").exists()
    closed_files = list((tmp_path / "brain" / "trades" / "closed").glob("*.md"))
    assert len(closed_files) == 1
    assert "Operating Note" in (tmp_path / "brain" / "analysis" / "latest-edge-report.md").read_text()
