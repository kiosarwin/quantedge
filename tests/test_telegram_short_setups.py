"""Tests for Telegram surfaces of the Phase D / Liq Sweep short setups.

Validates that ``TelegramNotifier`` renders the dedicated short-setup
metadata in:

  - ``trade_opened``   — explicit Phase D / Liq Sweep label + confidence + notes
  - ``trade_closed``   — label tag attached to the close card
  - ``heartbeat``      — compact "Short edges" tally
  - ``cycle_report``   — per-position label, scan-result badge, and a
                         "Short Setup Activity" summary block

Each test stubs ``TelegramNotifier._send`` so we capture the rendered
payload string without performing network IO.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.notifications.telegram import TelegramNotifier


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_notifier(monkeypatch) -> tuple[TelegramNotifier, dict]:
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    notifier = TelegramNotifier({"telegram": {}})
    sent: dict[str, str] = {}

    async def _fake_send(message: str) -> None:
        sent["message"] = message

    notifier._send = _fake_send  # type: ignore[attr-defined]
    return notifier, sent


def _short_setup(label: str = "phase_d", confidence: float = 0.82,
                 notes: str = "BREAK_SUPPORT|VOL_SPIKE(2.50x)|BEAR_CLOSE") -> SimpleNamespace:
    """Mimic the ShortEntrySignal contract the notifier reads from `bd`."""
    return SimpleNamespace(
        label=label,
        confidence=confidence,
        notes=notes,
        is_valid=True,
    )


def _breakdown(
    *,
    direction: str = "short",
    short_setup=None,
    sm_phase: str = "neutral",
    regime: str = "distribution",
    score: float = 78.0,
    regime_ok: bool = True,
    sm_ok: bool = True,
    ev_ok: bool = True,
    symbol: str = "BTC/USDT:USDT",
) -> SimpleNamespace:
    smart_money = SimpleNamespace(
        phase=SimpleNamespace(value=sm_phase),
        score=72.0,
        direction_bias=direction,
        reasoning="test",
        aligns_with=lambda d: d == direction,
    )
    regime_obj = SimpleNamespace(
        value=regime,
        label=regime.replace("_", " ").title(),
        tradeable=True,
    )
    return SimpleNamespace(
        symbol=symbol,
        direction=direction,
        total_score=score,
        regime=regime_obj,
        smart_money=smart_money,
        ev_result=SimpleNamespace(
            p_win=0.55, ev_net_pct=0.30, confidence=0.6,
        ),
        edge_result=None,
        feature_vector=None,
        spot_context=None,
        short_setup=short_setup,
        strategy_sleeve="reversal",
        regime_ok=regime_ok,
        smart_money_ok=sm_ok,
        ev_ok=ev_ok,
        all_gates_passed=regime_ok and sm_ok and ev_ok,
    )


def _setup(direction: str = "short") -> SimpleNamespace:
    """Minimal TradeSetup duck-type for the trade_opened renderer."""
    if direction == "short":
        return SimpleNamespace(
            symbol="BTC/USDT:USDT",
            direction="short",
            entry_price=100.0,
            stop_loss=104.0,
            tp1=96.0,
            tp2=92.0,
            tp3=88.0,
            r_distance=4.0,
            risk_pct=1.0,
            size_usd=1000.0,
            tp1_size_pct=0.5,
            tp2_size_pct=0.3,
            trail_size_pct=0.2,
            exit_profile="reversal",
        )
    return SimpleNamespace(
        symbol="BTC/USDT:USDT",
        direction="long",
        entry_price=100.0,
        stop_loss=96.0,
        tp1=104.0,
        tp2=108.0,
        tp3=112.0,
        r_distance=4.0,
        risk_pct=1.0,
        size_usd=1000.0,
        tp1_size_pct=0.5,
        tp2_size_pct=0.3,
        trail_size_pct=0.2,
        exit_profile="trend_following",
    )


# ──────────────────────────────────────────────────────────────────────────────
#  trade_opened
# ──────────────────────────────────────────────────────────────────────────────

def test_trade_opened_renders_phase_d_label_and_notes(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    bd = _breakdown(short_setup=_short_setup("phase_d", 0.82))
    asyncio.run(notifier.trade_opened(bd, _setup("short"), jim_notes=None))

    msg = sent["message"]
    assert "Short Setup" in msg
    assert "PHASE D" in msg
    assert "0.82" in msg
    # Notes are rendered (truncated/sanitized in italic) when present.
    assert "BREAK_SUPPORT" in msg
    assert "VOL_SPIKE" in msg


def test_trade_opened_renders_liq_sweep_label(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    bd = _breakdown(short_setup=_short_setup("liq_sweep", 0.71,
                                             notes="SWEEP_HIGH(100.5)|UPPER_WICK(0.62)|BEAR_CLOSE"))
    asyncio.run(notifier.trade_opened(bd, _setup("short"), jim_notes=None))

    msg = sent["message"]
    assert "LIQ SWEEP" in msg
    assert "0.71" in msg
    assert "SWEEP_HIGH" in msg


def test_trade_opened_skips_short_setup_line_when_absent(monkeypatch):
    """Long trades or shorts without a dedicated detector hit must NOT
    render the Short Setup block."""
    notifier, sent = _make_notifier(monkeypatch)
    bd_long = _breakdown(direction="long", short_setup=None)
    asyncio.run(notifier.trade_opened(bd_long, _setup("long"), jim_notes=None))
    assert "Short Setup" not in sent["message"]


def test_trade_opened_skips_short_setup_line_when_invalid(monkeypatch):
    """An invalid (low-confidence) setup must not be surfaced."""
    notifier, sent = _make_notifier(monkeypatch)
    invalid = SimpleNamespace(label="phase_d", confidence=0.30, notes="x", is_valid=False)
    bd = _breakdown(short_setup=invalid)
    asyncio.run(notifier.trade_opened(bd, _setup("short"), jim_notes=None))
    assert "Short Setup" not in sent["message"]


# ──────────────────────────────────────────────────────────────────────────────
#  trade_closed
# ──────────────────────────────────────────────────────────────────────────────

def test_trade_closed_includes_label_tag(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    asyncio.run(notifier.trade_closed(
        symbol="BTC/USDT", direction="short", pnl_usd=12.5, pnl_pct=1.25,
        reason="tp1", entry=100.0, exit_price=98.7,
        short_setup_label="phase_d", short_setup_confidence=0.84,
    ))
    msg = sent["message"]
    assert "Setup:" in msg
    assert "PHASE D" in msg
    assert "0.84" in msg


def test_trade_closed_omits_label_when_absent(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    asyncio.run(notifier.trade_closed(
        symbol="BTC/USDT", direction="long", pnl_usd=-5.0, pnl_pct=-0.5,
        reason="stop_loss", entry=100.0, exit_price=99.5,
    ))
    assert "Setup:" not in sent["message"]


# ──────────────────────────────────────────────────────────────────────────────
#  heartbeat
# ──────────────────────────────────────────────────────────────────────────────

def test_heartbeat_renders_short_edge_tally(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    summary = {
        "total": 3,
        "phase_d": 2,
        "liq_sweep": 1,
        "top": [
            {"symbol": "BTC/USDT", "label": "phase_d", "confidence": 0.84},
        ],
    }
    asyncio.run(notifier.heartbeat(
        equity=100.0, drawdown_pct=1.0, daily_pnl_pct=0.5,
        open_trades=0, top_signals=["BTC long 80"],
        short_setup_summary=summary,
    ))
    msg = sent["message"]
    assert "Short edges" in msg
    assert "PHASE_D" in msg and "`2`" in msg
    assert "LIQ_SWEEP" in msg and "`1`" in msg


def test_heartbeat_skips_short_edges_when_zero(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    asyncio.run(notifier.heartbeat(
        equity=100.0, drawdown_pct=1.0, daily_pnl_pct=0.5,
        open_trades=0, top_signals=[],
        short_setup_summary={"total": 0, "phase_d": 0, "liq_sweep": 0, "top": []},
    ))
    assert "Short edges" not in sent["message"]


# ──────────────────────────────────────────────────────────────────────────────
#  cycle_report — block, per-position, per-scan
# ──────────────────────────────────────────────────────────────────────────────

def test_render_short_setup_summary_block_and_compact():
    summary = {
        "total": 2, "phase_d": 1, "liq_sweep": 1,
        "top": [
            {"symbol": "ETH/USDT", "label": "phase_d", "confidence": 0.78},
            {"symbol": "SOL/USDT", "label": "liq_sweep", "confidence": 0.69},
        ],
    }
    block = TelegramNotifier._render_short_setup_summary_block(summary)
    text = "\n".join(block)
    assert "Short Setup Activity" in text
    assert "PHASE_D × 1" in text
    assert "LIQ_SWEEP × 1" in text
    assert "ETH" in text and "phase_d" in text
    assert "SOL" in text and "liq_sweep" in text

    compact = TelegramNotifier._render_short_setup_compact(summary)
    assert "Short edges" in compact
    assert "PHASE_D" in compact and "LIQ_SWEEP" in compact

    # Empty summary collapses to []/empty string.
    assert TelegramNotifier._render_short_setup_summary_block(None) == []
    assert TelegramNotifier._render_short_setup_summary_block({"total": 0}) == []
    assert TelegramNotifier._render_short_setup_compact(None) == ""
    assert TelegramNotifier._render_short_setup_compact({"total": 0}) == ""


def test_cycle_report_tags_open_position_with_short_setup(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    floating = [{
        "symbol": "BTC/USDT:USDT",
        "direction": "short",
        "entry": 100.0,
        "current": 98.0,
        "sl": 104.0,
        "tp1": 96.0,
        "size_usd": 500.0,
        "risk_pct": 1.0,
        "risk_usd": 5.0,
        "pnl_usd": 9.5,
        "pnl_pct": 0.95,
        "elapsed_s": 600.0,
        "opened_at": 0.0,
        "strategy_sleeve": "reversal",
        "exit_profile": "reversal",
        "short_setup_label": "phase_d",
        "short_setup_confidence": 0.85,
    }]
    asyncio.run(notifier.cycle_report(
        breakdowns=[],
        equity=100.0,
        drawdown_pct=0.0,
        daily_pnl_pct=0.0,
        open_trades=1,
        cycle_num=1,
        floating_positions=floating,
        open_positions=floating,
        starting_equity=100.0,
    ))
    msg = sent["message"]
    assert "Short Setup `phase_d`" in msg
    assert "0.85" in msg


def test_cycle_report_renders_summary_block_when_breakdowns_have_setups(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    bd1 = _breakdown(symbol="BTC/USDT:USDT",
                     short_setup=_short_setup("phase_d", 0.84))
    bd2 = _breakdown(symbol="ETH/USDT:USDT",
                     short_setup=_short_setup("liq_sweep", 0.71))
    summary = {
        "total": 2, "phase_d": 1, "liq_sweep": 1,
        "top": [
            {"symbol": "BTC/USDT:USDT", "label": "phase_d", "confidence": 0.84},
            {"symbol": "ETH/USDT:USDT", "label": "liq_sweep", "confidence": 0.71},
        ],
    }
    asyncio.run(notifier.cycle_report(
        breakdowns=[bd1, bd2],
        equity=100.0,
        drawdown_pct=0.0,
        daily_pnl_pct=0.0,
        open_trades=0,
        cycle_num=1,
        starting_equity=100.0,
        regime_thresholds={"distribution": 60},
        short_setup_summary=summary,
    ))
    msg = sent["message"]
    # Section header
    assert "Short Setup Activity" in msg
    # Per-scan badge appended on the scan-result rows
    assert "phase_d" in msg
    assert "liq_sweep" in msg


def test_cycle_report_omits_short_setup_section_when_summary_empty(monkeypatch):
    notifier, sent = _make_notifier(monkeypatch)
    bd = _breakdown(symbol="BTC/USDT:USDT", short_setup=None)
    asyncio.run(notifier.cycle_report(
        breakdowns=[bd],
        equity=100.0,
        drawdown_pct=0.0,
        daily_pnl_pct=0.0,
        open_trades=0,
        cycle_num=1,
        starting_equity=100.0,
        regime_thresholds={"distribution": 60},
        short_setup_summary=None,
    ))
    assert "Short Setup Activity" not in sent["message"]
