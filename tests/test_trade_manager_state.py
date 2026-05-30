"""Tests for TradeManager._load_state() graceful recovery from corrupted state files."""
import json
from pathlib import Path
from unittest.mock import MagicMock

from src.execution.trade_manager import TradeManager


def _make_trade_manager(state_path: Path) -> TradeManager:
    """Instantiate TradeManager with mocked dependencies and overridden STATE_PATH."""
    client = MagicMock()
    executor = MagicMock()
    risk = MagicMock()
    risk.can_open_trade.return_value = (True, "ok")
    risk.check_trade_exposure.return_value = (True, "ok")
    cfg = {
        "exit": {
            "tp1_r_multiple": 1.5,
            "tp2_r_multiple": 2.0,
            "tp1_size_pct": 0.50,
            "tp2_size_pct": 0.30,
            "trail_size_pct": 0.20,
            "breakeven_trigger_r": 1.0,
            "trailing_atr_multiplier": 1.5,
            "max_hold_duration_s": 172800,
        },
    }
    return TradeManager(client, executor, risk, cfg)


def test_load_state_corrupted_json(tmp_path, monkeypatch):
    """Corrupted JSON should not crash; trades dict should be empty."""
    state_file = tmp_path / "open_trades.json"
    state_file.write_text("{invalid json content!!!")
    monkeypatch.setattr(TradeManager, "STATE_PATH", state_file)

    tm = _make_trade_manager(state_file)
    assert tm._trades == {}


def test_load_state_empty_file(tmp_path, monkeypatch):
    """Empty file should not crash; trades dict should be empty."""
    state_file = tmp_path / "open_trades.json"
    state_file.write_text("")
    monkeypatch.setattr(TradeManager, "STATE_PATH", state_file)

    tm = _make_trade_manager(state_file)
    assert tm._trades == {}


def test_load_state_valid_json_malformed_trades(tmp_path, monkeypatch):
    """Valid JSON but with malformed trade entries (missing fields) should recover gracefully."""
    state_file = tmp_path / "open_trades.json"
    # Missing required fields like 'setup', 'entry_order_id', etc.
    data = {"trades": [{"bad_key": "bad_value"}, {"setup": "not_a_dict"}]}
    state_file.write_text(json.dumps(data))
    monkeypatch.setattr(TradeManager, "STATE_PATH", state_file)

    tm = _make_trade_manager(state_file)
    # Malformed entries should be skipped; no crash
    assert isinstance(tm._trades, dict)


def test_load_state_valid_json_wrong_types(tmp_path, monkeypatch):
    """Valid JSON with wrong types for trade fields should not crash."""
    state_file = tmp_path / "open_trades.json"
    data = {
        "trades": [
            {
                "setup": {
                    "symbol": 12345,  # should be string
                    "direction": None,  # should be string
                    "entry_price": "not_a_float",
                },
                "entry_order_id": 999,
                "sl_order_id": None,
                "tp1_order_id": False,
                "tp2_order_id": [],
            }
        ]
    }
    state_file.write_text(json.dumps(data))
    monkeypatch.setattr(TradeManager, "STATE_PATH", state_file)

    tm = _make_trade_manager(state_file)
    # Should not crash; invalid entries are skipped
    assert isinstance(tm._trades, dict)


def test_load_state_nonexistent_file(tmp_path, monkeypatch):
    """Non-existent file should not crash; trades dict should be empty."""
    state_file = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(TradeManager, "STATE_PATH", state_file)

    tm = _make_trade_manager(state_file)
    assert tm._trades == {}
