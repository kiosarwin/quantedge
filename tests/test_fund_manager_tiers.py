from src.models.fund_manager import FundManager


def test_fund_manager_keeps_four_open_trades_at_fresh_75_equity(monkeypatch, tmp_path):
    monkeypatch.setattr("src.models.fund_manager.STATE_PATH", tmp_path / "fund_manager_state.json")
    cfg = {
        "trading": {"max_open_trades": 4},
        "risk": {"max_leverage": 5, "risk_per_trade_pct": 1.0},
    }
    fund_manager = FundManager(cfg)

    changes = fund_manager.update_tiers(75.0)

    assert cfg["trading"]["max_open_trades"] == 4
    assert cfg["risk"]["max_leverage"] == 5
    assert cfg["risk"]["risk_per_trade_pct"] == 1.5
    assert any("risk_per_trade" in change for change in changes)
