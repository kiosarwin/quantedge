"""
Tests for the pydantic config-validation gate.

Verifies that:
  * The production ``config/config.yaml`` validates cleanly (a tripwire — if
    this fails, the bot will not start and we want to catch that in CI before
    the operator).
  * Identity preservation: ``validate_config(cfg) is cfg`` so existing dict-
    access call sites in main.py / scorer.py / risk_manager.py keep working.
  * Each cross-field consistency rule rejects its specific typo class with a
    localised, human-readable error.
  * Each bounded-range typo class is caught (kelly cap inflated by 10x,
    drawdown cap inflated by 10x, exit sizing not summing to 1.0, ...).
  * Lax sections accept extra keys (regression guard against accidentally
    flipping a section to strict mode and breaking additive YAML changes).
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.config import (
    ConfigValidationError,
    NinjaConfig,
    validate_config,
)


# ---------------------------------------------------------------------------
#  Fixtures                                                                   
# ---------------------------------------------------------------------------


@pytest.fixture
def production_cfg() -> dict:
    """The actual config/config.yaml. Mutate via ``copy.deepcopy`` per test."""
    path = Path(__file__).parent.parent / "config" / "config.yaml"
    return yaml.safe_load(path.read_text())


@pytest.fixture
def cfg(production_cfg) -> dict:
    """Per-test deep copy of the production config so mutations don't leak."""
    return copy.deepcopy(production_cfg)


# ---------------------------------------------------------------------------
#  Tripwire & identity                                                         
# ---------------------------------------------------------------------------


def test_production_config_validates_clean(production_cfg):
    """If this fails the bot will not start. CI must catch it before the operator."""
    out = validate_config(copy.deepcopy(production_cfg))
    assert out is not None  # validate_config raises on failure


def test_validate_config_returns_same_dict_object(cfg):
    """Identity-preserving so existing dict-access call sites keep working."""
    out = validate_config(cfg)
    assert out is cfg
    assert out["risk"]["risk_per_trade_pct"] == cfg["risk"]["risk_per_trade_pct"]


def test_pydantic_model_can_be_built_directly(cfg):
    """Direct model construction must also accept the production config."""
    NinjaConfig.model_validate(cfg)  # would raise on failure


# ---------------------------------------------------------------------------
#  Bounded-range typos (single-field)                                          
# ---------------------------------------------------------------------------


def test_kelly_max_kelly_pct_inflated_10x_is_rejected(cfg):
    """The canonical 'blew up the account' typo: 4.0 -> 40."""
    cfg["kelly"]["max_kelly_pct"] = 40.0
    with pytest.raises(ConfigValidationError, match="kelly.max_kelly_pct"):
        validate_config(cfg)


def test_risk_max_drawdown_pct_inflated_10x_is_rejected(cfg):
    """25.0 -> 250 typo (drawdown cap is in percent so this is impossible)."""
    cfg["risk"]["max_drawdown_pct"] = 250.0
    with pytest.raises(ConfigValidationError, match="risk.max_drawdown_pct"):
        validate_config(cfg)


def test_zero_paper_starting_equity_is_rejected(cfg):
    cfg["trading"]["paper_starting_equity"] = 0
    with pytest.raises(ConfigValidationError, match="paper_starting_equity"):
        validate_config(cfg)


def test_negative_recv_window_is_rejected(cfg):
    cfg["exchange"]["recv_window"] = -1
    with pytest.raises(ConfigValidationError, match="recv_window"):
        validate_config(cfg)


def test_leverage_above_binance_ceiling_is_rejected(cfg):
    """Binance USDT-perp ceiling is 125x."""
    cfg["risk"]["max_leverage"] = 200
    cfg["risk"]["default_leverage"] = 200
    with pytest.raises(ConfigValidationError, match="max_leverage"):
        validate_config(cfg)


def test_unknown_trading_mode_is_rejected(cfg):
    cfg["trading"]["mode"] = "yolo"
    with pytest.raises(ConfigValidationError, match="trading.mode"):
        validate_config(cfg)


def test_min_p_win_above_one_is_rejected(cfg):
    cfg["ev_model"]["min_p_win"] = 1.5
    with pytest.raises(ConfigValidationError, match="ev_model.min_p_win"):
        validate_config(cfg)


def test_negative_kelly_fraction_is_rejected(cfg):
    cfg["kelly"]["fraction"] = -0.1
    with pytest.raises(ConfigValidationError, match="kelly.fraction"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
#  Cross-field consistency rules                                               
# ---------------------------------------------------------------------------


def test_max_risk_per_trade_below_risk_per_trade_is_rejected(cfg):
    cfg["risk"]["risk_per_trade_pct"] = 1.5
    cfg["risk"]["max_risk_per_trade_pct"] = 1.0
    with pytest.raises(ConfigValidationError, match="max_risk_per_trade_pct"):
        validate_config(cfg)


def test_default_leverage_above_max_leverage_is_rejected(cfg):
    cfg["risk"]["max_leverage"] = 5
    cfg["risk"]["default_leverage"] = 10
    with pytest.raises(ConfigValidationError, match="default_leverage"):
        validate_config(cfg)


def test_max_symbol_risk_below_max_per_trade_is_rejected(cfg):
    """Catches 'single-symbol cap silently blocks every entry'."""
    cfg["risk"]["max_risk_per_trade_pct"] = 1.75
    cfg["risk"]["max_symbol_risk_pct"] = 1.0
    with pytest.raises(ConfigValidationError, match="max_symbol_risk_pct"):
        validate_config(cfg)


def test_exit_sizes_not_summing_to_one_is_rejected(cfg):
    cfg["exit"]["tp1_size_pct"] = 0.50
    cfg["exit"]["tp2_size_pct"] = 0.50
    cfg["exit"]["trail_size_pct"] = 0.50  # sum = 1.5
    with pytest.raises(ConfigValidationError, match=r"sum to.*1\.0"):
        validate_config(cfg)


def test_tp2_not_greater_than_tp1_is_rejected(cfg):
    cfg["exit"]["tp1_r_multiple"] = 2.0
    cfg["exit"]["tp2_r_multiple"] = 1.5
    with pytest.raises(ConfigValidationError, match=r"tp2_r_multiple.*>.*tp1_r_multiple"):
        validate_config(cfg)


def test_kelly_min_above_max_is_rejected(cfg):
    cfg["kelly"]["min_kelly_pct"] = 5.0
    cfg["kelly"]["max_kelly_pct"] = 4.0
    with pytest.raises(ConfigValidationError, match="min_kelly_pct"):
        validate_config(cfg)


def test_ev_max_kelly_above_kelly_max_is_rejected(cfg):
    cfg["kelly"]["max_kelly_pct"] = 4.0
    cfg["ev_model"]["max_kelly_pct"] = 5.0
    with pytest.raises(ConfigValidationError, match="ev_model.max_kelly_pct"):
        validate_config(cfg)


def test_live_mode_with_testnet_true_is_rejected(cfg):
    """Catches the canonical 'switched to live but forgot to flip testnet' typo."""
    cfg["trading"]["mode"] = "live"
    cfg["exchange"]["testnet"] = True
    with pytest.raises(ConfigValidationError, match="incompatible with .*testnet"):
        validate_config(cfg)


def test_live_mode_with_testnet_false_is_accepted(cfg):
    cfg["trading"]["mode"] = "live"
    cfg["exchange"]["testnet"] = False
    out = validate_config(cfg)
    assert out is cfg


# ---------------------------------------------------------------------------
#  Safety section — session multipliers                                        
# ---------------------------------------------------------------------------


def test_session_multiplier_zero_is_rejected(cfg):
    cfg["safety"]["session_size_multipliers"]["asia"] = 0.0
    with pytest.raises(ConfigValidationError, match="session_size_multipliers.asia"):
        validate_config(cfg)


def test_session_multiplier_above_five_is_rejected(cfg):
    cfg["safety"]["session_size_multipliers"]["ny"] = 6.0
    with pytest.raises(ConfigValidationError, match="session_size_multipliers.ny"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
#  Multiple issues collected in one pass                                       
# ---------------------------------------------------------------------------


def test_multiple_issues_all_surface_in_message(cfg):
    cfg["kelly"]["max_kelly_pct"] = 40.0
    cfg["risk"]["max_drawdown_pct"] = 250.0
    cfg["exit"]["tp1_size_pct"] = 0.5
    cfg["exit"]["tp2_size_pct"] = 0.5
    cfg["exit"]["trail_size_pct"] = 0.2  # sum = 1.2
    cfg["trading"]["paper_starting_equity"] = 0
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(cfg)
    msg = str(exc_info.value)
    # All four issues must be surfaced — operators should see the whole list,
    # not the first failure only.
    assert "kelly.max_kelly_pct" in msg
    assert "max_drawdown_pct" in msg
    assert "sum to" in msg
    assert "paper_starting_equity" in msg


# ---------------------------------------------------------------------------
#  Lax sections — additive keys must pass through                              
# ---------------------------------------------------------------------------


def test_extra_keys_in_indicators_section_pass_through(cfg):
    cfg["indicators"]["new_experimental_period"] = 99
    out = validate_config(cfg)
    assert out["indicators"]["new_experimental_period"] == 99


def test_unknown_top_level_section_pass_through(cfg):
    cfg["future_feature_flags"] = {"enable_x": True}
    out = validate_config(cfg)
    assert out["future_feature_flags"]["enable_x"] is True


def test_missing_required_top_level_section_is_rejected(cfg):
    """Required sections (exchange/trading/risk/exit/kelly/ev_model/safety) cannot be missing."""
    del cfg["risk"]
    with pytest.raises(ConfigValidationError, match="risk"):
        validate_config(cfg)
