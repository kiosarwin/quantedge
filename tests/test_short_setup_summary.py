"""Unit test for ``NinjaTrader._short_setup_summary``.

Exercised in isolation so we don't have to import the full ``src.main``
module (and its heavy dotenv/rich/ccxt dependency chain). The summary is a
``staticmethod`` that walks the per-scan breakdowns and emits the dict
consumed by ``TelegramNotifier.cycle_report`` / ``heartbeat``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_short_setup_summary():
    """Lift just the ``_short_setup_summary`` static method out of main.py.

    Importing ``src.main`` requires a long list of optional packages
    (dotenv, rich, ccxt, tenacity, ...). The helper itself only uses
    ``getattr`` and stdlib types, so we extract it via ``ast`` and exec it
    in a fresh namespace.
    """
    main_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    source = main_path.read_text(encoding="utf-8")
    import ast

    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "_short_setup_summary":
            # Strip the @staticmethod decorator so we can exec the def
            # directly without needing the enclosing class.
            node.decorator_list = []
            wrapper = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(wrapper, str(main_path), "exec"), ns)
            return ns["_short_setup_summary"]
    raise RuntimeError("_short_setup_summary not found in src/main.py")


_short_setup_summary = _load_short_setup_summary()


def _bd(symbol: str, label: str | None, confidence: float = 0.7,
        is_valid: bool = True) -> SimpleNamespace:
    setup = None
    if label is not None:
        setup = SimpleNamespace(
            label=label,
            confidence=confidence,
            is_valid=is_valid,
        )
    return SimpleNamespace(symbol=symbol, short_setup=setup)


def test_summary_is_empty_for_empty_or_no_setups():
    assert _short_setup_summary([]) == {
        "total": 0, "phase_d": 0, "liq_sweep": 0, "top": [],
    }
    bds = [_bd("BTC", None), _bd("ETH", None)]
    assert _short_setup_summary(bds)["total"] == 0


def test_summary_counts_phase_d_and_liq_sweep():
    bds = [
        _bd("BTC", "phase_d", 0.80),
        _bd("ETH", "phase_d", 0.72),
        _bd("SOL", "liq_sweep", 0.66),
        _bd("DOGE", None),
    ]
    summary = _short_setup_summary(bds)
    assert summary["total"] == 3
    assert summary["phase_d"] == 2
    assert summary["liq_sweep"] == 1


def test_summary_top_is_sorted_by_confidence_descending_and_capped_at_three():
    bds = [
        _bd("A", "phase_d", 0.55),
        _bd("B", "phase_d", 0.90),
        _bd("C", "liq_sweep", 0.75),
        _bd("D", "liq_sweep", 0.65),
        _bd("E", "phase_d", 0.80),
    ]
    summary = _short_setup_summary(bds)
    top = summary["top"]
    assert len(top) == 3
    assert [row["symbol"] for row in top] == ["B", "E", "C"]
    assert top[0]["confidence"] == 0.90


def test_summary_skips_invalid_or_blank_label():
    bds = [
        _bd("X", "phase_d", 0.80, is_valid=False),   # is_valid=False → skip
        _bd("Y", "", 0.80),                          # blank label  → skip
        _bd("Z", "none", 0.80),                      # placeholder  → skip
        _bd("OK", "liq_sweep", 0.71),                # counts
    ]
    summary = _short_setup_summary(bds)
    assert summary["total"] == 1
    assert summary["liq_sweep"] == 1
    assert summary["phase_d"] == 0
    assert summary["top"][0]["symbol"] == "OK"
