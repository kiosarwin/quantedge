import math

import pandas as pd

from src.models.time_series import (
    compute_time_series_diagnostics,
    time_series_directional_mult,
)


def _candles_from_returns(returns_pct):
    price = 100.0
    closes = []
    for ret in returns_pct:
        price *= math.exp(ret / 100.0)
        closes.append(price)
    return pd.DataFrame({"close": closes})


def test_time_series_diagnostics_uses_returns_and_markov_state():
    candles = _candles_from_returns([0.12] * 110)

    diag = compute_time_series_diagnostics(
        candles,
        lookback=96,
        markov_window=20,
        markov_threshold_pct=1.0,
    )

    assert diag.observations >= 90
    assert diag.state in {"stable", "volatile", "unstable"}
    assert diag.markov_state == "bull"
    assert diag.markov_bull_prob > diag.markov_bear_prob
    assert diag.markov_edge > 0.0

    long_score, long_size, long_reason = time_series_directional_mult(diag, "long")
    short_score, short_size, short_reason = time_series_directional_mult(diag, "short")

    assert long_score > short_score
    assert long_size >= short_size
    assert "markov_confirms_long" in long_reason or "drift_confirms_long" in long_reason
    assert "conflicts_short" in short_reason


def test_time_series_shock_reduces_score_and_size():
    candles = _candles_from_returns([0.02] * 80 + [4.0])

    diag = compute_time_series_diagnostics(
        candles,
        lookback=96,
        shock_z_threshold=2.0,
    )

    assert diag.state == "shock"
    score_mult, size_mult, reason = time_series_directional_mult(diag, "long")

    assert score_mult < 1.0
    assert size_mult < 0.8
    assert "return_shock" in reason
