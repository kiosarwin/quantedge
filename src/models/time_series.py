"""
Causal time-series diagnostics for live trading decisions.

This module intentionally avoids heavy ARIMA/GARCH fitting in the scan loop.
It implements the practical parts that are safe for realtime use:

* use log returns instead of raw prices;
* estimate current volatility with EWMA;
* detect volatility shocks and unstable return distributions;
* produce soft score/size multipliers rather than hard forecasts.

The output is metadata for the trading stack, not a standalone signal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesDiagnostics:
    observations: int = 0
    state: str = "insufficient"  # insufficient | stable | volatile | shock | unstable
    log_return_mean_pct: float = 0.0
    realized_vol_pct: float = 0.0
    ewma_vol_pct: float = 0.0
    volatility_ratio: float = 1.0
    latest_abs_return_z: float = 0.0
    drift_t_stat: float = 0.0
    stability_score: float = 0.0
    markov_state: str = "unknown"
    markov_bull_prob: float = 0.0
    markov_bear_prob: float = 0.0
    markov_sideways_prob: float = 0.0
    markov_edge: float = 0.0
    markov_transition_count: int = 0
    score_mult: float = 1.0
    size_mult: float = 1.0
    reason: str = "insufficient_data"


def unavailable_diagnostics(reason: str = "insufficient_data") -> TimeSeriesDiagnostics:
    return TimeSeriesDiagnostics(reason=reason)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def compute_time_series_diagnostics(
    candles: pd.DataFrame,
    *,
    lookback: int = 96,
    ewma_lambda: float = 0.94,
    min_returns: int = 30,
    shock_z_threshold: float = 2.8,
    vol_ratio_threshold: float = 1.65,
    markov_window: int = 20,
    markov_threshold_pct: float = 2.0,
) -> TimeSeriesDiagnostics:
    """Compute causal return/volatility diagnostics from OHLCV candles.

    The function uses only current and historical candles. It does not smooth
    with future data and does not fit in-sample forecasts, so it is safe to use
    in live scans.
    """
    if candles is None or candles.empty or "close" not in candles:
        return unavailable_diagnostics("missing_close_series")

    close = pd.to_numeric(candles["close"], errors="coerce").dropna()
    close = close[close > 0]
    if len(close) < min_returns + 1:
        return unavailable_diagnostics("not_enough_returns")

    close = close.tail(max(min_returns + 1, int(lookback) + 1))
    returns = (close.apply(math.log).diff().dropna() * 100.0)
    returns = returns.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(returns) < min_returns:
        return unavailable_diagnostics("not_enough_valid_returns")

    realized_vol = _safe_float(returns.std(ddof=1), 0.0)
    if realized_vol <= 0.0:
        return TimeSeriesDiagnostics(
            observations=int(len(returns)),
            state="stable",
            stability_score=1.0,
            reason="flat_returns",
        )

    alpha = 1.0 - max(0.01, min(0.99, float(ewma_lambda)))
    ewma_var = _safe_float((returns.pow(2).ewm(alpha=alpha, adjust=False).mean()).iloc[-1], 0.0)
    ewma_vol = math.sqrt(max(0.0, ewma_var))

    baseline_window = max(10, min(len(returns), int(len(returns) * 0.75)))
    baseline_vol = _safe_float(returns.tail(baseline_window).std(ddof=1), realized_vol) or realized_vol
    vol_ratio = ewma_vol / max(baseline_vol, 1e-9)
    latest_z = abs(_safe_float(returns.iloc[-1], 0.0)) / max(ewma_vol, 1e-9)

    mean_ret = _safe_float(returns.mean(), 0.0)
    drift_t = mean_ret / max(realized_vol / math.sqrt(len(returns)), 1e-9)

    rolling = returns.rolling(max(10, min(24, len(returns) // 2))).std().dropna()
    vol_instability = 0.0
    if len(rolling) >= 2:
        vol_instability = _safe_float(rolling.std(ddof=1), 0.0) / max(_safe_float(rolling.mean(), realized_vol), 1e-9)

    drift_penalty = min(0.35, abs(drift_t) / 12.0)
    vol_penalty = min(0.45, max(0.0, vol_ratio - 1.0) / max(1.0, vol_ratio_threshold - 1.0) * 0.30)
    shock_penalty = min(0.35, max(0.0, latest_z - 1.0) / max(1.0, shock_z_threshold - 1.0) * 0.25)
    instability_penalty = min(0.30, vol_instability * 0.35)
    stability = max(0.0, min(1.0, 1.0 - drift_penalty - vol_penalty - shock_penalty - instability_penalty))

    if latest_z >= shock_z_threshold:
        state = "shock"
        reason = f"return_shock_z={latest_z:.2f}"
    elif vol_ratio >= vol_ratio_threshold:
        state = "volatile"
        reason = f"ewma_vol_ratio={vol_ratio:.2f}"
    elif stability < 0.45:
        state = "unstable"
        reason = f"unstable_distribution={stability:.2f}"
    else:
        state = "stable"
        reason = f"stable_returns={stability:.2f}"

    if state == "stable":
        score_mult = 1.0
        size_mult = 1.0
    elif state == "volatile":
        score_mult = 0.94
        size_mult = 0.82
    elif state == "shock":
        score_mult = 0.88
        size_mult = 0.68
    else:
        score_mult = 0.91
        size_mult = 0.76

    return TimeSeriesDiagnostics(
        observations=int(len(returns)),
        state=state,
        log_return_mean_pct=round(mean_ret, 5),
        realized_vol_pct=round(realized_vol, 5),
        ewma_vol_pct=round(ewma_vol, 5),
        volatility_ratio=round(vol_ratio, 4),
        latest_abs_return_z=round(latest_z, 4),
        drift_t_stat=round(drift_t, 4),
        stability_score=round(stability, 4),
        **_markov_features(returns, window=markov_window, threshold_pct=markov_threshold_pct),
        score_mult=round(score_mult, 4),
        size_mult=round(size_mult, 4),
        reason=reason,
    )


def _markov_features(returns: pd.Series, *, window: int = 20, threshold_pct: float = 2.0) -> dict:
    """Estimate a 3-state observable Markov transition from rolling returns.

    States are mutually exclusive and collectively exhaustive:
    bull, bear, sideways.  The transition matrix is estimated only from the
    historical window available now.  The one-step probability vector for the
    current state is returned as metadata.
    """
    if returns is None or len(returns) < max(8, window + 3):
        return {
            "markov_state": "unknown",
            "markov_bull_prob": 0.0,
            "markov_bear_prob": 0.0,
            "markov_sideways_prob": 0.0,
            "markov_edge": 0.0,
            "markov_transition_count": 0,
        }

    window = max(3, int(window or 20))
    threshold = abs(float(threshold_pct or 2.0))
    rolling_ret = returns.rolling(window).sum().dropna()
    if len(rolling_ret) < 3:
        return {
            "markov_state": "unknown",
            "markov_bull_prob": 0.0,
            "markov_bear_prob": 0.0,
            "markov_sideways_prob": 0.0,
            "markov_edge": 0.0,
            "markov_transition_count": 0,
        }

    def label(value: float) -> int:
        if value > threshold:
            return 0  # bull
        if value < -threshold:
            return 1  # bear
        return 2      # sideways

    labels = [label(_safe_float(v, 0.0)) for v in rolling_ret.tolist()]
    counts = [[0, 0, 0] for _ in range(3)]
    for cur, nxt in zip(labels[:-1], labels[1:]):
        counts[cur][nxt] += 1

    current = labels[-1]
    row = counts[current]
    row_total = sum(row)
    total_transitions = sum(sum(r) for r in counts)
    if row_total <= 0:
        probs = [1.0 / 3.0] * 3
    else:
        # Laplace smoothing avoids zero-probability overconfidence on sparse rows.
        probs = [(v + 1.0) / (row_total + 3.0) for v in row]

    names = ["bull", "bear", "sideways"]
    return {
        "markov_state": names[current],
        "markov_bull_prob": round(probs[0], 4),
        "markov_bear_prob": round(probs[1], 4),
        "markov_sideways_prob": round(probs[2], 4),
        "markov_edge": round(probs[0] - probs[1], 4),
        "markov_transition_count": int(total_transitions),
    }


def time_series_directional_mult(
    diagnostics: TimeSeriesDiagnostics | None,
    direction: str,
) -> tuple[float, float, str]:
    """Return (score_mult, size_mult, reason) for a directional thesis."""
    if diagnostics is None or diagnostics.observations <= 0:
        return 1.0, 1.0, "time_series_unavailable"

    score_mult = float(diagnostics.score_mult)
    size_mult = float(diagnostics.size_mult)
    reason = diagnostics.reason

    drift = float(diagnostics.drift_t_stat or 0.0)
    markov_edge = float(diagnostics.markov_edge or 0.0)
    enough_markov = int(diagnostics.markov_transition_count or 0) >= 20
    if direction == "long":
        if drift >= 1.25 and diagnostics.state == "stable":
            score_mult *= 1.025
            reason += "; drift_confirms_long"
        elif drift <= -1.25:
            score_mult *= 0.955
            size_mult *= 0.94
            reason += "; drift_conflicts_long"
        if enough_markov and markov_edge >= 0.20:
            score_mult *= 1.025
            reason += "; markov_confirms_long"
        elif enough_markov and markov_edge <= -0.20:
            score_mult *= 0.94
            size_mult *= 0.92
            reason += "; markov_conflicts_long"
    elif direction == "short":
        if drift <= -1.25 and diagnostics.state == "stable":
            score_mult *= 1.025
            reason += "; drift_confirms_short"
        elif drift >= 1.25:
            score_mult *= 0.955
            size_mult *= 0.94
            reason += "; drift_conflicts_short"
        if enough_markov and markov_edge <= -0.20:
            score_mult *= 1.025
            reason += "; markov_confirms_short"
        elif enough_markov and markov_edge >= 0.20:
            score_mult *= 0.94
            size_mult *= 0.92
            reason += "; markov_conflicts_short"

    return round(max(0.70, min(1.06, score_mult)), 4), round(max(0.55, min(1.05, size_mult)), 4), reason
