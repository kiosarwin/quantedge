"""
Futures-specific sentiment signals: funding rate, open interest.
"""
from __future__ import annotations


def funding_sentiment_score(funding_rate: float, cfg: dict) -> float:
    """
    Score 0-100.  Contrarian: extreme positive funding = short bias.
    Neutral funding near zero → neutral score (50).
    """
    ind = cfg["indicators"]
    high_pos = ind["funding_high_threshold"]
    high_neg = ind["funding_low_threshold"]

    if funding_rate > high_pos:
        # Very bullish sentiment → crowded longs → short opportunity
        score = max(0.0, 100 - (funding_rate / high_pos) * 50)
    elif funding_rate < high_neg:
        # Very negative funding → crowded shorts → long opportunity
        score = max(0.0, 100 - (abs(funding_rate) / abs(high_neg)) * 50)
    else:
        # Neutral → moderate score
        score = 50.0 - (funding_rate / high_pos) * 25

    return float(max(0.0, min(100.0, score)))


def open_interest_score(oi_change_pct: float, cfg: dict) -> float:
    """
    Score 0-100.  Rising OI with price = trend confirmation.
    Falling OI = trend weakening.
    """
    ind = cfg["indicators"]
    threshold = ind["oi_change_threshold_pct"]

    if oi_change_pct >= threshold:
        score = min(100.0, 50 + (oi_change_pct / threshold) * 25)
    elif oi_change_pct <= -threshold:
        score = max(0.0, 50 + (oi_change_pct / threshold) * 25)
    else:
        score = 50.0

    return float(score)
