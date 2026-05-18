"""
Order book analysis — bid/ask imbalance, liquidity walls, absorption detection.
"""
from __future__ import annotations

import numpy as np


def bid_ask_imbalance(order_book: dict) -> float:
    """
    Returns bid/ask volume ratio.
    > 1.0 means more bid pressure, < 1.0 means more ask pressure.
    """
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    bid_vol = sum(float(b[1]) for b in bids)
    ask_vol = sum(float(a[1]) for a in asks)
    if ask_vol == 0:
        return 1.0
    return bid_vol / ask_vol


def find_walls(order_book: dict, multiplier: float = 5.0) -> dict:
    """
    Identifies large bid/ask walls (levels > multiplier × average level size).
    Returns {'bid_wall': price|None, 'ask_wall': price|None}.
    """
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    bid_wall = _find_wall(bids, multiplier)
    ask_wall = _find_wall(asks, multiplier)
    return {"bid_wall": bid_wall, "ask_wall": ask_wall}


def _find_wall(levels: list, multiplier: float) -> float | None:
    if not levels:
        return None
    sizes = [float(lvl[1]) for lvl in levels]
    avg = np.mean(sizes)
    for price, size in levels:
        if float(size) > avg * multiplier:
            return float(price)
    return None


def order_book_score(order_book: dict, cfg: dict, direction: str = "long") -> float:
    """
    Score 0-100 based on order book in favour of the proposed trade direction.
    """
    ob_cfg = cfg["order_book"]
    imbalance = bid_ask_imbalance(order_book)
    walls = find_walls(order_book, ob_cfg["wall_size_multiplier"])
    threshold = ob_cfg["imbalance_threshold"]

    score = 50.0   # neutral baseline

    if direction == "long":
        if imbalance > threshold:
            score += 30.0
        elif imbalance < 1 / threshold:
            score -= 30.0
        # Bid wall nearby = support
        if walls["bid_wall"] is not None:
            score += 20.0
    else:  # short
        if imbalance < 1 / threshold:
            score += 30.0
        elif imbalance > threshold:
            score -= 30.0
        if walls["ask_wall"] is not None:
            score += 20.0

    return float(max(0.0, min(100.0, score)))
