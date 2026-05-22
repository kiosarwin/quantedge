"""
Edge detector — observer-only.

Tracks every signal the bot considers, builds an in-memory ledger of
(pair × edge_type) outcomes, and produces a per-signal advisory:

    EdgeResult(
        status:     OBSERVER | ACCEPT | DEGRADED | BLOCK,
        score:      0..100 — composite signal quality
        confidence: 0..1   — sample-driven trust
        action:     PASS | SCALE_DOWN | BLOCK
        size_mult:  scalar applied on top of FundManager + Kelly
        regime_coverage: regimes the (pair, edge) has been seen in
        reasons:    short bullets
    )

The detector is *observer-only* unless `edge_policy.enabled` is true in
config.  It still emits scores for telemetry/learning.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)

__all__ = ["EdgeContext", "EdgeResult", "EdgeDetector", "EdgeMemory"]


# ----------------------------------------------------------------- contracts

@dataclass(frozen=True)
class EdgeContext:
    pair: str
    direction: str
    regime: str
    smart_money_phase: str
    structure_quality: float
    trend_strength: float
    volume_zscore: float
    oi_change_pct: float
    funding_rate: float
    volatility: float
    p_win: float
    total_score: float
    score_threshold: float


@dataclass
class EdgeResult:
    status: str = "OBSERVER"          # OBSERVER | ACCEPT | DEGRADED | BLOCK
    score: float = 50.0
    confidence: float = 0.0
    action: str = "PASS"               # PASS | SCALE_DOWN | BLOCK
    size_mult: float = 1.0
    regime_coverage: str = "unknown"
    reasons: list[str] = field(default_factory=list)
    edge_type: str = "unknown"

    def summary(self) -> str:
        cov = self.regime_coverage or "unknown"
        why = "; ".join(self.reasons[:3]) if self.reasons else "no flags"
        return (
            f"{self.status} score={self.score:.0f} conf={self.confidence:.2f} "
            f"action={self.action} size={self.size_mult:.2f}x cov={cov} ({why})"
        )


# ----------------------------------------------------------------- memory

@dataclass
class _PairEdge:
    samples: int = 0
    wins: int = 0
    realised_pnl_pct: float = 0.0
    recent_outcomes: list[float] = field(default_factory=list)  # rolling pnl_pct
    regimes: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class EdgeMemory:
    """Per (pair × edge_type) sample ledger.  Pure-Python, JSON-friendly."""

    _RECENT_KEEP = 20

    def __init__(self):
        self._buckets: dict[tuple[str, str], _PairEdge] = {}

    # ------------------------------------------------------------ writes

    def record_signal(self, pair: str, edge_type: str, regime: str) -> None:
        bucket = self._buckets.setdefault((pair, edge_type), _PairEdge())
        bucket.regimes[regime] += 1

    def record_outcome(self, pair: str, edge_type: str, regime: str, pnl_pct: float) -> None:
        bucket = self._buckets.setdefault((pair, edge_type), _PairEdge())
        bucket.samples += 1
        bucket.realised_pnl_pct += pnl_pct
        bucket.recent_outcomes.append(pnl_pct)
        if len(bucket.recent_outcomes) > self._RECENT_KEEP:
            bucket.recent_outcomes = bucket.recent_outcomes[-self._RECENT_KEEP :]
        if pnl_pct > 0:
            bucket.wins += 1
        bucket.regimes[regime] += 1

    def hydrate_from_trade_log(self, trade_log: Iterable) -> int:
        loaded = 0
        for trade in trade_log or []:
            pair = getattr(trade, "symbol", "") or "?"
            regime = getattr(trade, "regime", "") or "unknown"
            edge_type = _edge_type_from_trade(trade)
            pnl_pct = float(getattr(trade, "pnl_pct", 0.0) or 0.0)
            self.record_outcome(pair, edge_type, regime, pnl_pct)
            loaded += 1
        return loaded

    # ------------------------------------------------------------- reads

    def total_samples(self) -> int:
        return sum(b.samples for b in self._buckets.values())

    def lookup(self, pair: str, edge_type: str) -> _PairEdge | None:
        return self._buckets.get((pair, edge_type))

    def regime_coverage(self, pair: str, edge_type: str) -> str:
        b = self._buckets.get((pair, edge_type))
        if not b or not b.regimes:
            return "unknown"
        return ",".join(sorted(b.regimes))

    def summary_rows(self, limit: int = 5) -> list[dict]:
        rows = []
        for (pair, edge_type), b in self._buckets.items():
            if b.samples == 0:
                continue
            wr = b.wins / b.samples
            recent = b.recent_outcomes[-10:]
            recent_wr = (sum(1 for r in recent if r > 0) / len(recent)) if recent else 0.0
            rows.append(
                {
                    "pair": pair,
                    "edge_type": edge_type,
                    "sample_size": b.samples,
                    "overall_wr": wr,
                    "recent_wr": recent_wr,
                    "regime_coverage": self.regime_coverage(pair, edge_type),
                }
            )
        rows.sort(key=lambda r: r["sample_size"], reverse=True)
        return rows[:limit]


# ----------------------------------------------------------------- detector

class EdgeDetector:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        edge_cfg = (cfg or {}).get("edge_policy", {}) or {}
        self.is_blocking = bool(edge_cfg.get("enabled", False))
        self._min_samples_to_block = int(edge_cfg.get("min_samples_to_block", 8) or 8)
        self._block_recent_wr = float(edge_cfg.get("block_recent_wr", 0.30) or 0.30)
        self._degraded_recent_wr = float(edge_cfg.get("degraded_recent_wr", 0.42) or 0.42)
        self._size_floor = float(edge_cfg.get("min_size_mult", 0.55) or 0.55)
        self.memory = EdgeMemory()

    # ------------------------------------------------------------- API

    def evaluate(self, ctx: EdgeContext) -> EdgeResult:
        edge_type = _edge_type_from_ctx(ctx)
        # Always log the signal so memory grows even when the trade is rejected.
        self.memory.record_signal(ctx.pair, edge_type, ctx.regime)

        score = self._score(ctx)
        bucket = self.memory.lookup(ctx.pair, edge_type)
        sample_size = bucket.samples if bucket else 0
        confidence = min(1.0, sample_size / 12.0)

        recent_outcomes = bucket.recent_outcomes[-10:] if bucket else []
        recent_wr = (
            (sum(1 for r in recent_outcomes if r > 0) / len(recent_outcomes))
            if recent_outcomes
            else 0.0
        )

        reasons: list[str] = []
        action = "PASS"
        size_mult = 1.0
        status = "OBSERVER" if not self.is_blocking else "ACCEPT"

        if sample_size >= self._min_samples_to_block:
            if recent_wr < self._block_recent_wr:
                status = "BLOCK" if self.is_blocking else "DEGRADED"
                action = "BLOCK" if self.is_blocking else "SCALE_DOWN"
                size_mult = self._size_floor
                reasons.append(
                    f"recent WR {recent_wr:.0%} < {self._block_recent_wr:.0%}"
                    f" over last {len(recent_outcomes)} trades"
                )
            elif recent_wr < self._degraded_recent_wr:
                status = "DEGRADED"
                action = "SCALE_DOWN"
                size_mult = max(self._size_floor, 0.78)
                reasons.append(
                    f"recent WR {recent_wr:.0%} below comfort band"
                )

        if ctx.volatility >= 88:
            size_mult *= 0.92
            reasons.append("high volatility — slight size discount")

        if ctx.p_win >= 0.62 and ctx.total_score >= ctx.score_threshold + 8:
            score = min(100.0, score + 4.0)
            reasons.append("high p_win × score above threshold band")

        regime_coverage = self.memory.regime_coverage(ctx.pair, edge_type)

        return EdgeResult(
            status=status,
            score=round(score, 1),
            confidence=round(confidence, 3),
            action=action,
            size_mult=round(size_mult, 3),
            regime_coverage=regime_coverage,
            reasons=reasons,
            edge_type=edge_type,
        )

    # ------------------------------------------------------------- score

    @staticmethod
    def _score(ctx: EdgeContext) -> float:
        # Composite quality signal (rough, but useful for telemetry).
        score = 50.0
        score += (ctx.structure_quality - 50.0) * 0.30
        score += (ctx.trend_strength - 50.0) * 0.20
        score += min(20.0, max(-20.0, ctx.volume_zscore * 6.0))
        score += (ctx.p_win - 0.5) * 40.0
        score += min(10.0, max(-10.0, (ctx.total_score - ctx.score_threshold) * 0.6))
        if abs(ctx.funding_rate) > 0.0008:        # > 0.08% / 8h is stretched
            score -= 4.0
        if ctx.volatility >= 90:
            score -= 6.0
        return max(0.0, min(100.0, score))


# ------------------------------------------------------------ helpers

def _edge_type_from_ctx(ctx: EdgeContext) -> str:
    sm = ctx.smart_money_phase or "neutral"
    return f"{ctx.regime}|{sm}|{ctx.direction}"


def _edge_type_from_trade(trade) -> str:
    regime = getattr(trade, "regime", "") or "unknown"
    scores = getattr(trade, "scores", {}) or {}
    sm = scores.get("sm_phase", "neutral") or "neutral"
    direction = getattr(trade, "direction", "") or "long"
    return f"{regime}|{sm}|{direction}"
