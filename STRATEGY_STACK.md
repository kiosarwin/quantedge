# Ninja Trader Strategy Stack

> Session handoff: `session.md` contains the current runtime state; use it instead of reconstructing old context.

## Core Thesis

This bot now treats directional crypto-perp trading as a regime-aware stack:

1. `trend_following`
2. `compression_breakout`
3. `reversal`
4. `neutral` fallback

The research basis is:

- Time-series momentum / trend following is the strongest directional edge in liquid futures-like instruments.
- Crypto momentum weakens when cross-sectional dispersion becomes extreme.
- Reversal becomes more attractive around crowding, liquidation sweeps, and distribution states.
- Funding, basis, order flow, and OI are best used as filters and sizing overlays, not naive standalone alpha.

## Sleeves

### `trend_following`

Use when:

- regime = `trending_expansion`
- trend strength is strong
- structure quality is strong
- smart money is `trending`, `accumulation`, or `neutral`

Behavior:

- boosts score
- increases size
- lowers threshold slightly
- scales down when dispersion rises

### `compression_breakout`

Use when:

- regime = `accumulation_compression`
- structure is strong enough to justify breakout continuation
- smart money is `accumulation`, `liquidity_sweep`, or `neutral`

Behavior:

- small score boost
- slightly smaller size than trend sleeve

### `reversal`

Use when:

- smart money detects `liquidity_sweep`
- or regime = `distribution` with reversal-style smart money context

Behavior:

- moderate score boost
- smaller default size
- can gain relative priority when dispersion is extreme

## Dispersion Logic

Cross-sectional dispersion is computed from the standard deviation of current symbol momentum strengths.

- `normal`: trend sleeve allowed normally
- `warn`: trend sleeve partially de-risked
- `high`: trend sleeve materially de-risked, reversal sleeve relatively favored

This follows the academic result that momentum signal reliability degrades in dispersion tails.

## What To Measure

Forward evaluation should report:

- expectancy by `strategy_sleeve`
- expectancy by `direction`
- expectancy by `regime`
- expectancy by `strategy_sleeve x regime`
- turnover and fee drag by sleeve
- hit rate, profit factor, Sharpe, max drawdown, skew

## Current Implementation Scope

Implemented now:

- strategy router
- dispersion-aware score and sizing overlays
- state/reporting metadata for sleeve attribution

Not implemented yet:

- dedicated sleeve-specific exit logic
- market-neutral funding carry sleeve
- explicit cross-sectional dispersion factor in backtester reports
