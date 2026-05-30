# Quant Operating Model

> Session handoff: See the codebase and config for current operating state.

## Default posture

- Default mode is `paper`.
- A cohort is not live-eligible until it reaches `SMALL_LIVE` or `ACTIVE`.
- Cohorts with negative expectancy or weak PF are downgraded to `DEGRADED` or `DISABLED`.

## Trade admission order

1. Market data and regime scoring
2. Smart-money and EV gate
3. Score threshold
4. Cohort policy
5. Strategy lifecycle gate
6. Adaptive brain / ML soft gate
7. Risk guard
8. Exposure guard
9. Execution

Signal validity and execution permission are intentionally separate.
`neutral` is not treated as alpha in the live admission path; it is a residual routing state and is excluded by the default strategy allowlists.

`AdaptiveBrain` runs after the main rule gates as a sizing / pair-health overlay. It can veto a pair that has decayed, but it does not replace the router or the hard risk manager.

## Lifecycle statuses

- `RESEARCH`: sample too small, paper only
- `PAPER_VALIDATION`: enough observations to monitor, still paper only
- `SMALL_LIVE`: modest validation, may be allowed in live mode
- `ACTIVE`: strongest cohort health
- `DEGRADED`: edge weakening, block new entries
- `DISABLED`: edge invalid or risk too poor, hard block

## Cohort key

Primary cohort health is tracked by:

- `strategy_sleeve`
- `market_regime`
- `sm_phase`
- `session`
- `direction`

## Metrics to monitor

- trades
- win rate
- profit factor
- expectancy per trade
- expectancy per sleeve
- average win / average loss
- payoff ratio
- max drawdown
- stop-hit rate
- TP1-hit rate
- average time in trade
- outlier contribution share
- average fee + slippage drag

## Kill-switch triggers

- repeated heartbeat/balance fetch failures
- repeated runtime exceptions
- equity mismatch beyond configured tolerance
- max drawdown / daily loss / weekly loss / cooldown lock
