# Session Notes

Last updated: 2026-05-22

## Current State

- Repo `/home/arwin/ninja_trader` is a git repo on branch `work`. The worktree is currently dirty because code updates are still in flight; do not assume local/VM sync without checking `git status`.
- Current stack is futures-only and two-way: `trend_following` is allowed long and short, `reversal` remains the main short sleeve, `compression_breakout` stays experimental, and `neutral` is residual routing only.
- `AdaptiveBrain` is active in runtime as a sizing / pair-health overlay. `MLPredictor` remains a passive interface, not the main decision brain.
- Current paper baseline seeds from `$70`; paper-validation wideners are active, with `min_score_threshold` at `42`, `max_open_trades` base at `2`, and paper override widening to `5`.
- Telegram heartbeat is active every minute in the current config; fund-manager reports remain the higher-level performance summary cadence every 10 closed trades.
- `CAGR` / `MAR` in the Jim report are now guarded for short histories: 1-decimal display stays, but annualization is suppressed until the sample is deep enough.
- Live secrets were sanitized out of tracked markdown/config files; runtime `.env` stays on the VM.

## What Is Already Done

- Initialized git in the correct repo and linked it to GitHub.
- Removed live credential literals from markdown/config files before commit.
- Kept runtime artifacts out of the repository via `.gitignore`.
- Verified the relevant focused tests during prior PR work; rerun the smallest relevant slice before any new logic change.
- Production VM syncs should be treated as current only after an explicit deploy and verification.

## Current Runtime Behavior

- Bot is healthy when the VM is aligned to the current config snapshot.
- Scanner, risk guard, trade loop, and adaptive overlays are running from the current paper configuration.
- `neutral` should not be treated as alpha in admission logic; it remains a fallback / residual bucket.
- Dual-direction trend flow is enabled in config, so short trend ideas are no longer parked by default.
- `AdaptiveBrain` can block dead pairs and scale positions, but it does not replace the router or the risk manager.

## Connection Metadata

- GCP VM: `35.231.37.107`
- GCP user: `kiosarwin`
- GCP SSH key: `/root/.ssh/gcp_35_231_37_107`
- GitHub repo: `git@github.com:kiosarwin/ninja-trader.git`
- GitHub deploy key: `/root/.ssh/github_ninja_trader_deploy`
- Local branch: `work`
- Remote branch target: `master`
- Do not ask again for these stable connection details unless they change.

## Important Files

- [`src/main.py`](/home/arwin/ninja_trader/src/main.py)
- [`src/risk/risk_manager.py`](/home/arwin/ninja_trader/src/risk/risk_manager.py)
- [`src/risk/kelly_sizer.py`](/home/arwin/ninja_trader/src/risk/kelly_sizer.py)
- [`src/execution/trade_manager.py`](/home/arwin/ninja_trader/src/execution/trade_manager.py)
- [`src/models/adaptive_brain.py`](/home/arwin/ninja_trader/src/models/adaptive_brain.py)
- [`src/models/cohort_policy.py`](/home/arwin/ninja_trader/src/models/cohort_policy.py)
- [`src/scoring/scorer.py`](/home/arwin/ninja_trader/src/scoring/scorer.py)
- [`config/config.yaml`](/home/arwin/ninja_trader/config/config.yaml)
- [`tests/test_risk_manager.py`](/home/arwin/ninja_trader/tests/test_risk_manager.py)
- [`tests/test_paper_validation.py`](/home/arwin/ninja_trader/tests/test_paper_validation.py)
- [`tests/test_cohort_policy.py`](/home/arwin/ninja_trader/tests/test_cohort_policy.py)

## Verified Tests

- Re-run the focused tests that touch the file you changed before handoff.
- For runtime-sensitive changes, prefer the smallest targeted pytest slice plus a config compile check.

## Safe Next Step

If resuming in a new session, do this first:

1. `tmux capture-pane -pt ninja_trader -S -80`
2. `./venv/bin/python -c "from src.main import load_config, normalize_config; cfg=normalize_config(load_config('config/config.yaml')); print(cfg['trading']['paper_starting_equity'], cfg['risk']['min_risk_usd'], cfg['risk']['max_direction_risk_pct'], cfg['trading']['max_open_trades'], cfg['trading']['min_score_threshold'])"`
3. `pgrep -af "python -m src.main|python -m src|venv/bin/python -m src"`

If the live process drifts, restart from the `tmux` session after confirming config.

## Resume Prompt

Use this prompt in a new session:

> Continue in `/home/arwin/ninja_trader`. Read `session.md` first. The repo is active and the worktree may be dirty, so inspect `git status` before assuming anything is synced. Current paper run is seeded from `$70`, dual-direction trend flow is enabled, `AdaptiveBrain` is active, and `MLPredictor` is still a passive helper. Verify the live config and runtime state first, then only make surgical changes if a concrete bug or bottleneck is still present. Current key settings to preserve unless explicitly changed: `paper_starting_equity=70`, `risk_per_trade_pct=1.5`, `min_risk_usd=0.75`, `max_open_trades=2` base with paper override to `5`, `max_direction_risk_pct=5.50`, isolated margin, and `trend_long_only=false`. Prioritize checking `tmux capture-pane -pt ninja-vm`, `logs/futures_trader.log`, and whether the bot is healthy/opening trades as expected. Keep changes minimal and verify with the relevant tests before handoff.
