# Session Notes

Last updated: 2026-05-19

## Current State

- Repo `/home/arwin/ninja_trader` is a git repo on branch `work`.
- Strategy-stack alignment changes are in progress: `neutral` is residual routing only, live/paper allowlists center on `trend_following` and `reversal`, and `compression_breakout` stays experimental.
- Commit `3fdaa16` is pushed to `origin/master` and the VM is synced/restarted from that commit.
- Fresh paper reset remains in effect: persisted trade history and learned state were cleared, and the bot seeds from `$75`.
- Telegram heartbeat is off; `JIM SIMONS — FUND MANAGER REPORT` remains the primary Telegram cadence every 5 minutes.
- Live secrets were sanitized out of tracked markdown/config files; runtime `.env` stays on the VM.

## What Is Already Done

- Initialized git in the correct repo and linked it to GitHub.
- Removed live credential literals from markdown/config files before commit.
- Kept runtime artifacts out of the repository via `.gitignore`.
- Verified repo tests: `34 passed`.
- Deployed the current paper-reset baseline to production VM and restarted the bot successfully.

## Current Runtime Behavior

- Bot is healthy and scanning normally on the VM.
- Telegram heartbeat is disabled; fund manager report remains active every 5 minutes.
- Scanner, risk guard, and trade loop are running from the deployed commit.
- No local commit changes remain after pushing `work -> master` after the previous baseline.
- `neutral` should not be treated as alpha in admission logic; it is blocked by the default allowlists.

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
- [`src/models/cohort_policy.py`](/home/arwin/ninja_trader/src/models/cohort_policy.py)
- [`src/scoring/scorer.py`](/home/arwin/ninja_trader/src/scoring/scorer.py)
- [`config/config.yaml`](/home/arwin/ninja_trader/config/config.yaml)
- [`tests/test_risk_manager.py`](/home/arwin/ninja_trader/tests/test_risk_manager.py)
- [`tests/test_paper_validation.py`](/home/arwin/ninja_trader/tests/test_paper_validation.py)
- [`tests/test_cohort_policy.py`](/home/arwin/ninja_trader/tests/test_cohort_policy.py)

## Verified Tests

- `./venv/bin/python -m pytest tests/test_risk_manager.py tests/test_paper_validation.py tests/test_cohort_policy.py -q`
- Result: `26 passed`

## Safe Next Step

If resuming in a new session, do this first:

1. `tmux capture-pane -pt ninja_trader -S -80`
2. `./venv/bin/python -c "from src.main import load_config, normalize_config; cfg=normalize_config(load_config('config/config.yaml')); print(cfg['trading']['paper_starting_equity'], cfg['risk']['min_risk_usd'], cfg['risk']['max_direction_risk_pct'], cfg['trading']['max_open_trades'])"`
3. `pgrep -af "python -m src.main|python -m src|venv/bin/python -m src"`

If the live process drifts, restart from the `tmux` session after confirming config.

## Resume Prompt

Use this prompt in a new session:

> Continue in `/home/arwin/ninja_trader`. Read `session.md` first. Current paper run is fresh from `$75`, config is already tuned for small-account paper sampling, and the bot is running in `tmux` session `ninja-vm`. Do not re-litigate GitHub or repo issues; there is no git repo here. Verify the live config and runtime state first, then only make surgical changes if a concrete bug or bottleneck is still present. Current key settings to preserve unless explicitly changed: `paper_starting_equity=75`, `risk_per_trade_pct=1.0`, `min_risk_usd=0.75`, `max_open_trades=4`, `max_direction_risk_pct=2.50`, isolated margin, and `edge_policy=false`. Prioritize checking `tmux capture-pane -pt ninja-vm`, `logs/futures_trader.log`, and whether the bot is healthy/opening trades as expected. Keep changes minimal and verify with the relevant tests before handoff.
