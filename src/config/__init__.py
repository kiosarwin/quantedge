"""
Fail-fast config validation gate for `config/config.yaml`.

The bot's call sites use raw dict access (`cfg["risk"]["risk_per_trade_pct"]`),
which silently degrades when a key is mistyped or omitted: the offending stage
either trades with the wrong knob or freezes at runtime. For a bot whose risk
math, EV gate, and ML soft gate all read from the same dict, the canonical
"blew up the account" failure mode is `kelly.max_kelly_pct: 4.0` becoming
`40` (or `risk.max_drawdown_pct: 25.0` becoming `250`) without anything
noticing until the first oversized loss.

This module enforces a pydantic v2 schema on the loaded config:

* **Strict** on risk-critical sections — `exchange`, `trading`, `risk`,
  `exit`, `kelly`, `ev_model`, `safety`. Bounded ranges + cross-field
  consistency rules.
* **Lax** on additive sections (`indicators`, `strategy`, `lifecycle`,
  `edge_policy`, `paper_validation`, etc.) — the schema allows extra keys
  through verbatim. Promote individual sections to dedicated models
  incrementally as they stabilise.

Identity-preserving: ``validate_config`` returns the **same dict** that was
passed in (after pydantic validation). The pydantic model is a typed gate;
it does not flow through the rest of the codebase. Existing call sites such
as ``cfg["risk"]["risk_per_trade_pct"]`` keep working.

On failure, ``ConfigValidationError`` carries a structured human-readable
summary so operators can fix the YAML and re-run without grepping a Python
traceback.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .schema import NinjaConfig

__all__ = ["NinjaConfig", "ConfigValidationError", "validate_config"]


class ConfigValidationError(RuntimeError):
    """Raised when ``config.yaml`` fails the pydantic schema.

    The message is a structured multi-line summary intended to be printed
    directly to stderr by the entry point. Wraps the underlying
    ``pydantic.ValidationError`` as ``__cause__`` for debugging.
    """


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Validate *cfg* against ``NinjaConfig`` and return it unchanged.

    Identity-preserving so existing dict-access call sites keep working.

    Raises:
        ConfigValidationError: with a multi-issue human-readable summary.
    """
    try:
        NinjaConfig.model_validate(cfg)
    except ValidationError as exc:
        issues: list[str] = []
        for err in exc.errors():
            loc_parts = [str(p) for p in err.get("loc", ())]
            loc = ".".join(loc_parts) if loc_parts else "<root>"
            issues.append(f"    - {loc}: {err['msg']}")
        msg = (
            f"[CONFIG VALIDATION FAILED]\n"
            f"  {len(exc.errors())} issue(s):\n"
            + "\n".join(issues)
            + "\n\n  Fix the YAML and re-run. The bot deliberately refuses "
              "to start with an invalid config."
        )
        raise ConfigValidationError(msg) from exc
    return cfg
