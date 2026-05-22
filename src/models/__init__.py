"""
Quant model layer.

Public modules:
    regime_classifier   four-state market regime classifier
    pwin_engine         dataclass + helpers for p(win) priors
    ev_model            expected-value gate (sample-based)
    edge_detector       observer-only edge tracker (per pair × edge_type)
    ml_engine           pass-through ML hooks (training disabled by default)
    cohort_policy       sleeve / regime / attribution allowlist
    strategy_lifecycle  cohort lifecycle (RESEARCH → ACTIVE)
    fund_manager        Jim Simons-style sizing controller (existing)
    strategy_router     research-backed sleeve assignment (existing)
"""
