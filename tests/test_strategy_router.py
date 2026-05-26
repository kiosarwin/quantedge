from types import SimpleNamespace

from src.models.strategy_router import DispersionState, StrategyRouter


def _cfg():
    return {
        "strategy": {
            "trend_min_score": 65,
            "trend_min_structure": 58,
            "trend_long_only": True,
            "trend_require_participation": True,
            "trend_min_alignment": 0.54,
            "breakout_min_alignment": 0.57,
            "reversal_min_alignment": 0.45,
            "enable_compression_breakout": False,
            "reversal_max_volatility": 75,
            "allow_long_reversal": True,
            "allow_short_reversal": True,
            "short_reversal_require_distribution": True,
            "short_reversal_min_sm_score": 85,
            "short_reversal_min_structure": 62,
            "short_reversal_min_volume": 45,
            "enable_short_setups": True,
            "short_setup_min_confidence": 0.65,
            "short_setup_min_structure": 50,
            "short_setup_max_volatility": 85,
            "dispersion_warn": 28,
            "dispersion_high": 38,
        }
    }


def _short_setup(label: str = "phase_d", confidence: float = 0.80):
    return SimpleNamespace(
        strategy=SimpleNamespace(value=label),
        label=label,
        confidence=confidence,
        is_valid=True,
        entry_price=100.0,
        stop_loss=102.0,
        notes="test",
    )


def _feature_vector(
    alignment=0.62,
    market_structure="none",
    order_flow_imbalance=0.0,
    momentum_strength=0.0,
    vwap_distance=0.0,
    liquidation_pressure=0.0,
):
    return SimpleNamespace(
        market_structure=market_structure,
        order_flow_imbalance=order_flow_imbalance,
        momentum_strength=momentum_strength,
        vwap_distance=vwap_distance,
        liquidation_pressure=liquidation_pressure,
        directional_alignment=lambda _direction: alignment,
    )


def _breakdown(
    direction="long",
    regime="trending_expansion",
    sm_phase="liquidity_sweep",
    sm_bias="long",
    sm_score=90.0,
    structure_quality=70.0,
    volume_confirmation=60.0,
    open_interest=60.0,
    funding_sentiment=50.0,
    volatility=25.0,
    trend_strength=72.0,
    short_setup=None,
    feature_vector=None,
):
    return SimpleNamespace(
        direction=direction,
        regime=SimpleNamespace(value=regime),
        smart_money=SimpleNamespace(
            phase=SimpleNamespace(value=sm_phase),
            direction_bias=sm_bias,
            score=sm_score,
        ),
        structure_quality=structure_quality,
        volume_confirmation=volume_confirmation,
        open_interest=open_interest,
        funding_sentiment=funding_sentiment,
        volatility=volatility,
        trend_strength=trend_strength,
        short_setup=short_setup,
        feature_vector=feature_vector,
    )


def test_strategy_router_allows_short_reversal_only_when_conditions_match():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(direction="short", regime="distribution", sm_phase="distribution", sm_bias="short"),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "reversal"
    assert decision.reason.startswith("short reversal sleeve")


def test_strategy_router_allows_short_trend_following():
    """UPGRADED: Short trend-following is now valid per Jegadeesh & Titman momentum research."""
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(direction="short", regime="trending_expansion", sm_phase="neutral", sm_bias="short"),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "trend_following"
    assert "trend sleeve" in decision.reason


def test_strategy_router_blocks_trend_when_funding_is_extreme_against_direction():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(
            direction="long",
            regime="trending_expansion",
            sm_phase="trending",
            sm_bias="long",
            funding_sentiment=90.0,
        ),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "neutral"
    assert "funding/OI not aligned" in decision.reason


def test_strategy_router_allows_compression_breakout_only_with_participation():
    cfg = _cfg()
    cfg["strategy"]["enable_compression_breakout"] = True
    router = StrategyRouter(cfg)
    decision = router.evaluate(
        _breakdown(
            direction="long",
            regime="accumulation_compression",
            sm_phase="accumulation",
            sm_bias="neutral",
            structure_quality=68.0,
            volume_confirmation=70.0,
            open_interest=66.0,
        ),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "compression_breakout"
    assert "compression + participation" in decision.reason


def test_strategy_router_blocks_thin_trend_when_participation_required():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(
            direction="long",
            regime="trending_expansion",
            sm_phase="trending",
            sm_bias="long",
            volume_confirmation=44.0,
            open_interest=43.0,
            funding_sentiment=78.0,
        ),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "neutral"
    assert "not aligned" in decision.reason or "weak directional alignment" in decision.reason


def test_strategy_router_blocks_long_when_microstructure_contradicts_direction():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(
            direction="long",
            regime="trending_expansion",
            sm_phase="trending",
            sm_bias="long",
            feature_vector=_feature_vector(
                alignment=0.70,
                market_structure="bearish_bos",
                order_flow_imbalance=-35.0,
                momentum_strength=-20.0,
            ),
        ),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "neutral"
    assert "microstructure contradicts" in decision.reason


def test_strategy_router_allows_short_trend_with_bearish_microstructure():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(
            direction="short",
            regime="trending_expansion",
            sm_phase="neutral",
            sm_bias="short",
            feature_vector=_feature_vector(
                alignment=0.68,
                market_structure="bearish_bos",
                order_flow_imbalance=-30.0,
                momentum_strength=-25.0,
                vwap_distance=-1.0,
            ),
        ),
        DispersionState(value=0.0, state="normal"),
    )
    assert decision.sleeve == "trend_following"
    assert decision.setup_type == "trend_continuation"
    assert decision.quality_score > 0


def test_strategy_router_identifies_short_reversal_candidate():
    router = StrategyRouter(_cfg())
    breakdown = _breakdown(
        direction="short",
        regime="distribution",
        sm_phase="distribution",
        sm_bias="short",
    )
    assert router.is_short_reversal_candidate(breakdown) is True


# ──────────────────────────────────────────────────────────────────────────────
#  Phase D / Liq Sweep dedicated short-setup admission path
# ──────────────────────────────────────────────────────────────────────────────

def test_short_setup_routes_to_reversal_outside_distribution_regime():
    """A high-confidence Phase D setup should route to reversal even when
    the regime is not yet `distribution` — this is precisely the post-
    distribution breakdown case the dedicated detector targets."""
    router = StrategyRouter(_cfg())
    breakdown = _breakdown(
        direction="short",
        regime="trending_expansion",   # NOT distribution
        sm_phase="neutral",
        sm_bias="neutral",
        sm_score=40.0,
        structure_quality=60.0,
        volume_confirmation=50.0,
        volatility=55.0,
        short_setup=_short_setup("phase_d", confidence=0.80),
    )
    assert router.is_short_setup_candidate(breakdown) is True
    decision = router.evaluate(breakdown, DispersionState(value=0.0, state="normal"))
    assert decision.sleeve == "reversal"
    assert "phase_d" in decision.reason


def test_short_setup_below_confidence_threshold_is_rejected():
    router = StrategyRouter(_cfg())
    breakdown = _breakdown(
        direction="short",
        regime="trending_expansion",
        sm_phase="neutral",
        sm_bias="neutral",
        structure_quality=60.0,
        short_setup=_short_setup("phase_d", confidence=0.50),
    )
    assert router.is_short_setup_candidate(breakdown) is False


def test_short_setup_disabled_via_config():
    cfg = _cfg()
    cfg["strategy"]["enable_short_setups"] = False
    router = StrategyRouter(cfg)
    breakdown = _breakdown(
        direction="short",
        regime="trending_expansion",
        short_setup=_short_setup("liq_sweep", confidence=0.85),
    )
    assert router.is_short_setup_candidate(breakdown) is False


def test_short_setup_label_returns_strategy_label():
    router = StrategyRouter(_cfg())
    breakdown = _breakdown(
        direction="short",
        short_setup=_short_setup("liq_sweep", confidence=0.80),
    )
    assert router.short_setup_label(breakdown) == "liq_sweep"


def test_short_setup_blocked_by_low_structure_quality():
    router = StrategyRouter(_cfg())
    breakdown = _breakdown(
        direction="short",
        regime="trending_expansion",
        structure_quality=40.0,        # below the 50 floor
        short_setup=_short_setup("phase_d", confidence=0.80),
    )
    assert router.is_short_setup_candidate(breakdown) is False


def test_strategy_router_builds_explicit_setup_passport_with_rotation_context():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(
            direction="short",
            regime="trending_expansion",
            sm_phase="neutral",
            sm_bias="short",
            feature_vector=_feature_vector(
                alignment=0.68,
                market_structure="bearish_bos",
                order_flow_imbalance=-30.0,
                momentum_strength=-25.0,
                vwap_distance=-1.0,
                liquidation_pressure=55.0,
            ),
        ),
        DispersionState(value=0.0, state="normal"),
    )

    assert decision.passport is not None
    assert decision.passport.setup_type == "trend_continuation"
    assert decision.passport.sleeve == "trend_following"
    assert decision.passport.microstructure_state == "aligned"
    assert decision.passport.expected_path == "impulse_continuation"
    assert decision.passport.decision == "eligible"


def test_strategy_router_passport_marks_no_trade_contradiction():
    router = StrategyRouter(_cfg())
    decision = router.evaluate(
        _breakdown(
            direction="long",
            regime="trending_expansion",
            sm_phase="trending",
            sm_bias="long",
            feature_vector=_feature_vector(
                alignment=0.70,
                market_structure="bearish_bos",
                order_flow_imbalance=-35.0,
                momentum_strength=-20.0,
            ),
        ),
        DispersionState(value=0.0, state="normal"),
    )

    assert decision.passport is not None
    assert decision.passport.setup_type == "no_trade"
    assert decision.passport.microstructure_state == "contradicts_direction"
    assert decision.passport.decision == "no_trade"


def test_sector_for_asset_classifies_esports_from_binance_alpha_config():
    from src.models.strategy_passport import binance_alpha_assets, sector_for_asset

    for asset in ("B2", "ESPORTS", "HANA", "Q", "USELESS"):
        assert asset in binance_alpha_assets()
        assert sector_for_asset(asset) == "binance_alpha"


def test_sector_for_asset_classifies_current_non_alpha_universe():
    from src.models.strategy_passport import sector_for_asset

    expected = {
        "ADA": "l1",
        "ASTER": "perp_dex",
        "BZ": "new_listing",
        "CL": "commodity",
        "DEXE": "defi",
        "EDEN": "rwa",
        "EIGEN": "restaking",
        "ENA": "defi",
        "ERA": "l2",
        "FIDA": "solana_ecosystem",
        "GENIUS": "new_listing",
        "HYPE": "perp_dex",
        "LINK": "oracle",
        "LIT": "privacy_infra",
        "MU": "tradfi_equity",
        "NIL": "privacy_ai",
        "ONDO": "rwa",
        "PLUME": "rwa",
        "PHA": "depin",
        "SAGA": "gaming",
        "TON": "l1",
        "TRX": "l1",
        "UNI": "defi",
        "XAG": "commodity",
        "XAU": "commodity",
        "XRP": "payments",
        "ZEC": "privacy",
    }
    for asset, sector in expected.items():
        assert sector_for_asset(asset) == sector


def test_register_market_sectors_classifies_unknown_binance_metadata():
    from src.models import strategy_passport as passport

    passport._DYNAMIC_SECTOR_MAP.pop("ZZZ", None)
    passport._DYNAMIC_SECTOR_MAP.pop("NEWL1", None)
    registered = passport.register_market_sectors(
        {
            "ZZZ/USDT:USDT": {
                "base": "ZZZ",
                "info": {"underlyingType": "EQUITY", "underlyingSubType": ["TradFi"]},
            },
            "NEWL1/USDT:USDT": {
                "base": "NEWL1",
                "info": {"underlyingType": "CRYPTO", "underlyingSubType": ["Layer-1"]},
            },
        }
    )

    assert registered["ZZZ"] == "tradfi_equity"
    assert registered["NEWL1"] == "l1"
    assert passport.sector_for_asset("ZZZ") == "tradfi_equity"
    assert passport.sector_for_asset("NEWL1") == "l1"
