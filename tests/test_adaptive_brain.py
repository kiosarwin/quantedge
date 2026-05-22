"""
Comprehensive tests for the adaptive AI brain components.

Each component is tested in isolation, then the AdaptiveBrain orchestrator
is tested for the integrated decision flow.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.thompson_bandit import ThompsonBandit
from src.models.online_logistic import OnlineLogistic, FeatureVector
from src.models.regime_transition import RegimeTransitionMatrix
from src.models.alpha_decay import AlphaDecayTracker
from src.risk.vol_targeting import VolTargeter
from src.models.adaptive_brain import AdaptiveBrain


# ─────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path, monkeypatch):
    """Each test uses fresh state directories — no test pollution."""
    state_dir = tmp_path / "models"
    state_dir.mkdir()

    monkeypatch.setattr(
        "src.models.thompson_bandit.ThompsonBandit.STATE_PATH",
        state_dir / "bandit.json",
    )
    monkeypatch.setattr(
        "src.models.online_logistic.OnlineLogistic.STATE_PATH",
        state_dir / "logistic.json",
    )
    monkeypatch.setattr(
        "src.models.regime_transition.RegimeTransitionMatrix.STATE_PATH",
        state_dir / "transitions.json",
    )
    monkeypatch.setattr(
        "src.models.alpha_decay.AlphaDecayTracker.STATE_PATH",
        state_dir / "alpha_decay.json",
    )
    monkeypatch.setattr(
        "src.risk.vol_targeting.VolTargeter.STATE_PATH",
        state_dir / "vol_targeter.json",
    )


# ─────────────────────────────────────────────────────────────────────────
#  Thompson Sampling Bandit
# ─────────────────────────────────────────────────────────────────────────

class TestThompsonBandit:

    def test_initial_priors_are_uniform(self):
        bandit = ThompsonBandit(arms=["a", "b", "c"])
        assert abs(bandit.expected_winrate("a") - 0.5) < 1e-6
        assert abs(bandit.expected_winrate("b") - 0.5) < 1e-6

    def test_wins_update_posterior_correctly(self):
        bandit = ThompsonBandit(arms=["a"], prior_alpha=2.0, prior_beta=2.0)
        # 5 wins, 1 loss → posterior alpha=7, beta=3 → wr=0.7
        for _ in range(5):
            bandit.update("a", won=True)
        bandit.update("a", won=False)
        wr = bandit.expected_winrate("a")
        assert abs(wr - (7 / 10)) < 1e-6

    def test_context_buckets_are_independent(self):
        bandit = ThompsonBandit(arms=["a", "b"])
        # In context "X", arm 'a' wins all
        for _ in range(10):
            bandit.update("a", won=True, context="X")
        # In context "Y", arm 'a' loses all
        for _ in range(10):
            bandit.update("a", won=False, context="Y")
        assert bandit.expected_winrate("a", "X") > 0.7
        assert bandit.expected_winrate("a", "Y") < 0.3

    def test_size_mult_scales_with_winrate_and_confidence(self):
        bandit = ThompsonBandit(arms=["a"])
        # Initial: should be ~1.0 (no data)
        assert abs(bandit.size_mult("a") - 1.0) < 0.05
        # After 20 wins: should be near 1.4x
        for _ in range(20):
            bandit.update("a", won=True)
        mult_after = bandit.size_mult("a")
        assert mult_after > 1.2, f"Expected >1.2x after winning streak, got {mult_after}"

    def test_size_mult_drops_for_losing_arm(self):
        bandit = ThompsonBandit(arms=["a"])
        for _ in range(20):
            bandit.update("a", won=False)
        mult = bandit.size_mult("a")
        assert mult < 0.8, f"Expected <0.8x for losing arm, got {mult}"

    def test_thompson_sampling_eventually_picks_best_arm(self):
        random.seed(42)
        bandit = ThompsonBandit(arms=["good", "bad"])
        # Simulate: "good" arm wins 80%, "bad" arm wins 20%
        for _ in range(100):
            bandit.update("good", won=(random.random() < 0.8))
            bandit.update("bad", won=(random.random() < 0.2))
        # Run 100 sampling rounds — "good" should be picked majority
        picks = [bandit.best_arm()[0] for _ in range(100)]
        good_picks = sum(1 for p in picks if p == "good")
        assert good_picks >= 80, f"Bandit failed to converge: {good_picks}/100 picks for good arm"

    def test_state_persists_across_instances(self, tmp_path):
        state = tmp_path / "models" / "bandit.json"
        b1 = ThompsonBandit(arms=["a"])
        for _ in range(5):
            b1.update("a", won=True)

        b2 = ThompsonBandit(arms=["a"])
        # b2 should have loaded the same posterior
        assert abs(b2.expected_winrate("a") - b1.expected_winrate("a")) < 1e-6


# ─────────────────────────────────────────────────────────────────────────
#  Online Logistic Regression
# ─────────────────────────────────────────────────────────────────────────

class TestOnlineLogistic:

    def test_initial_prediction_is_near_neutral(self):
        model = OnlineLogistic()
        fv = FeatureVector(score=50, structure_quality=50, trend_strength=50)
        p = model.predict(fv)
        # Bias initialized at 0.1 → sigmoid(0.1) ≈ 0.525
        assert 0.45 < p < 0.60

    def test_learns_simple_pattern(self):
        """High score + high structure → win. Low → loss. Should learn."""
        random.seed(0)
        model = OnlineLogistic(learning_rate=0.1)

        # Train on 200 synthetic samples
        for _ in range(200):
            high_quality = random.random() < 0.5
            if high_quality:
                fv = FeatureVector(score=85, structure_quality=80, trend_strength=75)
                won = random.random() < 0.75  # high quality wins 75%
            else:
                fv = FeatureVector(score=40, structure_quality=35, trend_strength=30)
                won = random.random() < 0.30  # low quality wins 30%
            model.update(fv, won)

        # After training, predictions should differentiate
        high_p = model.predict(FeatureVector(score=85, structure_quality=80, trend_strength=75))
        low_p = model.predict(FeatureVector(score=40, structure_quality=35, trend_strength=30))
        assert high_p > low_p + 0.15, \
            f"Model failed to learn: high={high_p:.3f}, low={low_p:.3f}"

    def test_confidence_increases_with_samples(self):
        model = OnlineLogistic()
        assert model.confidence() == 0.0
        for _ in range(25):
            model.update(FeatureVector(), won=True)
        assert model.confidence() == 0.5
        for _ in range(25):
            model.update(FeatureVector(), won=True)
        assert model.confidence() == 1.0

    def test_feature_importance_returns_all_features(self):
        model = OnlineLogistic()
        fi = model.feature_importance()
        assert "score" in fi
        assert "direction" in fi
        assert "bias" in fi
        assert len(fi) == OnlineLogistic.N_FEATURES

    def test_drift_detection(self):
        model = OnlineLogistic(learning_rate=0.05)
        # Train on contradictory data — model can't learn → high loss
        for _ in range(50):
            fv = FeatureVector(score=80)
            # Random outcomes — pure noise, model can't predict
            won = random.random() < 0.5
            model.update(fv, won)
        # Loss should be near 0.69 (random); not necessarily "drifting"
        recent = model.recent_loss(30)
        assert 0.5 < recent < 1.2  # in expected noise range


# ─────────────────────────────────────────────────────────────────────────
#  Regime Transition Matrix
# ─────────────────────────────────────────────────────────────────────────

class TestRegimeTransitionMatrix:

    def test_smoothed_priors_avoid_zero(self):
        rm = RegimeTransitionMatrix()
        probs = rm.transition_probs("trending_expansion", session="asia")
        # All regimes get non-zero probability via Laplace smoothing
        for r in rm.REGIMES:
            assert probs[r] > 0
        # Probabilities sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_transition_observation_updates_matrix(self):
        rm = RegimeTransitionMatrix()
        # Symbol BTC transitions: trending → distribution → chaos
        rm.observe("BTC", "trending_expansion", session="asia")
        rm.observe("BTC", "distribution", session="asia")
        probs = rm.transition_probs("trending_expansion", session="asia")
        # distribution should now have higher probability than chaos
        assert probs["distribution"] > probs["chaos"]

    def test_most_likely_next_excludes_self(self):
        rm = RegimeTransitionMatrix()
        session_state = rm._session_state("asia")
        session_state.transitions["trending_expansion"]["distribution"] = 10.0
        session_state.transitions["trending_expansion"]["chaos"] = 2.0
        next_regime, prob = rm.most_likely_next("trending_expansion", session="asia")
        assert next_regime == "distribution"

    def test_stability_score_for_persistent_regime(self):
        rm = RegimeTransitionMatrix()
        # If we've seen mostly self-loops, stability is high
        session_state = rm._session_state("asia")
        session_state.transitions["trending_expansion"]["trending_expansion"] = 10.0
        session_state.transitions["trending_expansion"]["distribution"] = 1.0
        score = rm.stability_score("trending_expansion", session="asia")
        assert score > 0.5

    def test_session_buckets_are_isolated(self):
        rm = RegimeTransitionMatrix()
        rm.observe("BTC", "trending_expansion", session="asia")
        rm.observe("BTC", "distribution", session="asia")

        probs_asia = rm.transition_probs("trending_expansion", session="asia")
        probs_london = rm.transition_probs("trending_expansion", session="london")

        assert probs_asia["distribution"] > probs_asia["chaos"]
        assert abs(sum(probs_london.values()) - 1.0) < 1e-6
        assert abs(probs_london["distribution"] - 0.25) < 1e-6

    def test_decay_downweights_old_transitions(self):
        rm = RegimeTransitionMatrix(decay_half_life_s=1.0)
        rm.observe("BTC", "trending_expansion", session="asia")
        rm._sessions["asia"].transitions["trending_expansion"]["distribution"] = 20.0
        rm._sessions["asia"].last_decay_ts = time.time() - 10.0

        probs = rm.transition_probs("trending_expansion", session="asia")

        assert probs["distribution"] < 0.4
        assert abs(sum(probs.values()) - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────
#  Alpha Decay Tracker
# ─────────────────────────────────────────────────────────────────────────

class TestAlphaDecayTracker:

    def test_no_kill_with_insufficient_observations(self):
        tracker = AlphaDecayTracker(min_obs_to_kill=8)
        for _ in range(5):
            tracker.record("BTCUSDT", -2.0)  # 5 losses
        alive, _ = tracker.is_alive("BTCUSDT")
        assert alive is True  # need 8+ obs to kill

    def test_dead_pair_blocked_after_enough_losses(self):
        tracker = AlphaDecayTracker(decay_threshold=-0.15, min_obs_to_kill=8)
        # Series of consistent losses
        for _ in range(10):
            tracker.record("BADCOIN", -1.5)
        alive, _ = tracker.is_alive("BADCOIN")
        assert alive is False
        assert tracker.get_size_mult("BADCOIN") == 0.0

    def test_winning_pair_gets_size_boost(self):
        tracker = AlphaDecayTracker()
        # Consistent winners
        for _ in range(10):
            tracker.record("HOTCOIN", 2.5)
        mult = tracker.get_size_mult("HOTCOIN")
        assert mult > 1.0, f"Expected boost for hot pair, got {mult}"

    def test_summary_reports_pairs(self):
        tracker = AlphaDecayTracker()
        tracker.record("BTC", 1.0)
        tracker.record("BTC", 2.0)
        tracker.record("BTC", 1.5)
        s = tracker.summary()
        assert s["n_pairs_tracked"] == 1
        assert s["n_dead_pairs"] == 0


# ─────────────────────────────────────────────────────────────────────────
#  Volatility Targeter
# ─────────────────────────────────────────────────────────────────────────

class TestVolTargeter:

    def test_no_data_returns_target_vol(self):
        vt = VolTargeter(target_annual_vol=0.40)
        assert abs(vt.realized_vol() - 0.40) < 1e-6
        assert abs(vt.size_multiplier() - 1.0) < 0.01

    def test_calm_market_sizes_up(self):
        vt = VolTargeter(target_annual_vol=0.40)
        # 30 days of small returns (calm market)
        for _ in range(30):
            vt.record_return(0.2, force=True)  # 0.2% daily = ~3.8% annual vol
        mult = vt.size_multiplier()
        assert mult > 1.2, f"Expected size up in calm market, got {mult}"

    def test_volatile_market_sizes_down(self):
        vt = VolTargeter(target_annual_vol=0.40)
        # Alternating ±5% (high vol)
        for i in range(30):
            vt.record_return(5.0 if i % 2 == 0 else -5.0, force=True)
        mult = vt.size_multiplier()
        assert mult < 0.7, f"Expected size down in volatile market, got {mult}"

    def test_multiplier_bounded(self):
        vt = VolTargeter(target_annual_vol=0.40, min_mult=0.5, max_mult=1.5)
        # Force extreme calm
        for _ in range(30):
            vt.record_return(0.001, force=True)
        assert vt.size_multiplier() <= 1.5
        # Force extreme vol
        vt2 = VolTargeter(target_annual_vol=0.40, min_mult=0.5, max_mult=1.5)
        for i in range(30):
            vt2.record_return(20.0 if i % 2 == 0 else -20.0, force=True)
        assert vt2.size_multiplier() >= 0.5


# ─────────────────────────────────────────────────────────────────────────
#  AdaptiveBrain Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def _mock_breakdown(
    symbol="BTCUSDT",
    direction="long",
    score=70.0,
    regime="trending_expansion",
    sleeve="trend_following",
):
    """Build a minimal SignalBreakdown-like object for brain tests."""
    return SimpleNamespace(
        symbol=symbol,
        direction=direction,
        total_score=score,
        structure_quality=65.0,
        trend_strength=72.0,
        volume_confirmation=60.0,
        funding_sentiment=50.0,
        open_interest=55.0,
        volatility=50.0,
        regime=SimpleNamespace(value=regime),
        smart_money=SimpleNamespace(
            phase=SimpleNamespace(value="trending"),
            score=70.0,
            direction_bias=direction,
        ),
        feature_vector=SimpleNamespace(momentum_strength=15.0),
        ev_result=SimpleNamespace(p_win=0.55, ev_net_pct=0.10, confidence=0.5),
        regime_ok=True,
        smart_money_ok=True,
        ev_ok=True,
        strategy_sleeve=sleeve,
        edge_result=None,
    )


class TestAdaptiveBrain:

    def test_brain_decides_with_neutral_state(self):
        brain = AdaptiveBrain({"adaptive_brain": {"enabled": True}})
        bd = _mock_breakdown()
        decision = brain.decide(bd)
        assert decision is not None
        # Initial state: no learning yet, multiplier should be near 1.0
        assert 0.7 < decision.overall_size_mult < 1.5

    def test_brain_learns_and_predictions_change(self):
        random.seed(42)
        brain = AdaptiveBrain({"adaptive_brain": {"enabled": True}})

        # Train: high-score trades win
        for _ in range(50):
            bd = _mock_breakdown(score=85, regime="trending_expansion")
            brain.learn(bd, won=True, pnl_pct=2.0)

        # Train: low-score trades lose
        for _ in range(50):
            bd = _mock_breakdown(score=45, regime="chaos")
            brain.learn(bd, won=False, pnl_pct=-1.0)

        # Now check predictions
        good_decision = brain.decide(_mock_breakdown(score=85, regime="trending_expansion"))
        bad_decision = brain.decide(_mock_breakdown(score=45, regime="chaos"))

        assert good_decision.p_win > bad_decision.p_win, \
            f"Brain failed to learn: good={good_decision.p_win}, bad={bad_decision.p_win}"

    def test_brain_blocks_dead_pair(self):
        brain = AdaptiveBrain({"adaptive_brain": {"enabled": True}})
        # Force "BADCOIN" into dead state
        for _ in range(15):
            bd = _mock_breakdown(symbol="BADCOIN")
            brain.learn(bd, won=False, pnl_pct=-2.0)

        decision = brain.decide(_mock_breakdown(symbol="BADCOIN"))
        assert decision.should_block is True
        assert decision.pair_alive is False

    def test_disabled_brain_returns_neutral(self):
        brain = AdaptiveBrain({"adaptive_brain": {"enabled": False}})
        decision = brain.decide(_mock_breakdown())
        assert decision.overall_size_mult == 1.0
        assert decision.notes == ["brain disabled"]

    def test_status_report_returns_structured_data(self):
        brain = AdaptiveBrain({"adaptive_brain": {"enabled": True}})
        bd = _mock_breakdown()
        brain.learn(bd, won=True, pnl_pct=1.5)

        report = brain.status_report()
        assert report["enabled"] is True
        assert "ml" in report
        assert "bandit" in report
        assert "regime_transitions" in report
        assert "alpha_decay" in report
        assert "vol_targeter" in report
        assert report["ml"]["n_updates"] >= 1
