"""
Three-Signal Crisis Detection System
=====================================
Each signal is a standalone module, but they are designed to be combined.
Together they form a multi-modal early-warning system.

Signal architecture
-------------------

Signal 1 — Rough Volatility (Hurst DFA)
    What it measures : The *texture* of volatility — is it rough (anti-persistent)
                       or smooth (persistent)?
    Crisis signature : H drops below 0.25 AND vol_surprise spikes above 1.0.
                       Interpretation: the market is in hyper-mean-reverting mode,
                       typical of panic-driven de-risking.
    Lead time        : 3–10 trading days before acute phase.

Signal 2 — HMM Regime
    What it measures : The latent *state* of the market inferred from returns,
                       volatility, and rough-vol metrics.
    Crisis signature : Regime transitions to "Bear / High Vol" or "Pre-Crisis"
                       with confidence > 0.60.
    Lead time        : 5–15 trading days before acute phase.

Signal 3 — Topological Stress (TDA)
    What it measures : The *geometry* of inter-asset correlations.
    Crisis signature : Topological stress S(t) > 2.5σ — the correlation network
                       is fragmenting (β₀ spike) or herding into loops (β₁ spike).
    Lead time        : 1–4 weeks before acute phase.

Combined signal
---------------
When all three agree, confidence is highest.
A weighted vote produces a scalar Crisis Probability in [0, 1]:

    P_crisis = w1 × p_rough + w2 × p_regime + w3 × p_topo

Default weights: w1=0.30, w2=0.40, w3=0.30
(Regime signal weighted highest — it integrates the other two as features.)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RoughVolSignal:
    """Output from the Rough Volatility Engine."""
    hurst_index:         float = 0.5
    sigma_rough:         float = 0.15
    sigma_standard:      float = 0.15
    vol_surprise:        float = 0.0
    vol_regime_change_p: float = 0.5

    @property
    def crisis_probability(self) -> float:
        """
        Heuristic crisis probability from rough-vol metrics.

        P = sigmoid(roughness_score) × sigmoid(surprise_score)

        roughness_score: how far H is below 0.40 (anti-persistent crisis zone)
        surprise_score: how elevated vol_surprise is above 0.5 (unexpected shock)
        """
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        roughness_score = (0.40 - self.hurst_index) * 10   # >0 when rough
        surprise_score  = (self.vol_surprise - 0.5) * 3    # >0 when surprised

        p = sigmoid(roughness_score) * 0.5 + sigmoid(surprise_score) * 0.5
        return float(np.clip(p, 0.0, 1.0))

    @property
    def label(self) -> str:
        p = self.crisis_probability
        if p > 0.75:  return "CRISIS_IMMINENT"
        if p > 0.55:  return "ELEVATED"
        if p > 0.40:  return "WATCH"
        return "NORMAL"


@dataclass
class RegimeSignal:
    """Output from the HMM Regime Detector."""
    regime_label:  str   = "Transition"
    confidence:    float = 0.5
    is_pre_crisis: bool  = False
    is_recovery:   bool  = False

    @property
    def crisis_probability(self) -> float:
        """Map regime label + confidence to a crisis probability."""
        base = {
            "Bear / High Vol":   0.70,
            "Pre-Crisis":        0.85,
            "Transition":        0.40,
            "Bull / Low Vol":    0.10,
            "Euphoria / Bubble": 0.50,
            "Recovery":          0.20,
        }.get(self.regime_label, 0.40)
        # Confidence modulates: uncertain regime → closer to 0.5
        return float(np.clip(base * self.confidence + 0.5 * (1 - self.confidence), 0.0, 1.0))

    @property
    def label(self) -> str:
        p = self.crisis_probability
        if p > 0.75:  return "CRISIS_IMMINENT"
        if p > 0.55:  return "ELEVATED"
        if p > 0.40:  return "WATCH"
        return "NORMAL"


@dataclass
class TopologySignal:
    """Output from the TDA Topological Stress Monitor."""
    betti_0:      float = 0.0
    betti_1:      float = 0.0
    stress_score: float = 0.0
    stress_level: str   = "normal"

    @property
    def crisis_probability(self) -> float:
        """Convert stress z-score to probability via a logistic map."""
        # S=0 → p≈0.50; S=2.5 → p≈0.92; S=-2 → p≈0.12
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(sigmoid(self.stress_score - 0.5), 0.0, 1.0))

    @property
    def label(self) -> str:
        if self.stress_level == "alert":    return "CRISIS_IMMINENT"
        if self.stress_level == "elevated": return "ELEVATED"
        return "NORMAL"


@dataclass
class CombinedSignal:
    """Weighted combination of the three crisis signals."""
    rough_vol: RoughVolSignal = field(default_factory=RoughVolSignal)
    regime:    RegimeSignal   = field(default_factory=RegimeSignal)
    topology:  TopologySignal = field(default_factory=TopologySignal)

    # Default weights (regime highest — it integrates the other two)
    w_rough:    float = 0.30
    w_regime:   float = 0.40
    w_topology: float = 0.30

    @property
    def crisis_probability(self) -> float:
        return float(
            self.w_rough    * self.rough_vol.crisis_probability
            + self.w_regime * self.regime.crisis_probability
            + self.w_topology * self.topology.crisis_probability
        )

    @property
    def signals_in_agreement(self) -> int:
        """Count how many signals are above ELEVATED threshold (p > 0.55)."""
        return sum([
            self.rough_vol.crisis_probability > 0.55,
            self.regime.crisis_probability    > 0.55,
            self.topology.crisis_probability  > 0.55,
        ])

    @property
    def label(self) -> str:
        p = self.crisis_probability
        n = self.signals_in_agreement
        if p > 0.75 and n >= 2:  return "⚠️  CRISIS_IMMINENT"
        if p > 0.60 or n >= 2:   return "🔶  ELEVATED"
        if p > 0.45:              return "🟡  WATCH"
        return "🟢  NORMAL"

    def summary(self) -> str:
        lines = [
            f"{'='*52}",
            f"  CRISIS DETECTOR — Combined Signal",
            f"{'='*52}",
            f"  Crisis probability  : {self.crisis_probability:.1%}",
            f"  Signals in agreement: {self.signals_in_agreement}/3",
            f"  Status              : {self.label}",
            f"{'─'*52}",
            f"  Rough Vol  [{self.rough_vol.label:17s}]  p={self.rough_vol.crisis_probability:.2f}",
            f"    H={self.rough_vol.hurst_index:.3f}  surprise={self.rough_vol.vol_surprise:+.2f}",
            f"  Regime     [{self.regime.label:17s}]  p={self.regime.crisis_probability:.2f}",
            f"    {self.regime.regime_label}  conf={self.regime.confidence:.2f}",
            f"  Topology   [{self.topology.label:17s}]  p={self.topology.crisis_probability:.2f}",
            f"    β₀={self.topology.betti_0:.2f}  β₁={self.topology.betti_1:.2f}  S={self.topology.stress_score:.2f}σ",
            f"{'='*52}",
        ]
        return "\n".join(lines)
