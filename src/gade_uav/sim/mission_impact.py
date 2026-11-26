"""
Bounded mission-impact and defensive-attenuation model for GADe-UAV.

This module introduces a transparent causal bridge:

    active attack campaign
        -> modeled mission impact
        -> defensive attenuation
        -> residual mission impact

It contains NO learned parameters and requires NO training.

Important:
- Coefficients are simulation parameters, not claimed physical constants.
- All effects are bounded.
- The model is intended for controlled sensitivity experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MissionImpact:
    """
    Per-tick mission impact.

    command_delay_ms:
        additional command-path delay.

    link_loss_prob:
        bounded probability-like link degradation in [0, 1].

    energy_extra_pct_per_s:
        additional battery percentage drain per simulated second.

    speed_scale:
        multiplicative mission-velocity factor in (0, 1].
        1.0 means no navigation/progress degradation.
    """

    command_delay_ms: float = 0.0
    link_loss_prob: float = 0.0
    energy_extra_pct_per_s: float = 0.0
    speed_scale: float = 1.0

    def clipped(self) -> "MissionImpact":
        return MissionImpact(
            command_delay_ms=max(0.0, min(self.command_delay_ms, 250.0)),
            link_loss_prob=max(0.0, min(self.link_loss_prob, 0.95)),
            energy_extra_pct_per_s=max(
                0.0,
                min(self.energy_extra_pct_per_s, 1.0),
            ),
            speed_scale=max(0.10, min(self.speed_scale, 1.0)),
        )

    @staticmethod
    def zero() -> "MissionImpact":
        return MissionImpact()


# ---------------------------------------------------------------------
# Nominal attack-impact profiles
# ---------------------------------------------------------------------
#
# These are controlled simulation assumptions.
# They must later be subjected to low/nominal/high sensitivity analysis.
#
# Reconnaissance intentionally has very low direct mission impact.
# FakeLanding has the strongest progress/navigation effect.
# Jamming / DoS-family attacks primarily affect link/command quality.
# ---------------------------------------------------------------------

ATTACK_IMPACT_PROFILES: dict[str, MissionImpact] = {
    "BruteForce": MissionImpact(
        command_delay_ms=8.0,
        link_loss_prob=0.05,
        energy_extra_pct_per_s=0.010,
        speed_scale=0.98,
    ),
    "DDoS": MissionImpact(
        command_delay_ms=35.0,
        link_loss_prob=0.25,
        energy_extra_pct_per_s=0.020,
        speed_scale=0.90,
    ),
    "Deauthentication": MissionImpact(
        command_delay_ms=45.0,
        link_loss_prob=0.40,
        energy_extra_pct_per_s=0.010,
        speed_scale=0.82,
    ),
    "DoS": MissionImpact(
        command_delay_ms=30.0,
        link_loss_prob=0.22,
        energy_extra_pct_per_s=0.015,
        speed_scale=0.90,
    ),
    "FakeLanding": MissionImpact(
        command_delay_ms=12.0,
        link_loss_prob=0.08,
        energy_extra_pct_per_s=0.010,
        speed_scale=0.65,
    ),
    "ICMP_Flooding": MissionImpact(
        command_delay_ms=22.0,
        link_loss_prob=0.15,
        energy_extra_pct_per_s=0.015,
        speed_scale=0.93,
    ),
    "Jamming": MissionImpact(
        command_delay_ms=55.0,
        link_loss_prob=0.45,
        energy_extra_pct_per_s=0.020,
        speed_scale=0.75,
    ),
    "MITM": MissionImpact(
        command_delay_ms=28.0,
        link_loss_prob=0.18,
        energy_extra_pct_per_s=0.012,
        speed_scale=0.88,
    ),
    "Reconnaissance": MissionImpact(
        command_delay_ms=2.0,
        link_loss_prob=0.01,
        energy_extra_pct_per_s=0.002,
        speed_scale=0.995,
    ),
    "Replay": MissionImpact(
        command_delay_ms=18.0,
        link_loss_prob=0.12,
        energy_extra_pct_per_s=0.008,
        speed_scale=0.92,
    ),
    "UDP_Flooding": MissionImpact(
        command_delay_ms=32.0,
        link_loss_prob=0.23,
        energy_extra_pct_per_s=0.018,
        speed_scale=0.90,
    ),
}


# ---------------------------------------------------------------------
# Defensive effectiveness
# ---------------------------------------------------------------------
#
# Values represent the fraction of modeled attack impact attenuated by an
# admitted defensive/deception action.
#
# They are deliberately bounded below 1.0.
# These are simulation parameters and must be sensitivity-tested.
# ---------------------------------------------------------------------

DEFAULT_DEFENSE_EFFECTIVENESS: dict[str, float] = {
    "BruteForce": 0.45,
    "DDoS": 0.55,
    "Deauthentication": 0.45,
    "DoS": 0.50,
    "FakeLanding": 0.35,
    "ICMP_Flooding": 0.50,
    "Jamming": 0.40,
    "MITM": 0.55,
    "Reconnaissance": 0.30,
    "Replay": 0.50,
    "UDP_Flooding": 0.55,
}


@dataclass(frozen=True)
class ImpactSample:
    """One realized pre/post-defense mission-impact sample."""

    family: str | None
    pre_defense: MissionImpact
    post_defense: MissionImpact
    effectiveness: float
    defense_active: bool


class MissionImpactModel:
    """
    Deterministic-under-seed bounded mission-impact model.

    intensity:
        global multiplier applied to attack severity.
        Suggested experiments:
            low     = 0.5
            nominal = 1.0
            high    = 1.5
    """

    def __init__(
        self,
        *,
        seed: int,
        intensity: float = 1.0,
        profiles: Mapping[str, MissionImpact] | None = None,
        defense_effectiveness: Mapping[str, float] | None = None,
        noise_fraction: float = 0.10,
    ) -> None:
        if intensity < 0:
            raise ValueError("intensity must be >= 0")

        if not (0.0 <= noise_fraction <= 0.5):
            raise ValueError("noise_fraction must be in [0, 0.5]")

        self._rng = np.random.default_rng(seed)
        self.intensity = float(intensity)
        self.profiles = dict(profiles or ATTACK_IMPACT_PROFILES)
        self.defense_effectiveness = dict(
            defense_effectiveness or DEFAULT_DEFENSE_EFFECTIVENESS
        )
        self.noise_fraction = float(noise_fraction)

    def _noise(self) -> float:
        """
        Multiplicative bounded noise centered at 1.0.
        """
        f = self.noise_fraction
        if f == 0:
            return 1.0
        return float(self._rng.uniform(1.0 - f, 1.0 + f))

    def pre_defense_impact(
        self,
        family: str | None,
    ) -> MissionImpact:
        if family is None:
            return MissionImpact.zero()

        base = self.profiles.get(family)

        if base is None:
            # Unknown context is conservative but does not invent a severe effect.
            return MissionImpact.zero()

        severity = self.intensity * self._noise()

        # For speed_scale, convert degradation (1 - scale), multiply that,
        # then convert back.
        speed_degradation = (
            1.0 - base.speed_scale
        ) * severity

        return MissionImpact(
            command_delay_ms=
                base.command_delay_ms * severity,
            link_loss_prob=
                base.link_loss_prob * severity,
            energy_extra_pct_per_s=
                base.energy_extra_pct_per_s * severity,
            speed_scale=
                1.0 - speed_degradation,
        ).clipped()

    def attenuate(
        self,
        family: str | None,
        impact: MissionImpact,
        *,
        defense_active: bool,
        effectiveness_scale: float = 1.0,
    ) -> ImpactSample:
        if family is None or not defense_active:
            return ImpactSample(
                family=family,
                pre_defense=impact,
                post_defense=impact,
                effectiveness=0.0,
                defense_active=False,
            )

        eta = float(
            self.defense_effectiveness.get(
                family,
                0.0,
            )
        )

        eta *= float(effectiveness_scale)
        eta = max(0.0, min(eta, 0.95))

        residual = 1.0 - eta

        # speed_scale is handled through degradation.
        pre_speed_deg = 1.0 - impact.speed_scale
        post_speed_deg = pre_speed_deg * residual

        post = MissionImpact(
            command_delay_ms=
                impact.command_delay_ms * residual,
            link_loss_prob=
                impact.link_loss_prob * residual,
            energy_extra_pct_per_s=
                impact.energy_extra_pct_per_s * residual,
            speed_scale=
                1.0 - post_speed_deg,
        ).clipped()

        return ImpactSample(
            family=family,
            pre_defense=impact,
            post_defense=post,
            effectiveness=eta,
            defense_active=True,
        )

    def sample(
        self,
        family: str | None,
        *,
        defense_active: bool,
        effectiveness_scale: float = 1.0,
    ) -> ImpactSample:
        pre = self.pre_defense_impact(family)

        return self.attenuate(
            family,
            pre,
            defense_active=defense_active,
            effectiveness_scale=effectiveness_scale,
        )


def scalar_impact(impact: MissionImpact) -> float:
    """
    Dimensionless normalized impact score for mitigation analysis.

    This is intentionally a secondary aggregate metric.
    Paper reporting should retain the component outcomes separately.

    Component normalizers correspond to the hard clipping ranges above.
    """
    delay = impact.command_delay_ms / 250.0
    loss = impact.link_loss_prob / 0.95
    energy = impact.energy_extra_pct_per_s / 1.0
    speed_degradation = (1.0 - impact.speed_scale) / 0.90

    return float(
        0.30 * delay
        + 0.30 * loss
        + 0.15 * energy
        + 0.25 * speed_degradation
    )
