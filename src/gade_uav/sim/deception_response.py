"""
Bounded causal attacker-response model for GADe-UAV deception evaluation.

This module supplies simulation semantics for attacker diversion after a
shield-admitted deceptive action.

IMPORTANT
---------
The probabilities and engagement durations are simulation parameters.
They are not claimed to be measured physical-world UAV attacker rates.

One attacker-response attempt is permitted per campaign. This prevents
repeated candidate generation from making eventual diversion artificially
approach probability 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_DIVERSION_PROBABILITY = {
    "safe_beacon": 0.55,
    "safe_telemetry": 0.45,
    "high_cost_beacon": 0.55,
}

# ------------------------------------------------------------------
# Bounded family-theme compatibility assumptions.
#
# These coefficients are simulation parameters, NOT empirical attacker
# response measurements. They are frozen before GenAI effectiveness
# evaluation so that the evaluation cannot tune them post hoc.
#
# 1.0 = neutral compatibility
# >1  = comparatively better contextual match
# <1  = comparatively weaker contextual match
# ------------------------------------------------------------------

THEME_COMPATIBILITY = {
    "BruteForce": {
        "identity_decoy": 1.10,
        "telemetry_mimic": 0.85,
        "service_lure": 1.15,
        "network_presence": 1.00,
    },
    "DDoS": {
        "identity_decoy": 0.85,
        "telemetry_mimic": 0.80,
        "service_lure": 0.90,
        "network_presence": 1.15,
    },
    "Deauthentication": {
        "identity_decoy": 1.00,
        "telemetry_mimic": 0.80,
        "service_lure": 0.85,
        "network_presence": 1.15,
    },
    "DoS": {
        "identity_decoy": 0.85,
        "telemetry_mimic": 0.80,
        "service_lure": 0.95,
        "network_presence": 1.15,
    },
    "FakeLanding": {
        "identity_decoy": 0.90,
        "telemetry_mimic": 1.20,
        "service_lure": 0.75,
        "network_presence": 0.85,
    },
    "ICMP_Flooding": {
        "identity_decoy": 0.85,
        "telemetry_mimic": 0.80,
        "service_lure": 0.90,
        "network_presence": 1.15,
    },
    "Jamming": {
        "identity_decoy": 0.70,
        "telemetry_mimic": 0.70,
        "service_lure": 0.75,
        "network_presence": 1.05,
    },
    "MITM": {
        "identity_decoy": 1.05,
        "telemetry_mimic": 1.15,
        "service_lure": 0.90,
        "network_presence": 1.10,
    },
    "Reconnaissance": {
        "identity_decoy": 1.10,
        "telemetry_mimic": 1.00,
        "service_lure": 1.15,
        "network_presence": 1.10,
    },
    "Replay": {
        "identity_decoy": 0.95,
        "telemetry_mimic": 1.20,
        "service_lure": 0.80,
        "network_presence": 0.90,
    },
    "UDP_Flooding": {
        "identity_decoy": 0.85,
        "telemetry_mimic": 0.80,
        "service_lure": 0.90,
        "network_presence": 1.15,
    },
}

MIN_DWELL_S = 0.5
MAX_DWELL_S = 2.0

# A diversion counts as a successful deception engagement only if the
# attacker remains redirected for at least this duration.
SUSTAINED_ENGAGEMENT_THRESHOLD_S = 0.5


@dataclass
class CampaignDeceptionState:
    campaign_id: str
    family: str
    start_s: float

    attempted: bool = False
    attempted_variant: str | None = None
    attempted_theme: str | None = None
    attempt_time_s: float | None = None

    base_diversion_probability: float | None = None
    theme_compatibility: float | None = None
    diversion_probability: float | None = None
    diverted: bool = False
    diversion_start_s: float | None = None
    diversion_until_s: float | None = None

    total_diverted_s: float = 0.0
    diversion_success: bool = False
    sustained_success: bool = False
    returned_to_target: bool = False

    ended: bool = False
    end_s: float | None = None


class DeceptionResponseModel:
    """
    Campaign-level causal attacker diversion model.

    Definitions
    -----------
    Attack-diversion probability:
        successful initial diversions / eligible deception attempts.

    DSR:
        campaigns sustaining decoy engagement for at least
        SUSTAINED_ENGAGEMENT_THRESHOLD_S / eligible deception attempts.

    Return-to-target:
        a successfully diverted campaign whose modeled diversion dwell
        expires before the campaign itself ends.
    """

    def __init__(
        self,
        *,
        seed: int,
        action_probabilities: dict[str, float] | None = None,
        min_dwell_s: float = MIN_DWELL_S,
        max_dwell_s: float = MAX_DWELL_S,
        sustained_threshold_s: float =
            SUSTAINED_ENGAGEMENT_THRESHOLD_S,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        self._prob = dict(
            ACTION_DIVERSION_PROBABILITY
            if action_probabilities is None
            else action_probabilities
        )

        self._min_dwell_s = float(min_dwell_s)
        self._max_dwell_s = float(max_dwell_s)
        self._sustained_threshold_s = float(
            sustained_threshold_s
        )

        if not (
            0.0
            < self._min_dwell_s
            <= self._max_dwell_s
        ):
            raise ValueError(
                "Require 0 < min_dwell_s <= max_dwell_s"
            )

        for name, p in self._prob.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"Invalid diversion probability "
                    f"{name}={p}"
                )

        self._campaigns: dict[
            str,
            CampaignDeceptionState,
        ] = {}

    def register_campaign(
        self,
        *,
        campaign_id: str,
        family: str,
        now_s: float,
    ) -> None:
        if campaign_id in self._campaigns:
            return

        self._campaigns[campaign_id] = (
            CampaignDeceptionState(
                campaign_id=campaign_id,
                family=family,
                start_s=float(now_s),
            )
        )

    def _close_active_diversion(
        self,
        state: CampaignDeceptionState,
        *,
        stop_s: float,
        returned: bool,
    ) -> None:
        if (
            not state.diverted
            or state.diversion_start_s is None
        ):
            return

        duration = max(
            0.0,
            float(stop_s)
            - state.diversion_start_s,
        )

        state.total_diverted_s += duration
        state.diverted = False
        state.diversion_start_s = None
        state.diversion_until_s = None

        if returned:
            state.returned_to_target = True

        if (
            state.total_diverted_s
            >= self._sustained_threshold_s
        ):
            state.sustained_success = True

    def advance(
        self,
        now_s: float,
    ) -> None:
        now_s = float(now_s)

        for state in self._campaigns.values():
            if (
                state.ended
                or not state.diverted
                or state.diversion_until_s is None
            ):
                continue

            if now_s >= state.diversion_until_s:
                self._close_active_diversion(
                    state,
                    stop_s=state.diversion_until_s,
                    returned=True,
                )

    def attempt(
        self,
        *,
        campaign_id: str,
        variant: str,
        now_s: float,
        theme: str | None = None,
    ) -> bool:
        """
        Apply at most one attacker-response draw per campaign.

        Returns True iff the campaign is newly diverted.
        """
        if campaign_id not in self._campaigns:
            raise KeyError(
                f"Unknown campaign {campaign_id!r}"
            )

        state = self._campaigns[campaign_id]

        if state.ended or state.attempted:
            return False

        if variant not in self._prob:
            raise ValueError(
                f"No deception probability for "
                f"variant {variant!r}"
            )

        base_p = float(
            self._prob[variant]
        )

        compatibility = 1.0

        if theme is not None:
            family_map = THEME_COMPATIBILITY.get(
                state.family
            )

            if family_map is None:
                raise ValueError(
                    f"No theme compatibility map for "
                    f"family {state.family!r}"
                )

            if theme not in family_map:
                raise ValueError(
                    f"Unknown deception theme {theme!r}"
                )

            compatibility = float(
                family_map[theme]
            )

        # Keep the resulting probability valid even if future
        # sensitivity studies change compatibility coefficients.
        p = min(
            1.0,
            max(
                0.0,
                base_p * compatibility,
            ),
        )

        state.attempted = True
        state.attempted_variant = variant
        state.attempted_theme = theme
        state.attempt_time_s = float(now_s)
        state.base_diversion_probability = base_p
        state.theme_compatibility = compatibility
        state.diversion_probability = p

        success = bool(
            self._rng.random() < p
        )

        if not success:
            return False

        dwell_s = float(
            self._rng.uniform(
                self._min_dwell_s,
                self._max_dwell_s,
            )
        )

        state.diversion_success = True
        state.diverted = True
        state.diversion_start_s = float(now_s)
        state.diversion_until_s = (
            float(now_s) + dwell_s
        )

        return True

    def is_diverted(
        self,
        campaign_id: str,
        *,
        now_s: float,
    ) -> bool:
        self.advance(now_s)

        state = self._campaigns.get(
            campaign_id
        )

        return bool(
            state is not None
            and state.diverted
            and not state.ended
        )

    def end_campaign(
        self,
        *,
        campaign_id: str,
        now_s: float,
    ) -> None:
        state = self._campaigns.get(
            campaign_id
        )

        if state is None or state.ended:
            return

        now_s = float(now_s)

        if state.diverted:
            stop_s = min(
                now_s,
                state.diversion_until_s
                if state.diversion_until_s
                is not None
                else now_s,
            )

            # Campaign ended before the attacker returned to
            # the real target.
            self._close_active_diversion(
                state,
                stop_s=stop_s,
                returned=False,
            )

        state.ended = True
        state.end_s = now_s

        if (
            state.total_diverted_s
            >= self._sustained_threshold_s
        ):
            state.sustained_success = True

    def snapshot(self) -> list[dict]:
        rows = []

        for state in self._campaigns.values():
            rows.append(
                {
                    "campaign_id":
                        state.campaign_id,
                    "family":
                        state.family,
                    "attempted":
                        state.attempted,
                    "attempted_variant":
                        state.attempted_variant,
                    "attempted_theme":
                        state.attempted_theme,
                    "base_diversion_probability":
                        state.base_diversion_probability,
                    "theme_compatibility":
                        state.theme_compatibility,
                    "diversion_probability":
                        state.diversion_probability,
                    "diversion_success":
                        state.diversion_success,
                    "sustained_success":
                        state.sustained_success,
                    "returned_to_target":
                        state.returned_to_target,
                    "total_diverted_s":
                        state.total_diverted_s,
                    "ended":
                        state.ended,
                }
            )

        return rows

    def summary(self) -> dict:
        states = list(
            self._campaigns.values()
        )

        attempted = [
            s for s in states
            if s.attempted
        ]

        diverted = [
            s for s in attempted
            if s.diversion_success
        ]

        sustained = [
            s for s in attempted
            if s.sustained_success
        ]

        returned = [
            s for s in diverted
            if s.returned_to_target
        ]

        n_attempted = len(attempted)
        n_diverted = len(diverted)
        n_sustained = len(sustained)

        return {
            "registered_campaigns":
                len(states),
            "eligible_deception_attempts":
                n_attempted,
            "diverted_campaigns":
                n_diverted,
            "sustained_engagements":
                n_sustained,
            "returned_to_target":
                len(returned),

            "attack_diversion_probability":
                (
                    n_diverted / n_attempted
                    if n_attempted
                    else None
                ),

            "deception_success_ratio":
                (
                    n_sustained / n_attempted
                    if n_attempted
                    else None
                ),

            "return_rate_given_diversion":
                (
                    len(returned) / n_diverted
                    if n_diverted
                    else None
                ),

            "mean_diversion_duration_s":
                (
                    float(
                        np.mean(
                            [
                                s.total_diverted_s
                                for s in diverted
                            ]
                        )
                    )
                    if diverted
                    else None
                ),
        }
