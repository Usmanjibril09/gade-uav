"""Simulation loop skeleton.

Advances the SimClock, ticks every registered drone, syncs new
positions into Grid. No adversary, no effector, no shield hook yet
(those arrive in patches 2-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gade_uav.shield.actions import Action, CostVector
from gade_uav.shield.decide import Shield, ShieldContext, ShieldDecision, Verdict

from .adversary import Adversary, AttackEvent
from .clock import SimClock
from .drone import Drone
from .effector import Effector
from .grid import Grid
from .mission import Mission
from .mission_controller import MissionGuidanceController
from .mission_impact import MissionImpact
from .recorder import Recorder


@dataclass
class SimulationLoop:
    grid: Grid
    clock: SimClock = field(default_factory=SimClock)
    _drones: dict[str, Drone] = field(default_factory=dict)
    _adversary: Adversary | None = None
    _last_events: list[AttackEvent] = field(default_factory=list)
    _shield: Shield | None = None
    _effector: Effector | None = None
    _recorder: Recorder | None = None
    _missions: dict[str, Mission] = field(default_factory=dict)
    _mission_guidance: MissionGuidanceController = field(
        default_factory=MissionGuidanceController
    )
    _mission_impact: MissionImpact = field(
        default_factory=MissionImpact.zero
    )

    def attach_mission(self, drone_id: str, mission: Mission) -> None:
        """Bind a Mission to an already-added MissionDrone."""
        if drone_id not in self._drones:
            raise KeyError(f"drone_id {drone_id!r} not added")
        self._missions[drone_id] = mission

    def attach_adversary(self, adv: Adversary) -> None:
        """Attach an Adversary. Its poll() is called every step()."""
        self._adversary = adv

    def attach_shield(self, shield: Shield) -> None:
        self._shield = shield

    def attach_effector(self, effector: Effector) -> None:
        self._effector = effector

    def attach_recorder(self, recorder: Recorder) -> None:
        self._recorder = recorder

    def set_mission_impact(self, impact: MissionImpact) -> None:
        """Set bounded mission impact for subsequent simulation steps."""
        self._mission_impact = impact.clipped()

    def clear_mission_impact(self) -> None:
        """Restore nominal mission behavior."""
        self._mission_impact = MissionImpact.zero()

    @property
    def mission_impact(self) -> MissionImpact:
        """Current bounded mission-impact state."""
        return self._mission_impact

    @property
    def last_events(self) -> list[AttackEvent]:
        """AttackEvents emitted during the most recent step(). Read-only view."""
        return list(self._last_events)

    def deploy(
        self,
        action: Action,
        context: ShieldContext,
        *,
        primary_attack_family: str | None = None,
        campaign_id: str | None = None,
    ) -> ShieldDecision:
        """Orchestrator entry point: shield-check, execute if admitted, record.

        Returns the ShieldDecision so callers can react.
        Requires attach_shield() and attach_effector() to have been called.
        """
        if self._shield is None:
            raise RuntimeError("deploy() requires attach_shield() first")
        if self._effector is None:
            raise RuntimeError("deploy() requires attach_effector() first")
        now_s = self.clock.now_s()
        decision = self._shield.decide(
            action, context, now_s=now_s, primary_attack_family=primary_attack_family
        )
        observed: CostVector | None = None
        # Run effector on any action that reached L3 — i.e., admitted (DEPLOY)
        # or rejected with an L3 gate result. This gives us counterfactual cost for
        # L3-rejects, which M3 needs to learn from (breaks selection bias per
        # Jin & Ren 2024, JRSS-B). L1/L2 rejects are NOT executed since those
        # actions are forbidden by construction — running them would violate
        # the shield's safety contract.
        reached_l3 = decision.verdict == Verdict.DEPLOY or (
            decision.verdict == Verdict.REJECT
            and decision.layer_failed == "L3"
            and decision.gate_decision is not None
        )
        if reached_l3:
            observed = self._effector.execute(
                decision.action_out or action,
                mission_phase=context.mission_phase.value,
                primary_attack_family=primary_attack_family,
            )
        if decision.verdict == Verdict.DEPLOY:
            # Feed observed cost back into the shield's conformal calibration.
            # Only DEPLOY outcomes feed calibration — L3-rejected observations
            # are counterfactuals for training M3, not for online recalibration.
            self._shield.observe_cost_outcome(decision, observed)
        if self._recorder is not None:
            self._recorder.record(
                t_s=now_s,
                tick=self.clock.tick,
                action=decision.action_out or action,
                decision=decision,
                observed_cost=observed,
                mission_phase=context.mission_phase.value,
                primary_attack_family=primary_attack_family,
                campaign_id=campaign_id,
            )
        return decision

    def add_drone(self, drone: Drone) -> None:
        if drone.drone_id in self._drones:
            raise ValueError(f"drone_id {drone.drone_id!r} already added")
        self._drones[drone.drone_id] = drone
        self.grid.register(drone.drone_id, drone.position)

    def step(self) -> float:
        """Advance one dt. Returns new now_s()."""
        dt = self.clock.dt_s

        # Command mission-drone velocities before kinematic integration.
        # Mission.step() still owns all mission-phase transitions.
        for drone_id, mission in self._missions.items():
            drone = self._drones[drone_id]
            self._mission_guidance.command(
                drone,
                mission,
                speed_scale=self._mission_impact.speed_scale,
            )

        for drone in self._drones.values():
            new_pos = drone.tick(dt)
            # Clamp to grid bounds instead of raising; drones near the
            # boundary just stop moving in the offending axis.
            clamped = self._clamp_to_grid(new_pos)
            if clamped != new_pos:
                drone.position = clamped
            self.grid.update_position(drone.drone_id, clamped)

            # Additional attack-induced energy burden applies only to
            # mission drones; nominal drain already occurs in Drone.tick().
            if drone.drone_id in self._missions:
                extra_drain = (
                    self._mission_impact.energy_extra_pct_per_s * dt
                )
                drone.battery_pct = max(
                    0.0,
                    drone.battery_pct - extra_drain,
                )
        # Advance any attached missions (mutates drone.mission_phase).
        now_s_for_missions = self.clock.now_s()
        for drone_id, mission in self._missions.items():
            drone = self._drones[drone_id]
            # Only MissionDrone subclass is expected here; skip others.
            if hasattr(drone, "mission_phase"):
                mission.step(drone, now_s_for_missions)  # type: ignore[arg-type]
        # Poll adversary against current drone roster.
        if self._adversary is not None and self._drones:
            self._last_events = self._adversary.poll(
                now_s=self.clock.now_s(),
                dt_s=dt,
                targetable_drone_ids=list(self._drones.keys()),
            )
        else:
            self._last_events = []
        return self.clock.advance()

    def _clamp_to_grid(self, pos):
        from .grid import Position2D

        return Position2D(
            x=min(max(pos.x, 0.0), self.grid.width_m),
            y=min(max(pos.y, 0.0), self.grid.height_m),
            z=max(0.0, pos.z),
        )

    def run(self, n_ticks: int) -> None:
        for _ in range(n_ticks):
            self.step()

    @property
    def drones(self) -> dict[str, Drone]:
        return dict(self._drones)
