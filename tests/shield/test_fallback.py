"""Tests for the fallback policy loader (spec §8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gade_uav.shield.actions import MissionPhase
from gade_uav.shield.fallback import (
    KNOWN_ATTACK_FAMILIES,
    UNKNOWN_FAMILY_KEY,
    FallbackAction,
    FallbackPolicy,
)


REPO_ROOT = Path(__file__).parents[2]
YAML_PATH = REPO_ROOT / "configs" / "fallback_policy.yaml"


class TestYamlLoader:
    def test_loads_shipped_config(self):
        policy = FallbackPolicy.from_yaml(YAML_PATH)
        assert policy.default_action == FallbackAction.HAND_OFF_TO_IDS
        # All 18 known families + unknown are present.
        assert set(policy.table.keys()) == KNOWN_ATTACK_FAMILIES | {UNKNOWN_FAMILY_KEY}
        # Every family has every phase.
        for family, per_phase in policy.table.items():
            assert set(per_phase.keys()) == set(MissionPhase), family

    def test_rejects_top_level_non_mapping(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("just a string")
        with pytest.raises(ValueError, match="top-level must be a mapping"):
            FallbackPolicy.from_yaml(bad)

    def test_rejects_missing_default_action(self, tmp_path):
        raw = yaml.safe_load(YAML_PATH.read_text())
        del raw["default_action"]
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="default_action must be a string"):
            FallbackPolicy.from_yaml(bad)

    def test_rejects_missing_attack_family(self, tmp_path):
        raw = yaml.safe_load(YAML_PATH.read_text())
        del raw["table"]["T7_MITM"]
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="missing attack families"):
            FallbackPolicy.from_yaml(bad)

    def test_rejects_missing_phase_in_family(self, tmp_path):
        raw = yaml.safe_load(YAML_PATH.read_text())
        del raw["table"]["T7_MITM"]["cruise"]
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match=r"missing phases"):
            FallbackPolicy.from_yaml(bad)

    def test_rejects_unknown_attack_family(self, tmp_path):
        raw = yaml.safe_load(YAML_PATH.read_text())
        raw["table"]["T99_Bogus"] = raw["table"]["T7_MITM"]
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="unknown attack families"):
            FallbackPolicy.from_yaml(bad)

    def test_rejects_unknown_phase(self, tmp_path):
        raw = yaml.safe_load(YAML_PATH.read_text())
        raw["table"]["T7_MITM"]["bogus_phase"] = "NO_OP"
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="unknown phase"):
            FallbackPolicy.from_yaml(bad)

    def test_rejects_unknown_action(self, tmp_path):
        raw = yaml.safe_load(YAML_PATH.read_text())
        raw["table"]["T7_MITM"]["cruise"] = "BOGUS_ACTION"
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="unknown"):
            FallbackPolicy.from_yaml(bad)


class TestSelect:
    def test_known_family_returns_exact_cell(self):
        policy = FallbackPolicy.from_yaml(YAML_PATH)
        # From configs/fallback_policy.yaml:
        # T12_FakeLanding is NO_OP in every phase.
        for phase in MissionPhase:
            assert policy.select("T12_FakeLanding", phase) == FallbackAction.NO_OP
        # T7_MITM is HAND_OFF_TO_IDS in every phase.
        for phase in MissionPhase:
            assert policy.select("T7_MITM", phase) == FallbackAction.HAND_OFF_TO_IDS

    def test_unknown_family_uses_unknown_row(self):
        policy = FallbackPolicy.from_yaml(YAML_PATH)
        r = policy.select("T404_NotAFamily", MissionPhase.CRUISE)
        # The unknown row in the shipped config sets cruise -> HAND_OFF_TO_IDS
        assert r == FallbackAction.HAND_OFF_TO_IDS

    def test_none_family_uses_unknown_row(self):
        policy = FallbackPolicy.from_yaml(YAML_PATH)
        r = policy.select(None, MissionPhase.TAKEOFF)
        # unknown/takeoff -> NO_OP in the shipped config
        assert r == FallbackAction.NO_OP
