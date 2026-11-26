"""Canonical label taxonomy for GADe-UAV.

Single source of truth for attack labels across UAV-NIDD, UAV Attack Dataset,
and GUIDE MAVLink sequences. Every downstream preprocessing and evaluation
script imports from here. Do not define labels elsewhere.

Design decisions locked in Phase 0:
    1. Canonical labels correspond to threat-model families T1-T17.
    2. Fine-grained variants (e.g., DDoS vs DoS vs UDP Flooding) preserved.
    3. Normal is a first-class label.
    4. Null/missing labels are dropped with a logged count.
    5. Raw dataset labels are normalized via explicit mapping tables,
       not fuzzy matching. New raw labels must be added by amendment.

See docs/01_threat_model.md for the threat-model rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttackGroup(str, Enum):
    """Threat-model groupings from docs/01_threat_model.md §3.2."""

    NORMAL = "G0_Normal"
    RECONNAISSANCE = "G1_Reconnaissance"
    NETWORK_DOS = "G2_NetworkLayerDoS"
    MITM_CREDENTIAL = "G3_MITM_Credential"
    CYBER_PHYSICAL = "G4_CyberPhysical"
    MISSION_PROTOCOL = "G5_MissionProtocol"


@dataclass(frozen=True)
class CanonicalLabel:
    """A canonical label with its threat-model identifier and group.

    threat_id: T-code from docs/01_threat_model.md §3.1 (T1-T17), or 'T0' for Normal.
    canonical: canonical string form used in all interim/processed data.
    group: analysis group for reporting.
    """

    threat_id: str
    canonical: str
    group: AttackGroup


# ---------------------------------------------------------------------------
# Canonical taxonomy — single source of truth
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, CanonicalLabel] = {
    label.canonical: label
    for label in [
        CanonicalLabel("T0", "Normal", AttackGroup.NORMAL),
        # G1: Reconnaissance
        CanonicalLabel("T1", "Reconnaissance", AttackGroup.RECONNAISSANCE),
        CanonicalLabel("T1", "Scanning", AttackGroup.RECONNAISSANCE),
        # G2: Network-Layer DoS
        CanonicalLabel("T2", "DoS", AttackGroup.NETWORK_DOS),
        CanonicalLabel("T3", "DDoS", AttackGroup.NETWORK_DOS),
        CanonicalLabel("T4", "UDP_Flooding", AttackGroup.NETWORK_DOS),
        CanonicalLabel("T5", "ICMP_Flooding", AttackGroup.NETWORK_DOS),
        CanonicalLabel("T6", "Deauthentication", AttackGroup.NETWORK_DOS),
        # G3: MITM + Credential
        CanonicalLabel("T7", "MITM", AttackGroup.MITM_CREDENTIAL),
        CanonicalLabel("T8", "Replay", AttackGroup.MITM_CREDENTIAL),
        CanonicalLabel("T9", "BruteForce", AttackGroup.MITM_CREDENTIAL),
        CanonicalLabel("T18", "EvilTwin", AttackGroup.MITM_CREDENTIAL),
        # G4: Cyber-Physical
        CanonicalLabel("T10", "Jamming", AttackGroup.CYBER_PHYSICAL),
        CanonicalLabel("T11", "GPS_Spoofing", AttackGroup.CYBER_PHYSICAL),
        # G5: Mission-Specific + Protocol-Level
        CanonicalLabel("T12", "FakeLanding", AttackGroup.MISSION_PROTOCOL),
        CanonicalLabel(
            "T13", "MAVLink_Heartbeat_Flooding", AttackGroup.MISSION_PROTOCOL
        ),
        CanonicalLabel("T14", "MAVLink_Ping_Flooding", AttackGroup.MISSION_PROTOCOL),
        CanonicalLabel("T15", "MAVLink_Request_Flooding", AttackGroup.MISSION_PROTOCOL),
        CanonicalLabel("T16", "MAVLink_Corruption", AttackGroup.MISSION_PROTOCOL),
        CanonicalLabel("T17", "MAVLink_Suspension", AttackGroup.MISSION_PROTOCOL),
    ]
}

CANONICAL_NAMES: frozenset[str] = frozenset(TAXONOMY.keys())


# ---------------------------------------------------------------------------
# Raw-to-canonical mappings — per dataset
# ---------------------------------------------------------------------------

# UAV-NIDD raw labels observed in the smoke test.
# Left column = raw label string as it appears in the CSVs.
# Right column = canonical form (must exist in TAXONOMY).
UAV_NIDD_MAPPING: dict[str, str] = {
    # --- Normal ---
    "Normal": "Normal",
    "Benign": "Normal",  # GSC view uses "Benign" for normal traffic
    # --- Reconnaissance / Scanning (T1) ---
    "Scanning": "Scanning",
    "Reconnaissance": "Reconnaissance",  # AP & GSC views
    "Reconnassiance": "Reconnaissance",  # UAV view typo
    # --- Network-layer DoS (T2-T6) ---
    "DoS": "DoS",
    "DDoS": "DDoS",
    "UDP Flooding": "UDP_Flooding",
    "ICMP Flooding": "ICMP_Flooding",
    "De-authentication": "Deauthentication",
    # --- MITM / Credential (T7-T9, T18) ---
    "MITM": "MITM",
    "Replay": "Replay",  # AP view
    "replay": "Replay",  # UAV view lowercase
    "BruteForce": "BruteForce",  # UAV view
    "Bruteforce": "BruteForce",  # GSC view
    "Brute-Force": "BruteForce",  # AP view
    "Evil Twin": "EvilTwin",  # GSC view
    "Ewil Twin": "EvilTwin",  # AP view typo
    "EvilTwin": "EvilTwin",
    "evil twin": "EvilTwin",
    # --- Cyber-physical (T10-T11) ---
    "Jamming": "Jamming",
    # --- Mission / Protocol (T12-T17) ---
    "FakeLanding": "FakeLanding",  # UAV view
    "Fake Landing": "FakeLanding",  # AP & GSC views (trailing spaces stripped by normalize)
}

# UAV Attack Dataset labels are inferred from directory structure, not from a
# single label column. This mapping is applied to the directory-derived label.
# UAV Attack Dataset labels are inferred from the leaf directory name.
# Two sub-datasets share this mapping:
#   - "Live GPS Spoofing and Jamming/{Benign Flight|GPS Jamming|GPS Spoofing}"
#   - "Simulated - OTU Survey/{airframe}/{Normal|GPS Spoofing|Ping DoS}"
UAV_ATTACK_MAPPING: dict[str, str] = {
    # Normal / benign flight
    "Benign Flight": "Normal",  # Live sub-dataset
    "Normal": "Normal",  # Simulated sub-dataset
    # Cyber-physical (T10, T11)
    "GPS Jamming": "Jamming",
    "GPS Spoofing": "GPS_Spoofing",
    # Network-layer DoS (T2) -- Ping flood targeting the GCS link
    "Ping DoS": "DoS",
}

# GUIDE labels come from filename patterns (heartbeat, ping, request,
# corruption, suspension, normal). All are MAVLink protocol-level attacks.
GUIDE_MAPPING: dict[str, str] = {
    "normal": "Normal",
    "heartbeat": "MAVLink_Heartbeat_Flooding",
    "ping": "MAVLink_Ping_Flooding",
    "request": "MAVLink_Request_Flooding",
    "corruption": "MAVLink_Corruption",
    "suspension": "MAVLink_Suspension",
}


# ---------------------------------------------------------------------------
# Normalization API
# ---------------------------------------------------------------------------


class UnknownLabelError(ValueError):
    """Raised when a raw label has no canonical mapping.

    This is intentional: unknown labels must be added to the mapping table
    explicitly, not silently defaulted, so that dataset drift is caught.
    """


def normalize_uav_nidd(raw: str | None) -> str | None:
    """Normalize a UAV-NIDD raw label to canonical form.

    Returns None if the raw label is None (null in the CSV).
    Raises UnknownLabelError for unrecognized non-null labels.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped == "" or stripped.lower() == "null":
        return None
    if stripped not in UAV_NIDD_MAPPING:
        raise UnknownLabelError(f"UAV-NIDD: unknown raw label {stripped!r}")
    return UAV_NIDD_MAPPING[stripped]


def normalize_uav_attack(directory_segment: str) -> str:
    """Normalize a UAV Attack Dataset directory segment to canonical form."""
    if directory_segment not in UAV_ATTACK_MAPPING:
        raise UnknownLabelError(
            f"UAV Attack: unknown directory segment {directory_segment!r}"
        )
    return UAV_ATTACK_MAPPING[directory_segment]


def normalize_guide(filename: str) -> str:
    """Normalize a GUIDE .npy filename to canonical form.

    Extracts the attack-type keyword from the filename (e.g.,
    'heartbeat(filtered)_sequences.npy' -> 'heartbeat').
    """
    for keyword, canonical in GUIDE_MAPPING.items():
        if keyword in filename.lower():
            return canonical
    raise UnknownLabelError(f"GUIDE: no attack keyword in filename {filename!r}")


def group_of(canonical: str) -> AttackGroup:
    """Return the analysis group for a canonical label."""
    if canonical not in TAXONOMY:
        raise UnknownLabelError(f"Unknown canonical label {canonical!r}")
    return TAXONOMY[canonical].group


def threat_id_of(canonical: str) -> str:
    """Return the threat-model T-code for a canonical label."""
    if canonical not in TAXONOMY:
        raise UnknownLabelError(f"Unknown canonical label {canonical!r}")
    return TAXONOMY[canonical].threat_id


def all_canonical_labels() -> list[str]:
    """Return canonical labels in a stable order (by threat_id)."""
    return sorted(TAXONOMY.keys(), key=lambda name: (TAXONOMY[name].threat_id, name))


def summary() -> dict[str, dict]:
    """Return a summary of the taxonomy for logging/export."""
    return {
        canonical: {
            "threat_id": lbl.threat_id,
            "group": lbl.group.value,
        }
        for canonical, lbl in TAXONOMY.items()
    }
