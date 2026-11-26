"""
Bounded local GenAI candidate generator for GADe-UAV.

The language model never emits executable UAV commands or arbitrary
Action objects. It produces only a small JSON proposal that is parsed
and mapped by trusted code onto an existing bounded defensive profile.

Designed for OFFLINE / cached generation because local CPU generation
latency is orders of magnitude above the runtime shield deadline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from gade_uav.shield.actions import (
    Action,
    ActionType,
    CostVector,
    Effector,
    GenMeta,
)


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

GENAI_TEMPLATE_ID = "genai_bounded_v1"

ALLOWED_CANDIDATE_TYPES = {
    "honey_beacon",
    "decoy_telemetry",
}

ALLOWED_PROFILES = {
    "low",
    "medium",
    "high",
}

ALLOWED_THEMES = {
    "identity_decoy",
    "telemetry_mimic",
    "service_lure",
    "network_presence",
}


@dataclass(frozen=True)
class GenerationResult:
    valid: bool
    raw_text: str
    parsed_json: dict | None
    action: Action | None
    generation_latency_ms: float
    error: str | None


class BoundedGenAIGenerator:
    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = device

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_id,
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
        )

        self.model.eval()
        self.model.to(device)

        # Avoid warnings from model generation defaults when
        # deterministic decoding is requested.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    @staticmethod
    def _context_hash(
        *,
        family: str,
        phase: str,
        decoy_id: str,
    ) -> str:
        text = (
            f"{family}|{phase}|{decoy_id}"
        )

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _prompt(
        *,
        family: str,
        phase: str,
    ) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You generate one bounded defensive decoy candidate "
                    "for an isolated UAV cybersecurity simulator. "
                    "You have no authority to control the UAV. "
                    "Return ONLY one valid JSON object. "
                    "Do not add Markdown or explanation. "
                    "Do not copy lists of choices into a field. "

                    "candidate_type must be exactly ONE of: "
                    "honey_beacon OR decoy_telemetry. "

                    "profile must be exactly ONE of: "
                    "low OR medium OR high. "

                    "theme must be exactly ONE of: "
                    "identity_decoy, telemetry_mimic, "
                    "service_lure, network_presence. "

                    "Theme meanings: "
                    "identity_decoy represents a plausible decoy identity; "
                    "telemetry_mimic represents plausible benign-looking "
                    "decoy telemetry; "
                    "service_lure represents a simulated defensive service "
                    "or interaction lure; "
                    "network_presence represents a plausible decoy network "
                    "presence. "

                    "Choose the theme that best matches the observed "
                    "attack family and mission context. "

                    "confidence must be a JSON number from 0.70 through 0.99. "
                    "label must contain only letters, digits, underscore, "
                    "or hyphen and must be at most 32 characters."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Examples of valid responses:\n"

                    '{"candidate_type":"honey_beacon",'
                    '"profile":"low",'
                    '"theme":"identity_decoy",'
                    '"label":"RECON_DECOY",'
                    '"confidence":0.82}\n'

                    '{"candidate_type":"decoy_telemetry",'
                    '"profile":"medium",'
                    '"theme":"telemetry_mimic",'
                    '"label":"REPLAY_DECOY",'
                    '"confidence":0.88}\n\n'

                    "Do not copy an example mechanically. "
                    "Select values according to the supplied context. "

                    f"Attack family: {family}. "
                    f"Mission phase: {phase}. "

                    "Return exactly five JSON fields: "
                    "candidate_type, profile, theme, label, confidence."
                ),
            },
        ]

    @staticmethod
    def _validate(
        obj: dict,
    ) -> tuple[bool, str | None]:
        required = {
            "candidate_type",
            "profile",
            "theme",
            "label",
            "confidence",
        }

        if set(obj.keys()) != required:
            return (
                False,
                "JSON keys do not match required schema",
            )

        ctype = obj.get(
            "candidate_type"
        )

        if ctype not in ALLOWED_CANDIDATE_TYPES:
            return (
                False,
                "candidate_type not allowed",
            )

        profile = obj.get(
            "profile"
        )

        if profile not in ALLOWED_PROFILES:
            return (
                False,
                "profile not allowed",
            )

        theme = obj.get(
            "theme"
        )

        if theme not in ALLOWED_THEMES:
            return (
                False,
                "theme not allowed",
            )

        label = obj.get(
            "label"
        )

        if (
            not isinstance(label, str)
            or not label
            or len(label) > 32
        ):
            return (
                False,
                "invalid label",
            )

        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        )

        if any(
            ch not in allowed_chars
            for ch in label
        ):
            return (
                False,
                "label contains disallowed characters",
            )

        confidence = obj.get(
            "confidence"
        )

        if not isinstance(
            confidence,
            (int, float),
        ):
            return (
                False,
                "confidence must be numeric",
            )

        confidence = float(
            confidence
        )

        if not (
            0.70
            <= confidence
            <= 0.99
        ):
            return (
                False,
                "confidence outside [0.70, 0.99]",
            )

        return True, None

    @staticmethod
    def _cost_for(
        *,
        candidate_type: str,
        profile: str,
    ) -> CostVector:
        """
        Trusted bounded resource profiles.

        These are simulation-side declared candidate costs, not
        LLM-generated costs.
        """

        if candidate_type == "honey_beacon":
            table = {
                "low": CostVector(
                    0.01,
                    2.0,
                    0.5,
                    48.0,
                    0.01,
                    0.1,
                ),
                "medium": CostVector(
                    0.02,
                    3.5,
                    0.9,
                    96.0,
                    0.02,
                    0.25,
                ),
                "high": CostVector(
                    0.08,
                    12.0,
                    3.0,
                    800.0,
                    0.08,
                    1.5,
                ),
            }

        else:
            table = {
                "low": CostVector(
                    0.015,
                    3.0,
                    0.8,
                    96.0,
                    0.015,
                    0.2,
                ),
                "medium": CostVector(
                    0.025,
                    4.5,
                    1.1,
                    128.0,
                    0.025,
                    0.35,
                ),
                "high": CostVector(
                    0.05,
                    8.0,
                    2.0,
                    256.0,
                    0.05,
                    0.8,
                ),
            }

        return table[profile]

    @classmethod
    def _to_action(
        cls,
        *,
        obj: dict,
        family: str,
        phase: str,
        decoy_id: str,
    ) -> Action:
        ctype = str(
            obj["candidate_type"]
        )

        profile = str(
            obj["profile"]
        )

        label = str(
            obj["label"]
        )

        confidence = float(
            obj["confidence"]
        )

        ctx_hash = cls._context_hash(
            family=family,
            phase=phase,
            decoy_id=decoy_id,
        )

        meta = GenMeta(
            confidence=confidence,
            schema_version="genai-1.0",
            generation_context_hash=
                ctx_hash,
            template_id=
                GENAI_TEMPLATE_ID,
        )

        cost = cls._cost_for(
            candidate_type=ctype,
            profile=profile,
        )

        if ctype == "honey_beacon":
            return Action(
                type=ActionType.HONEY_BEACON,
                payload={
                    "beacon_id":
                        f"genai_{decoy_id}_{label}",
                    "advertise": {
                        "ssid": label,
                    },
                },
                target_iface=
                    Effector.HONEY_BEACON_CHANNEL,
                predicted_cost=
                    cost,
                gen_meta=
                    meta,
            )

        return Action(
            type=ActionType.DECOY_TELEMETRY,
            payload={
                "decoy_id":
                    decoy_id,
                "fields": {
                    # Trusted fixed telemetry values.
                    # The language model does not create
                    # arbitrary flight-state values.
                    "alt_m": 20.0,
                    "bat_pct": 95.0,
                    "label": label,
                },
            },
            target_iface=
                Effector.DECOY_MAVLINK_CHANNEL,
            predicted_cost=
                cost,
            gen_meta=
                meta,
        )

    def generate(
        self,
        *,
        family: str,
        phase: str,
        decoy_id: str = "h0",
        max_new_tokens: int = 80,
    ) -> GenerationResult:
        messages = self._prompt(
            family=family,
            phase=phase,
        )

        prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        t0 = time.perf_counter()

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=
                    max_new_tokens,
                do_sample=False,
                pad_token_id=
                    self.tokenizer.eos_token_id,
            )

        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        generated = (
            self.tokenizer.decode(
                output[0][
                    inputs[
                        "input_ids"
                    ].shape[1]:
                ],
                skip_special_tokens=True,
            )
            .strip()
        )

        try:
            obj = json.loads(
                generated
            )
        except Exception as exc:
            return GenerationResult(
                valid=False,
                raw_text=generated,
                parsed_json=None,
                action=None,
                generation_latency_ms=
                    latency_ms,
                error=(
                    f"json_parse:"
                    f"{type(exc).__name__}"
                ),
            )

        if not isinstance(
            obj,
            dict,
        ):
            return GenerationResult(
                valid=False,
                raw_text=generated,
                parsed_json=None,
                action=None,
                generation_latency_ms=
                    latency_ms,
                error=
                    "top-level JSON is not object",
            )

        valid, error = (
            self._validate(
                obj
            )
        )

        if not valid:
            return GenerationResult(
                valid=False,
                raw_text=generated,
                parsed_json=obj,
                action=None,
                generation_latency_ms=
                    latency_ms,
                error=error,
            )

        try:
            action = self._to_action(
                obj=obj,
                family=family,
                phase=phase,
                decoy_id=decoy_id,
            )

        except Exception as exc:
            return GenerationResult(
                valid=False,
                raw_text=generated,
                parsed_json=obj,
                action=None,
                generation_latency_ms=
                    latency_ms,
                error=(
                    f"action_build:"
                    f"{type(exc).__name__}:"
                    f"{exc}"
                ),
            )

        return GenerationResult(
            valid=True,
            raw_text=generated,
            parsed_json=obj,
            action=action,
            generation_latency_ms=
                latency_ms,
            error=None,
        )

    @staticmethod
    def save_result(
        result: GenerationResult,
        path: Path,
        *,
        family: str,
        phase: str,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        row = {
            "family":
                family,
            "phase":
                phase,
            "valid":
                result.valid,
            "raw_text":
                result.raw_text,
            "parsed_json":
                result.parsed_json,
            "generation_latency_ms":
                result.generation_latency_ms,
            "error":
                result.error,
        }

        path.write_text(
            json.dumps(
                row,
                indent=2,
                sort_keys=True,
            )
        )
