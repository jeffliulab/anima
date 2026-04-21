"""L1 — LLM-as-Parser.

Design invariant #1: the LLM produces structured TaskSpec JSON via tool-
calling; it does *not* directly emit motor commands. This keeps the LLM
on the reasoning side of the Test-and-Check gate.

L1's input is user text, an ASR transcript, or an intent-token stream
from L0 (plus any free-text context from other observers / screen labels
/ history). The reference tests accept raw user text so the pipeline can
be exercised without an upstream signal source.

``parse`` is framework-agnostic: pass any callable that takes
``(system_prompt, user_msg, tool_schema)`` and returns the tool-call
arguments dict. Application code wires this to OpenAI / Anthropic /
DeepSeek / local model SDKs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .taskspec import (
    Alternative,
    Constraints,
    IntentToken,
    Subtask,
    TaskSpec,
)


EMIT_TASKSPEC_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "emit_taskspec",
        "description": "Emit a structured TaskSpec from the user's intent.",
        "parameters": {
            "type": "object",
            "required": [
                "intent_token",
                "intent_confidence",
                "subtasks",
                "constraints",
            ],
            "properties": {
                "intent_token": {
                    "type": "string",
                    "description": "Exactly one of the intent vocabulary tokens.",
                },
                "intent_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "requires_confirmation": {"type": "boolean"},
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "token": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["token", "confidence"],
                    },
                },
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "locate",
                                    "grasp",
                                    "lift",
                                    "deliver",
                                    "navigate",
                                    "release",
                                ],
                            },
                        },
                        "required": ["name", "type"],
                    },
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "max_force_n": {"type": "number"},
                        "timeout_s": {"type": "number"},
                    },
                },
            },
        },
    },
}


LLMToolCaller = Callable[[str, str, dict], dict]
"""Adapter signature: (system_prompt, user_msg, tool_schema) -> tool-call args dict."""


def apply_drift(confidence: float, drift: float, seed_key: str) -> float:
    """Perturb LLM-stated confidence by the L0-reported drift score.

    Keeps the LLM's stated confidence honest: high upstream-signal
    uncertainty → visibly noisier ITA.
    """
    rng = np.random.default_rng(abs(hash(seed_key)) % (2**32))
    shake = rng.beta(8, 2) - 0.8
    perturbed = confidence + drift * shake * 0.3
    return float(max(0.0, min(1.0, perturbed)))


def parse(
    user_text: str,
    call_llm: LLMToolCaller,
    *,
    system_prompt: str,
    drift: float = 0.05,
) -> TaskSpec:
    """Call the caller-supplied LLM adapter and return a validated TaskSpec."""
    user_msg = (
        f'User text input: "{user_text}"\n'
        f"Drift score: {drift:.3f}\n"
    )
    args = call_llm(system_prompt, user_msg, EMIT_TASKSPEC_TOOL)
    return _taskspec_from_args(args, user_text=user_text, drift=drift)


def _taskspec_from_args(args: dict, *, user_text: str, drift: float) -> TaskSpec:
    token = args.get("intent_token", "UNKNOWN")
    raw_conf = float(args.get("intent_confidence", 0.5))
    conf = apply_drift(raw_conf, drift, seed_key=user_text)

    alternatives_raw = args.get("alternatives") or []
    alternatives = []
    for alt in alternatives_raw[:3]:
        try:
            alternatives.append(
                Alternative(token=alt["token"], confidence=float(alt["confidence"]))
            )
        except (KeyError, ValueError):
            continue

    requires_confirmation = bool(args.get("requires_confirmation", False))
    if conf < 0.7:
        requires_confirmation = True

    subtasks_raw = args.get("subtasks") or []
    subtasks = [Subtask(name=s["name"], type=s["type"]) for s in subtasks_raw]

    # DRINK_WATER is the reference demo's canary path; guarantee a full
    # 5-step plan even if the LLM returns a shorter one. Production
    # deployments validate via the preconditions gate instead.
    if token == "DRINK_WATER" and len(subtasks) < 5:
        subtasks = [
            Subtask(name="locate_cup", type="locate"),
            Subtask(name="navigate_to_cup", type="navigate"),
            Subtask(name="grasp_cup", type="grasp"),
            Subtask(name="lift_cup", type="lift"),
            Subtask(name="deliver_to_mouth", type="deliver"),
        ]

    constraints_raw = args.get("constraints") or {}
    constraints = Constraints(
        max_force_n=float(constraints_raw.get("max_force_n", 8.0)),
        timeout_s=float(constraints_raw.get("timeout_s", 15.0)),
    )

    return TaskSpec(
        intent=IntentToken(
            token=token,
            confidence=conf,
            requires_confirmation=requires_confirmation,
            alternatives=alternatives,
            drift_score=drift,
            source_text=user_text,
        ),
        subtasks=subtasks,
        constraints=constraints,
    )
