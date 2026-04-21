"""Test-and-Check — the six-gate validation layer between L1 and L2.

Design invariant #2: every TaskSpec passes six gates before execution.
A single failed gate aborts the task and routes it to a fallback channel
(user confirmation / caregiver review / non-signal-path E-stop) depending
on which gate failed.

  1. JSON gate           — TaskSpec is syntactically valid.
  2. Intent gate         — token ∈ vocabulary; not UNKNOWN.
  3. Skill gate          — every subtask.type exists in the registry.
  4. Params gate         — numeric bounds (force, timeout) are sane.
  5. Safety gate         — ITA ≥ threshold; drift ≤ threshold.
  6. Preconditions gate  — domain-specific (e.g. target object must be
                           locatable before a grasp subtask is enqueued).

This module implements gates 1-5 as pure functions. Gate 6 is deferred
to the caller because it needs runtime context (a scene, a sensor feed,
a user record). Safety-critical applications (medical, industrial)
raise the gate thresholds in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .taskspec import TaskSpec


@dataclass(frozen=True)
class GateResult:
    ok: bool
    gate: str
    reason: str = ""


DEFAULT_ITA_MIN = 0.70
DEFAULT_DRIFT_MAX = 0.35
DEFAULT_MAX_FORCE_N = 15.0
DEFAULT_MAX_TIMEOUT_S = 60.0


def check_intent(taskspec: TaskSpec) -> GateResult:
    if taskspec.intent.token == "UNKNOWN":
        return GateResult(False, "intent", "UNKNOWN token — route to fallback channel.")
    return GateResult(True, "intent")


def check_skill(
    taskspec: TaskSpec,
    known_skill_types: Mapping[str, object],
) -> GateResult:
    for st in taskspec.subtasks:
        if st.type not in known_skill_types:
            return GateResult(
                False, "skill", f"Unknown skill type '{st.type}' for subtask '{st.name}'."
            )
    return GateResult(True, "skill")


def check_params(
    taskspec: TaskSpec,
    max_force_n: float = DEFAULT_MAX_FORCE_N,
    max_timeout_s: float = DEFAULT_MAX_TIMEOUT_S,
) -> GateResult:
    c = taskspec.constraints
    if c.max_force_n <= 0 or c.max_force_n > max_force_n:
        return GateResult(False, "params", f"max_force_n {c.max_force_n} out of bounds.")
    if c.timeout_s <= 0 or c.timeout_s > max_timeout_s:
        return GateResult(False, "params", f"timeout_s {c.timeout_s} out of bounds.")
    return GateResult(True, "params")


def check_safety(
    taskspec: TaskSpec,
    ita_min: float = DEFAULT_ITA_MIN,
    drift_max: float = DEFAULT_DRIFT_MAX,
) -> GateResult:
    intent = taskspec.intent
    if intent.confidence < ita_min:
        return GateResult(
            False, "safety", f"ITA {intent.confidence:.2f} < {ita_min} — require confirmation."
        )
    if intent.drift_score > drift_max:
        return GateResult(
            False, "safety", f"drift {intent.drift_score:.2f} > {drift_max} — re-calibrate."
        )
    return GateResult(True, "safety")


def run_gates(
    taskspec: TaskSpec,
    known_skill_types: Mapping[str, object],
    *,
    ita_min: float = DEFAULT_ITA_MIN,
    drift_max: float = DEFAULT_DRIFT_MAX,
) -> list[GateResult]:
    """Run gates 2-5 (gate 1 is already satisfied if TaskSpec parsed).

    Returns the full list so a UI can highlight *which* gate failed,
    not just pass/fail. Callers that want fail-fast should iterate and
    break on the first ``ok=False``.
    """
    results = [
        check_intent(taskspec),
        check_skill(taskspec, known_skill_types),
        check_params(taskspec),
        check_safety(taskspec, ita_min=ita_min, drift_max=drift_max),
    ]
    return results
