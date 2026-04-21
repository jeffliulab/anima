"""ANIMA — Cognitive framework for intent-to-action embodied AI.

Six-layer stack: L0 Signal → L1 Parser → L2 Planner → L3 Skill → L4 Adapter → L5 Assessment.
Framework-level: domain-agnostic by design. Signals can be text, voice transcription,
BCI intent tokens, or any other upstream source; devices can be manipulators, mobile
bases, wheelchairs, or future humanoids.

Public API lives in :mod:`anima.taskspec`. Layer modules (``l0_signal`` … ``l5_assessment``)
are importable directly; see ``docs/00-overview.md`` for the reading map and design
invariants.
"""

__version__ = "0.1.0"

from .taskspec import (
    Alternative,
    Constraints,
    FiveFactors,
    IntentToken,
    PEARecord,
    Subtask,
    TaskSpec,
)

__all__ = [
    "Alternative",
    "Constraints",
    "FiveFactors",
    "IntentToken",
    "PEARecord",
    "Subtask",
    "TaskSpec",
    "__version__",
]
