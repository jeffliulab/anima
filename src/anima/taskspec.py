"""TaskSpec — the canonical data contract between the six ANIMA layers.

Every layer boundary in the framework speaks TaskSpec (or a strict subset).

Design notes:

  * ``intent_confidence``     — ITA input (L5)
  * ``requires_confirmation`` — drives a user/caregiver gating channel
  * ``drift_score``           — MQA input (L5); models input-signal uncertainty
                                (BCI manifold drift, speech transcription
                                confidence, vision detection confidence, …)
  * ``estop_channels`` (implicit) — enforced by L4 Embodied Adapter

The ``IntentTokenName`` vocabulary below is an *example* 35-token set drawn
from the first reference application (medical-care ADL). Downstream
applications extend or replace it (chess-specific intents, household-task
intents, etc.). The framework contract is the **structure** of TaskSpec,
not any particular vocabulary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

UTC = timezone.utc


IntentTokenName = Literal[
    # ADL (Activities of Daily Living) — 15 tokens
    "DRINK_WATER",
    "EAT_FOOD",
    "GRASP",
    "LIFT",
    "DELIVER",
    "PLACE",
    "RELEASE",
    "WIPE_MOUTH",
    "SCRATCH_ITCH",
    "ADJUST_PILLOW",
    "ADJUST_BLANKET",
    "HAND_OVER",
    "OPEN_BOTTLE",
    "POUR",
    "STIR",
    # Navigation — 10 tokens
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "GOTO_BED",
    "GOTO_TABLE",
    "GOTO_DOOR",
    "GOTO_BATHROOM",
    "FOLLOW_CAREGIVER",
    "STOP_MOVING",
    # Device control — 5 tokens
    "TURN_ON_LIGHT",
    "TURN_OFF_LIGHT",
    "ADJUST_TV",
    "CALL_ELEVATOR",
    "OPEN_CURTAIN",
    # Emergency — 2 tokens (non-signal-path E-stop is a separate channel)
    "CALL_HELP",
    "EMERGENCY_STOP",
    # Meta — 3 tokens
    "CONFIRM",
    "CANCEL",
    "UNKNOWN",
]

SubtaskType = Literal["locate", "grasp", "lift", "deliver", "navigate", "release"]


class Alternative(BaseModel):
    token: IntentTokenName
    confidence: float = Field(ge=0.0, le=1.0)


class IntentToken(BaseModel):
    """L0 → L1 boundary payload.

    ``drift_score`` is an upstream-signal uncertainty estimate. For BCI
    sources this is neural-manifold drift; for speech it could be an ASR
    confidence; for vision, detection confidence. The downstream contract
    is the same across sources.
    """

    token: IntentTokenName
    confidence: float = Field(ge=0.0, le=1.0)
    requires_confirmation: bool = False
    alternatives: list[Alternative] = Field(default_factory=list)
    drift_score: float = Field(default=0.05, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_text: str = ""


class Subtask(BaseModel):
    name: str
    type: SubtaskType


class Constraints(BaseModel):
    max_force_n: float = 8.0
    timeout_s: float = 15.0


class TaskSpec(BaseModel):
    """L1 output → L2 input. Immutable once emitted."""

    intent: IntentToken
    subtasks: list[Subtask]
    constraints: Constraints = Field(default_factory=Constraints)
    device: str = "mock_device"


class FiveFactors(BaseModel):
    """L5 event-triggered self-assessment output.

    Per design invariant #5, ``goa`` is computed multiplicatively:
        goa = ita * p_plan * sqa
    """

    ita: float = Field(default=0.0, ge=0.0, le=1.0)
    mqa: float = Field(default=1.0, ge=0.0, le=1.0)
    sqa: float = Field(default=1.0, ge=0.0, le=1.0)
    goa: float = Field(default=0.0, ge=0.0, le=1.0)
    pea_count: int = 0


class PEARecord(BaseModel):
    """Post-Execution Assessment entry. Append-only."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    intent_token: IntentTokenName
    outcome: Literal["success", "fail", "cancel"]
    user_text: str = ""
