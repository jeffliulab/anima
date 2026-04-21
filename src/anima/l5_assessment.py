"""L5 — Five-Factor Self-Assessment (ITA / MQA / SQA / GOA / PEA).

Design invariants this file enforces:

  * #3 Five-factor event-triggered assessment (not continuous logging).
  * #4 Three-stage time evaluation (Pre / Runtime / Post) orthogonal
       to the five factors.
  * #5 GOA composition is multiplicative: ``P = ∏ P_i``. Averaging is
       forbidden because it masks low-probability bottlenecks.
  * #6 PEA retrieval is three-factor:
       ``score = recency * 0.5 + relevance * 3.0 + importance * 2.0``.
       This module stores PEA entries; retrieval weights are re-applied
       by whichever retrieval caller needs them (retrieval itself is
       kept out of this module so tests do not need a vector store).

Factor mapping:

  * ITA = intent confidence from ``IntentToken``
  * MQA = ``1 - drift_score`` (upstream-signal uncertainty; see L0 docstring)
  * SQA = rolling success rate blended with a Beta prior
  * GOA = ITA * P_plan * SQA
  * PEA = append-only JSONL log
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .taskspec import FiveFactors, IntentToken, PEARecord, TaskSpec

UTC = timezone.utc

P_PLAN_MOCK = 0.95
P_SKILL_PRIOR = 0.91
SQA_WINDOW = 10


def _read_last_outcomes(pea_log: Path, limit: int) -> list[str]:
    if not pea_log.exists():
        return []
    lines: list[str] = []
    with pea_log.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    out: list[str] = []
    for line in lines[-limit:]:
        try:
            rec = json.loads(line)
            outcome = rec.get("outcome")
            if outcome in ("success", "fail"):
                out.append(outcome)
        except json.JSONDecodeError:
            continue
    return out


def compute_sqa(
    pea_log: Path,
    window: int = SQA_WINDOW,
    prior_weight: float = 3.0,
) -> float:
    """Rolling success rate blended with a prior.

    With ``prior_weight=3`` and a ``P_SKILL_PRIOR=0.91`` base rate, roughly
    four real observations are needed before empirical rate dominates the
    prior — prevents SQA from swinging wildly on the first few runs.
    """
    outcomes = _read_last_outcomes(pea_log, window)
    n = len(outcomes)
    successes = sum(1 for o in outcomes if o == "success")
    num = successes + P_SKILL_PRIOR * prior_weight
    den = n + prior_weight
    return max(0.0, min(1.0, num / den))


def compute_pre_goa(
    ita: float,
    p_plan: float = P_PLAN_MOCK,
    p_skill: float | None = None,
    pea_log: Path | None = None,
) -> float:
    if p_skill is None:
        if pea_log is None:
            raise ValueError("Pass p_skill directly or provide pea_log for SQA.")
        p_skill = compute_sqa(pea_log)
    return max(0.0, min(1.0, ita * p_plan * p_skill))


def compute_five_factors(taskspec: TaskSpec, pea_log: Path) -> FiveFactors:
    """Event-triggered at the Pre stage: right before L2 hands the tree
    to the executor. Runtime and Post stages call this function again
    at their respective trigger points."""
    intent = taskspec.intent
    ita = intent.confidence
    mqa = max(0.0, 1.0 - intent.drift_score)
    sqa = compute_sqa(pea_log)
    goa = compute_pre_goa(ita, p_skill=sqa)
    return FiveFactors(ita=ita, mqa=mqa, sqa=sqa, goa=goa, pea_count=pea_count(pea_log))


def pea_count(pea_log: Path) -> int:
    if not pea_log.exists():
        return 0
    with pea_log.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def log_pea(pea_log: Path, intent: IntentToken, outcome: str) -> PEARecord:
    """Append-only write. PEA entries are immutable once flushed."""
    pea_log.parent.mkdir(parents=True, exist_ok=True)
    record = PEARecord(
        timestamp=datetime.now(UTC),
        intent_token=intent.token,
        outcome=outcome,  # type: ignore[arg-type]
        user_text=intent.source_text,
    )
    with pea_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return record
