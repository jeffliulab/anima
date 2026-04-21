"""L2 — Planner.

TaskSpec → py_trees BehaviorTree. Each ``Subtask.type`` resolves to a skill
class registered in L3. The planner is deliberately dumb: it does not
reason about the plan, just assembles the tree in order. Reasoning lives
in L1 (LLM); validation lives in the Test-and-Check gate.

Design invariant #7: behavior-tree runtime, no ad-hoc state machines.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Mapping

import py_trees

from .l3_skill import MockSkillBehaviour, SkillBehaviour
from .taskspec import TaskSpec


SkillRegistry = Mapping[str, type[SkillBehaviour]]


def build_tree(
    taskspec: TaskSpec,
    skill_registry: SkillRegistry | None = None,
    on_status_change: Callable[[str, str], Awaitable[None]] | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> py_trees.trees.BehaviourTree:
    """Assemble a py_trees Sequence from a validated TaskSpec.

    When ``skill_registry`` is None (or a subtask type is missing), falls
    back to ``MockSkillBehaviour`` so the tree can still be ticked end-to-
    end without a simulator/robot in the loop.
    """
    root = py_trees.composites.Sequence(
        name=taskspec.intent.token.lower(), memory=True
    )
    registry: SkillRegistry = skill_registry or {}
    ctx: dict = {}  # shared blackboard across the behaviour tree

    for st in taskspec.subtasks:
        skill_cls = registry.get(st.type)
        if skill_cls is not None:
            root.add_child(
                skill_cls(
                    name=st.name,
                    ctx=ctx,
                    on_status_change=on_status_change,
                    loop=loop,
                )
            )
        else:
            root.add_child(
                MockSkillBehaviour(
                    name=st.name,
                    ticks_to_success=3,
                    on_status_change=on_status_change,
                    loop=loop,
                )
            )
    return py_trees.trees.BehaviourTree(root)


async def run_tree(
    tree: py_trees.trees.BehaviourTree,
    tick_interval_s: float = 0.4,
    max_ticks: int = 3000,
) -> py_trees.common.Status:
    """Drive a tree by ticking with asyncio.sleep between ticks.

    Returns SUCCESS / FAILURE when the root terminates, or the last
    status if ``max_ticks`` is exhausted (caller must treat that as
    a timeout).
    """
    for _ in range(max_ticks):
        tree.tick()
        status = tree.root.status
        if status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            return status
        await asyncio.sleep(tick_interval_s)
    return tree.root.status
