"""L3 — Skill Registry & Executor.

Design invariant #8: Function-Calling + Affordance Scoring (not RAG) when
the skill set is < 100 entries.

This module ships two kinds of skill node:

  * ``MockSkillBehaviour`` — ticks RUNNING → SUCCESS after N ticks, useful
    for offline tests / UI development without a simulator or real device.
  * ``SkillBehaviour``     — abstract base for real actuation. Subclass
    and implement ``_start_action`` / ``_tick``; the executor handles the
    timeout, blackboard, and status dispatch.

A real deployment registers its skill classes with L2 at ``build_tree``
time. Skills cooperate through the shared ``ctx`` blackboard so
``locate → navigate → grasp → lift → deliver`` can pass targets forward.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Protocol

import py_trees
from py_trees.common import Status

logger = logging.getLogger(__name__)


class StatusCallback(Protocol):
    def __call__(self, node_name: str, status: str) -> Awaitable[None]: ...


# -----------------------------------------------------------------------------
# Mock skill — used whenever no real actuator is wired up.
# -----------------------------------------------------------------------------


class MockSkillBehaviour(py_trees.behaviour.Behaviour):
    """Fake a skill that completes successfully after ``ticks_to_success`` ticks."""

    def __init__(
        self,
        name: str,
        ticks_to_success: int = 3,
        on_status_change: Callable[[str, str], Awaitable[None]] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        ctx: dict | None = None,
    ):
        super().__init__(name=name)
        self._ticks_to_success = ticks_to_success
        self._ticks_done = 0
        self._on_status_change = on_status_change
        self._loop = loop
        self._ctx = ctx or {}

    def initialise(self) -> None:
        self._ticks_done = 0
        self._dispatch("running")

    def update(self) -> Status:
        self._ticks_done += 1
        if self._ticks_done >= self._ticks_to_success:
            self._dispatch("success")
            return Status.SUCCESS
        return Status.RUNNING

    def _dispatch(self, status: str) -> None:
        if not self._on_status_change or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._on_status_change(self.name, status), self._loop
        )


# -----------------------------------------------------------------------------
# Base class for real skills — drive an actuator via start_action + tick.
# -----------------------------------------------------------------------------


class SkillBehaviour(py_trees.behaviour.Behaviour):
    """Base for async skills. Non-blocking; polled each tick.

    Subclass and override ``_start_action`` (called once on initialise) and
    ``_tick`` (called every tick; return True when the skill is done).
    Timeout is enforced by the base; status events are dispatched on the
    asyncio loop passed by L2.
    """

    TIMEOUT_S = 15.0

    def __init__(
        self,
        name: str,
        ctx: dict,
        on_status_change: Callable[[str, str], Awaitable[None]] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(name=name)
        self._ctx = ctx
        self._on_status_change = on_status_change
        self._loop = loop
        self._t_start = 0.0
        self._phase = "init"

    def initialise(self) -> None:
        self._t_start = time.time()
        self._phase = "init"
        self._dispatch("running")
        try:
            self._start_action()
        except Exception:
            logger.exception("skill %s start failed", self.name)

    def update(self) -> Status:
        try:
            done = self._tick()
        except Exception:
            logger.exception("skill %s tick failed", self.name)
            self._dispatch("fail")
            return Status.FAILURE
        if done:
            self._dispatch("success")
            return Status.SUCCESS
        if time.time() - self._t_start > self.TIMEOUT_S:
            logger.warning("skill %s timed out", self.name)
            self._dispatch("fail")
            return Status.FAILURE
        return Status.RUNNING

    def _start_action(self) -> None:  # override
        pass

    def _tick(self) -> bool:  # override, return True when done
        return True

    def _dispatch(self, status: str) -> None:
        if not self._on_status_change or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._on_status_change(self.name, status), self._loop
        )
