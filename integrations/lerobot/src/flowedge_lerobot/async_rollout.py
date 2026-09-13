"""Periodic action delivery with one inference worker and a bounded chunk queue.

This Python evaluation runner measures soft deadlines. It is not covered by the
native allocation-free contract. The application must supply an explicit fallback.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns, sleep

import numpy as np

from .rollout import MAX_ROLLOUT_STEPS


@dataclass(frozen=True)
class AsyncRolloutResult:
    steps: int
    stopped: bool
    startup_ms: float
    elapsed_ms: float
    missed_deadlines: int
    queue_underruns: int
    rejected_plans: int
    dropped_chunks: int
    inference_calls: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    jitter_p99_ms: float
    observation_age_p99_ms: float | None
    plan_ids: tuple


def run_async_rollout(
    robot,
    predict,
    *,
    fallback,
    steps,
    period_ms,
    action_steps,
    observation_steps=1,
    refill_threshold=0.75,
    seed=0,
    fixed_plan=None,
    selector=None,
    clock=perf_counter_ns,
    wait=sleep,
):
    """predict(history, plan, seed) returns a complete executable chunk.

    First inference bootstraps the queue before periodic delivery starts. Later
    chunks are aligned to their observation tick; already elapsed actions are
    discarded. History is copied before dispatch and never shared with the worker.
    """
    if (
        not 0 < steps <= MAX_ROLLOUT_STEPS
        or action_steps <= 0
        or observation_steps <= 0
    ):
        raise ValueError("invalid rollout or horizon bounds")
    if not isfinite(period_ms) or period_ms <= 0 or not 0 < refill_threshold <= 1:
        raise ValueError("invalid control period or refill threshold")
    period_ns = max(1, int(period_ms * 1e6))
    fixed_plan = fixed_plan or {"id": 0, "steps": 10, "solver": "ddim"}
    history = deque(maxlen=observation_steps)
    actions = deque(maxlen=action_steps)
    latencies, delivery_times, ages, plan_ids = [], [], [], []
    misses = underruns = rejected = dropped = calls = 0
    pending = None
    pending_plan = fixed_plan
    pending_tick = 0
    pending_observation_ns = 0
    completed = 0
    boot = clock()
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="flowedge-inference"
    )

    def infer(frames, plan, request_seed, dispatched_ns=None):
        started = clock() if dispatched_ns is None else dispatched_ns
        chunk = np.asarray(predict(frames, plan, request_seed), dtype=np.float32).copy()
        if (
            chunk.ndim != 2
            or chunk.shape[0] != action_steps
            or not np.isfinite(chunk).all()
        ):
            raise ValueError(
                "predict must return a finite, complete executable action chunk"
            )
        return chunk, max(1, clock() - started)

    try:
        robot.reset()
        first = deepcopy(robot.observe())
        first_observation_ns = clock()
        history.extend([first] * observation_steps)
        # Startup has no running action queue. Use the caller's declared fixed plan.
        first_chunk, duration = infer(tuple(history), fixed_plan, seed)
        shape = first_chunk.shape[1:]
        actions.extend((a.copy(), first_observation_ns) for a in first_chunk)
        calls += 1
        latencies.append(duration / 1e6)
        plan_ids.append(fixed_plan["id"])
        if selector is not None:
            selector.observe(fixed_plan, duration)
        start = clock()
        startup_ms = (start - boot) / 1e6
        for tick in range(steps):
            target = start + tick * period_ns
            remaining = target - clock()
            if remaining > 0:
                wait(remaining / 1e9)
            observation_ns = clock()
            observation = deepcopy(robot.observe())
            history.append(observation)
            if pending is not None and pending.done():
                chunk, duration = pending.result()
                if chunk.shape[1:] != shape:
                    raise ValueError("action dimension changed during rollout")
                latencies.append(duration / 1e6)
                if selector is not None:
                    selector.observe(pending_plan, duration)
                elapsed_actions = tick - pending_tick
                if elapsed_actions < action_steps:
                    actions.clear()
                    actions.extend(
                        (a.copy(), pending_observation_ns)
                        for a in chunk[elapsed_actions:]
                    )
                else:
                    dropped += 1
                pending = None
            if actions:
                action, source_ns = actions.popleft()
                ages.append((clock() - source_ns) / 1e6)
            else:
                underruns += 1
                action = np.asarray(fallback(observation), dtype=np.float32)
                if action.shape != shape or not np.isfinite(action).all():
                    raise ValueError(
                        "fallback must return a finite action of the configured shape"
                    )
            robot.send_action(action)
            delivery_times.append(clock())
            completed += 1
            if clock() > target + period_ns:
                misses += 1
            if getattr(robot, "done", False):
                break
            if pending is None and len(actions) <= int(action_steps * refill_threshold):
                plan = (
                    fixed_plan
                    if selector is None
                    else selector.select(
                        buffered_actions=len(actions),
                        period_ns=period_ns,
                        slack_ns=max(1, len(actions)) * period_ns,
                    )
                )
                if plan is None:
                    rejected += 1
                else:
                    pending_plan, pending_tick = plan, tick
                    pending_observation_ns = observation_ns
                    pending = executor.submit(
                        infer, deepcopy(tuple(history)), plan, seed + calls, clock()
                    )
                    calls += 1
                    plan_ids.append(plan["id"])
        elapsed_ms = (clock() - start) / 1e6
    finally:
        try:
            robot.stop()
        finally:
            # Stop delivery before waiting for an in-flight bounded inference call.
            executor.shutdown(wait=True, cancel_futures=True)
    if pending is not None and not pending.cancelled():
        # Do not hide errors merely because the episode ended during inference.
        pending.result()
    p50, p95, p99 = np.percentile(latencies, (50, 95, 99))
    jitter = np.abs(np.diff(delivery_times) - period_ns) / 1e6
    return AsyncRolloutResult(
        completed,
        True,
        startup_ms,
        elapsed_ms,
        misses,
        underruns,
        rejected,
        dropped,
        calls,
        float(p50),
        float(p95),
        float(p99),
        float(np.percentile(jitter, 99)) if len(jitter) else 0.0,
        float(np.percentile(ages, 99)) if ages else None,
        tuple(plan_ids),
    )
