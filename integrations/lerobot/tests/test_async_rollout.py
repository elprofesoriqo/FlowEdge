import unittest
from unittest.mock import patch
from concurrent.futures import Future

import numpy as np

from flowedge_lerobot.async_rollout import run_async_rollout


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += round(seconds * 1e9)


class InlineExecutor:
    def __init__(self, **kwargs):
        pass

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, **kwargs):
        pass


class Robot:
    def reset(self):
        self.actions = []
        self.stopped = False

    def observe(self):
        return np.array([len(self.actions)], dtype=np.float32)

    def send_action(self, action):
        self.actions.append(action.copy())

    def stop(self):
        self.stopped = True


class AsyncTests(unittest.TestCase):
    def run_case(self, predict, **kwargs):
        clock, robot = Clock(), Robot()
        with patch("flowedge_lerobot.async_rollout.ThreadPoolExecutor", InlineExecutor):
            result = run_async_rollout(
                robot,
                predict,
                fallback=lambda _: np.array([-1.0]),
                steps=8,
                period_ms=10,
                action_steps=4,
                clock=clock,
                wait=clock.wait,
                **kwargs,
            )
        return result, robot

    def test_periodic_delivery_and_chunk_overlap_are_measured(self):
        result, robot = self.run_case(
            lambda history, plan, seed: np.arange(4, dtype=np.float32)[:, None]
        )
        self.assertTrue(robot.stopped)
        self.assertEqual(result.steps, 8)
        self.assertEqual(result.missed_deadlines, 0)
        self.assertEqual(result.jitter_p99_ms, 0)
        self.assertEqual(result.queue_underruns, 0)
        self.assertEqual(result.elapsed_ms, 70)
        self.assertGreater(result.inference_calls, 1)

    def test_no_feasible_plan_uses_explicit_fallback_and_counts_underruns(self):
        class Reject:
            def observe(self, *args):
                pass

            def select(self, **kwargs):
                return None

        result, robot = self.run_case(lambda *args: np.ones((4, 1)), selector=Reject())
        self.assertEqual(result.queue_underruns, 4)
        self.assertGreater(result.rejected_plans, 0)
        self.assertEqual(robot.actions[-1].tolist(), [-1.0])

    def test_inference_failure_stops_robot(self):
        robot = Robot()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            run_async_rollout(
                robot,
                lambda *args: (_ for _ in ()).throw(RuntimeError("failed")),
                fallback=lambda _: np.zeros(1),
                steps=1,
                period_ms=10,
                action_steps=4,
            )
        self.assertTrue(robot.stopped)

    def test_delayed_chunk_is_discarded_after_its_action_horizon(self):
        clock, robot = Clock(), Robot()

        class DelayedFuture:
            def __init__(self, function, args):
                self.value = function(*args)
                self.ready = clock() + 60_000_000

            def done(self):
                return clock() >= self.ready

            def cancelled(self):
                return False

            def result(self):
                return self.value

        class DelayedExecutor(InlineExecutor):
            def submit(self, function, *args):
                return DelayedFuture(function, args)

        with patch(
            "flowedge_lerobot.async_rollout.ThreadPoolExecutor", DelayedExecutor
        ):
            result = run_async_rollout(
                robot,
                lambda *args: np.ones((4, 1)),
                fallback=lambda _: np.array([-1.0]),
                steps=8,
                period_ms=10,
                action_steps=4,
                clock=clock,
                wait=clock.wait,
            )
        self.assertGreater(result.dropped_chunks, 0)
        self.assertEqual(result.queue_underruns, 4)
        self.assertEqual(robot.actions[-1].tolist(), [-1.0])

    def test_delivery_overrun_is_measured_against_absolute_period(self):
        clock, robot = Clock(), Robot()

        def slow_observe():
            clock.now += 15_000_000
            return np.zeros(1)

        robot.observe = slow_observe
        with patch("flowedge_lerobot.async_rollout.ThreadPoolExecutor", InlineExecutor):
            result = run_async_rollout(
                robot,
                lambda *args: np.ones((4, 1)),
                fallback=lambda _: np.zeros(1),
                steps=3,
                period_ms=10,
                action_steps=4,
                clock=clock,
                wait=clock.wait,
            )
        self.assertEqual(result.missed_deadlines, 3)
