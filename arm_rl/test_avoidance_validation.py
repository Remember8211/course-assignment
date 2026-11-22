"""Behavioral tests for the independent, read-only model validation pipeline.

Run from the project root with:
    python -B -m unittest discover -s arm_rl -p test_avoidance_validation.py -v

The grazing-contact fixture is deliberately outside the sampled task domain:
its purpose is to expose an endpoint-only collision checker with one legal
0.05 rad action.  No existing environment or model is modified by these tests.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from env3link import ArmEnv3Link, Q0, fk, seg_point_dist
from validate_avoidance import (
    AuditEnv,
    inspect_motion,
    motion_samples,
    rollout,
    surface_clearances,
)


def task(q0=Q0, goal=(0.65, 0.40), obstacles=()):
    """Supply both external metadata and the old environment's three fields."""
    return {
        "id": "test",
        "task_id": "test",
        "group": 1,
        "scenario": "A" if not len(obstacles) else "C",
        "variant": "A" if not len(obstacles) else "C",
        "q0": np.asarray(q0, dtype=float).copy(),
        "goal": np.asarray(goal, dtype=float).copy(),
        "obstacles": [np.asarray(c, dtype=float).copy() for c in obstacles],
        "obstacle_radius": 0.08,
        "radius": 0.08,
    }


class GeometryChecks(unittest.TestCase):
    def test_batch_clearance_matches_independent_scalar_geometry(self):
        rng = np.random.default_rng(603)
        qs = rng.uniform(-np.pi, np.pi, size=(40, 3))
        obstacles = np.array([[0.31, 0.24], [-0.55, 0.10]])
        actual = surface_clearances(qs, obstacles)
        expected = []
        for q in qs:
            pts = fk(q)
            expected.append(min(
                seg_point_dist(pts[i], pts[i + 1], c) - 0.08 - 0.02
                for i in range(3) for c in obstacles
            ))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_motion_subdivision_preserves_endpoints_and_resolution(self):
        q0 = np.array([0.1, -0.4, 0.7])
        q1 = q0 + [0.05, -0.0341, 0.0017]
        qs = motion_samples(q0, q1, 0.001)
        np.testing.assert_allclose(qs[0], q0, atol=0, rtol=0)
        np.testing.assert_allclose(qs[-1], q1, atol=1e-15, rtol=0)
        self.assertLessEqual(float(np.max(np.abs(np.diff(qs, axis=0)))), 0.001 + 1e-14)

    def test_safe_endpoints_do_not_hide_midmotion_collision(self):
        q0 = np.array([-0.025, 0.0, 0.0])
        q1 = np.array([0.025, 0.0, 0.0])
        obstacle = [[1.0999, 0.0]]
        end_gaps = surface_clearances(np.array([q0, q1]), obstacle)
        self.assertTrue(np.all(end_gaps > 0))
        self.assertLess(float(surface_clearances(np.zeros((1, 3)), obstacle)[0]), 0)
        result = inspect_motion(q0, q1, obstacle)
        self.assertTrue(result["collision"])
        self.assertLessEqual(result["min_clearance"], 0)
        self.assertLess(float(result["q_end"][0]), q1[0])
        self.assertLessEqual(float(np.max(np.abs(np.diff(result["samples"], axis=0)))), 0.001 + 1e-14)
        finer = inspect_motion(q0, q1, obstacle, resolution=0.0005)
        self.assertTrue(finer["collision"])

    def test_no_obstacle_motion_is_unrestricted(self):
        result = inspect_motion(Q0, Q0 + 0.02, [])
        self.assertFalse(result["collision"])
        self.assertTrue(np.isinf(result["min_clearance"]))
        np.testing.assert_allclose(result["q_end"], Q0 + 0.02)


class EnvironmentChecks(unittest.TestCase):
    def test_collision_takes_precedence_inside_goal_region(self):
        env = AuditEnv(tolerance=0.02)
        env.reset(task=task(
            q0=[-0.025, 0, 0], goal=[1, 0], obstacles=[[1.0999, 0]],
        ))
        self.assertGreater(np.linalg.norm(fk(env.q)[-1] - [1, 0]), 0.02)
        _, status, detail = env.advance([1, 0, 0])
        self.assertEqual(status, "collision")
        self.assertLess(detail["dist"], 0.02)
        self.assertLessEqual(detail["min_clearance"], 0)

    def test_action_and_joint_clipping(self):
        q0 = np.array([np.pi - 0.01, -np.pi + 0.01, 0.0])
        env = AuditEnv(tolerance=0.02)
        env.reset(task=task(q0=q0, goal=[0.0, 0.7]))
        env.advance([10, -10, 10])
        np.testing.assert_allclose(env.q, [np.pi, -np.pi, 0.05], atol=1e-14)
        self.assertTrue(np.all(np.abs(env.q - q0) <= 0.05 + 1e-14))

    def test_tolerance_changes_terminal_classification(self):
        t = task(goal=fk(Q0)[-1] + [0.05, 0.0])
        loose = AuditEnv(tolerance=0.08)
        strict = AuditEnv(tolerance=0.02)
        loose.reset(task=t)
        strict.reset(task=t)
        _, loose_status, loose_info = loose.advance([0, 0, 0])
        _, strict_status, strict_info = strict.advance([0, 0, 0])
        self.assertEqual(loose_status, "success")
        self.assertIsNone(strict_status)
        self.assertAlmostEqual(loose_info["dist"], strict_info["dist"], places=14)

    def test_timeout_is_exclusive_and_occurs_at_120_actions(self):
        env = AuditEnv(tolerance=0.02)
        env.reset(task=task(goal=[0.1, 0.9]))
        for _ in range(119):
            _, status, _ = env.advance([0, 0, 0])
            self.assertIsNone(status)
        _, status, _ = env.advance([0, 0, 0])
        self.assertEqual(status, "timeout")

    def test_original_13_dim_observation_is_preserved(self):
        t = task(obstacles=[[-0.6, -0.3], [-0.6, 0.3]])
        old = ArmEnv3Link(n_obstacles=2, obstacle_radius=0.08)
        new = AuditEnv(tolerance=0.02)
        o_old, _ = old.reset(task=t)
        o_new, _ = new.reset(task=t)
        self.assertEqual(o_new.shape, (13,))
        np.testing.assert_array_equal(o_old, o_new)
        for action in ([0.1, -0.2, 0.3], [-0.3, 0.1, 0.2]):
            o_old, _, _, _, _ = old.step(action)
            o_new, status, _ = new.advance(action)
            self.assertIsNone(status)
            np.testing.assert_array_equal(o_old, o_new)

    def test_known_reference_path_can_be_replayed(self):
        q_goal = Q0 + np.array([0.10, -0.05, 0.05])
        t = task(goal=fk(q_goal)[-1], obstacles=[[-0.6, -0.3], [-0.6, 0.3]])
        # Two legal actions, exact endpoint: this tests execution, not planning.
        reference = motion_samples(Q0, q_goal, resolution=0.05)
        self.assertLessEqual(len(reference) - 1, 120)
        self.assertGreater(float(np.min(surface_clearances(reference, t["obstacles"]))), 0.01)
        env = AuditEnv(tolerance=0.02)
        env.reset(task=t)
        status = None
        for next_q in reference[1:]:
            _, status, _ = env.advance((next_q - env.q) / 0.05)
            if status is not None:
                break
        self.assertEqual(status, "success")
        self.assertLess(np.linalg.norm(fk(env.q)[-1] - t["goal"]), 0.02)


class RolloutChecks(unittest.TestCase):
    def test_deterministic_controller_repeats_identically(self):
        t = task(obstacles=[[-0.6, -0.3], [-0.6, 0.3]])
        first = rollout(t, "dls_noavoid", tolerance=0.02)
        second = rollout(t, "dls_noavoid", tolerance=0.02)
        # The result is a reproducible data artifact, with no wall-clock field.
        self.assertEqual(first, second)

    def test_no_obstacle_result_uses_json_null_not_infinity(self):
        result = rollout(task(), "dls_noavoid", tolerance=0.08)
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("Infinity", serialized)
        self.assertNotIn("NaN", serialized)
        self.assertIn("null", serialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
