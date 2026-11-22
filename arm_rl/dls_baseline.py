"""经典参照: 雅可比阻尼最小二乘(DLS) + 零空间避障梯度。仅作参照, 不作优越性主张。"""
from __future__ import annotations

import numpy as np

from env3link import (ACTION_SCALE, MAX_STEPS, fk, jacobian, seg_point_dist)


def min_clearance(q, obstacles, ob_r):
    if not obstacles:
        return float("inf")
    pts = fk(q)
    return min(seg_point_dist(pts[i], pts[i + 1], c) - ob_r
               for i in range(3) for c in obstacles)


def dls_action(env, margin=0.18, gain=1.0, avoid_gain=1.5):
    q = env.q
    goal = env.task["goal"]
    pts = fk(q)
    ee = pts[-1]
    J = jacobian(q)
    lam = 0.05
    JJt = J @ J.T + (lam ** 2) * np.eye(2)
    Jpin = J.T @ np.linalg.inv(JJt)
    dq_task = Jpin @ (gain * (goal - ee))
    N = np.eye(3) - Jpin @ J
    dq = dq_task
    clear = min_clearance(q, env.task["obstacles"], env.ob_r)
    if np.isfinite(clear) and clear < margin:
        h = 1e-4
        g = np.zeros(3)
        for j in range(3):
            qp = q.copy(); qp[j] += h
            qm = q.copy(); qm[j] -= h
            g[j] = (min_clearance(qp, env.task["obstacles"], env.ob_r)
                    - min_clearance(qm, env.task["obstacles"], env.ob_r)) / (2 * h)
        gn = g / (np.linalg.norm(g) + 1e-8)
        dq = dq + N @ (avoid_gain * (margin - clear) / margin * gn)
    dq = np.clip(dq, -ACTION_SCALE, ACTION_SCALE)
    return dq / ACTION_SCALE


def dls_solve(env, task, max_steps=MAX_STEPS):
    env.reset(task=task)
    done = False
    info = {"success": False, "steps": 0, "clearance": float("inf"),
            "path_length": 0.0, "collision": False}
    for _ in range(max_steps):
        a = dls_action(env)
        _, _, term, trunc, info = env.step(a)
        done = term or trunc
        if done:
            break
    return info
