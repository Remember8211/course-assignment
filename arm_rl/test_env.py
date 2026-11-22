"""T1: 环境数学与接口自检。"""
import numpy as np

from env3link import (ArmEnv3Link, fk, jacobian, make_tasks, seg_point_dist,
                      LINK_LEN, Q0)


def test_fk_jacobian():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, 3)
        J = jacobian(q)
        Jn = np.zeros((2, 3))
        h = 1e-6
        for j in range(3):
            qp = q.copy(); qp[j] += h
            qm = q.copy(); qm[j] -= h
            Jn[:, j] = (fk(qp)[-1] - fk(qm)[-1]) / (2 * h)
        assert np.max(np.abs(J - Jn)) < 1e-6, (J, Jn)
    # 全伸展时末端距离 = 链长和
    assert abs(np.linalg.norm(fk(np.zeros(3))[-1]) - LINK_LEN.sum()) < 1e-9
    print("T1 fk/jacobian OK")


def test_seg_dist():
    assert abs(seg_point_dist([0, 0], [1, 0], [0.5, 1.0]) - 1.0) < 1e-9
    assert abs(seg_point_dist([0, 0], [1, 0], [2.0, 0.0]) - 1.0) < 1e-9
    assert abs(seg_point_dist([0, 0], [1, 0], [0.5, 0.0]) - 0.0) < 1e-9
    print("T1 seg_dist OK")


def test_env_interface():
    env = ArmEnv3Link(seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (13,), obs.shape
    for _ in range(300):
        a = np.random.uniform(-1, 1, 3)
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape == (13,)
        assert np.all(np.isfinite(obs))
        if term or trunc:
            obs, _ = env.reset()
    # 无障碍模式 obs 维度不变
    env0 = ArmEnv3Link(n_obstacles=0, seed=2)
    o0, _ = env0.reset(seed=2)
    assert o0.shape == (13,)
    assert np.all(o0[9:13] == 0.0)
    # 任务集可复现
    t1 = make_tasks(5, seed=7)
    t2 = make_tasks(5, seed=7)
    for a, b in zip(t1, t2):
        assert np.allclose(a["goal"], b["goal"])
    print("T1 env interface OK")


if __name__ == "__main__":
    test_fk_jacobian()
    test_seg_dist()
    test_env_interface()
    print("ALL T1 PASS")
