"""平面三连杆机械臂运动学环境(纯 numpy, 无仿真器/ROS)。

任务: 末端从固定起始位形出发到达随机目标点, 途中绕开随机圆形障碍。
状态/动作/奖励严格按实施计划规格。
"""
from __future__ import annotations

import numpy as np

LINK_LEN = np.array([0.40, 0.30, 0.30], dtype=np.float64)
LINK_RADIUS = 0.02
JOINT_LIMIT = float(np.pi)
Q0 = np.array([0.2, 0.5, -0.6], dtype=np.float64)
GOAL_TOL = 0.08
MAX_STEPS = 120
ACTION_SCALE = 0.05
REACH_MAX = float(LINK_LEN.sum())


def fk(q: np.ndarray) -> np.ndarray:
    """正运动学: 返回 (4,2) 关节点坐标, [0] 为基座, [-1] 为末端。"""
    q = np.asarray(q, dtype=np.float64)
    theta = np.cumsum(q)
    pts = [np.zeros(2)]
    for i, l in enumerate(LINK_LEN):
        pts.append(pts[-1] + l * np.array([np.cos(theta[i]), np.sin(theta[i])]))
    return np.array(pts)


def jacobian(q: np.ndarray) -> np.ndarray:
    """末端雅可比 (2,3)。"""
    q = np.asarray(q, dtype=np.float64)
    theta = np.cumsum(q)
    J = np.zeros((2, 3))
    for j in range(3):
        J[0, j] = -np.sum(LINK_LEN[j:] * np.sin(theta[j:]))
        J[1, j] = np.sum(LINK_LEN[j:] * np.cos(theta[j:]))
    return J


def seg_point_dist(p1: np.ndarray, p2: np.ndarray, c: np.ndarray) -> float:
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float); c = np.asarray(c, float)
    v = p2 - p1
    L2 = float(v @ v)
    t = 0.0 if L2 < 1e-12 else float(np.clip(((c - p1) @ v) / L2, 0.0, 1.0))
    return float(np.linalg.norm(c - (p1 + t * v)))


def sample_task(rng: np.random.Generator, n_obstacles: int, ob_r: float,
                goal_range=(0.55, 0.95), q0: np.ndarray | None = None):
    """采样一个任务: 障碍中心列表 + 目标 + 起始位形, 保证起始无碰撞且目标可达。"""
    q0 = Q0 if q0 is None else np.asarray(q0, float)
    ee0 = fk(q0)[-1]
    ee0_ang = float(np.arctan2(ee0[1], ee0[0]))
    for _ in range(200):
        centers = []
        tries = 0
        while len(centers) < n_obstacles and tries < 3000:
            tries += 1
            rad = rng.uniform(0.2, 0.8)
            ang = rng.uniform(0.0, 2 * np.pi)
            c = np.array([rad * np.cos(ang), rad * np.sin(ang)])
            if np.linalg.norm(c) < 0.15 + ob_r:
                continue
            # 障碍离起点末端远一点, 避免直接堵住出发方向
            if np.linalg.norm(c - ee0) < 0.30:
                continue
            if any(np.linalg.norm(c - o) < 0.25 for o in centers):
                continue
            centers.append(c)
        if len(centers) < n_obstacles:
            continue
        pts = fk(q0)
        clear0 = (
            min(seg_point_dist(pts[i], pts[i + 1], c) - ob_r
                for i in range(3) for c in centers)
            if centers else float("inf")
        )
        if clear0 < 0.05:
            continue
        for _ in range(3000):
            ga = ee0_ang + rng.uniform(-0.9, 0.9)
            gr = rng.uniform(*goal_range)
            g = np.array([gr * np.cos(ga), gr * np.sin(ga)])
            if gr > REACH_MAX - 0.05:
                continue
            if all(np.linalg.norm(g - c) >= 0.18 + ob_r for c in centers):
                return {"obstacles": centers, "goal": g, "q0": q0.copy()}
    # 兜底: 无障碍
    return {"obstacles": [], "goal": np.array([0.6, 0.3]), "q0": q0.copy()}


def make_tasks(n: int, seed: int, n_obstacles: int = 2, ob_r: float = 0.08,
               goal_range=(0.55, 0.95), perturb_q0: float = 0.0):
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n):
        q0 = Q0 if perturb_q0 <= 0 else np.clip(
            Q0 + rng.normal(0.0, perturb_q0, 3), -JOINT_LIMIT, JOINT_LIMIT)
        tasks.append(sample_task(rng, n_obstacles, ob_r, goal_range, q0))
    return tasks


class ArmEnv3Link:
    """obs(13) = q(3)+ee(2)+goal(2)+(goal-ee)(2)+每障碍(ee到心距, 段到面最小距)(2*2)。"""

    def __init__(self, n_obstacles: int = 2, obstacle_radius: float = 0.08,
                 include_obstacle_obs: bool = True, sparse_reward: bool = False,
                 seed: int | None = None) -> None:
        self.n_obstacles = int(n_obstacles)
        self.ob_r = float(obstacle_radius)
        self.include_obstacle_obs = bool(include_obstacle_obs)
        self.sparse_reward = bool(sparse_reward)
        self.obs_dim = 13
        self.act_dim = 3
        self._rng = np.random.default_rng(seed)
        self.task = None
        self.q = Q0.copy()
        self.steps = 0
        self.ep_min_clear = float("inf")
        self.ep_path = 0.0
        self._prev_dist = None

    # ---------------- 几何 ----------------
    def _clearances(self, q=None):
        pts = fk(self.q if q is None else q)
        if not self.task["obstacles"]:
            return float("inf"), pts
        d = min(
            seg_point_dist(pts[i], pts[i + 1], c) - self.ob_r
            for i in range(3) for c in self.task["obstacles"]
        )
        return d, pts

    def _obs(self, pts: np.ndarray) -> np.ndarray:
        ee = pts[-1]
        g = self.task["goal"]
        parts = [self.q, ee, g, g - ee]
        obs_list = self.task["obstacles"]
        if obs_list:
            d_ee = [float(np.linalg.norm(ee - c)) for c in obs_list]
            clears = [
                min(seg_point_dist(pts[i], pts[i + 1], c) - self.ob_r for i in range(3))
                for c in obs_list
            ]
            i1 = int(np.argmin(d_ee))
            i2 = int(np.argmin(clears))
            parts.append(np.array([min(d_ee[i1], 2.0), clears[i1]]))
            parts.append(np.array([min(d_ee[i2], 2.0), clears[i2]]))
        else:
            parts.append(np.zeros(2))
            parts.append(np.zeros(2))
        vec = np.concatenate(parts).astype(np.float32)
        if not self.include_obstacle_obs:
            vec[9:13] = 0.0
        return vec

    # ---------------- gym 风格接口 ----------------
    def reset(self, seed: int | None = None, task: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.task = task if task is not None else sample_task(
            self._rng, self.n_obstacles, self.ob_r)
        self.q = self.task["q0"].copy()
        self.steps = 0
        self.ep_min_clear = float("inf")
        self.ep_path = 0.0
        pts = fk(self.q)
        self._prev_dist = float(np.linalg.norm(pts[-1] - self.task["goal"]))
        self._prev_pts = pts
        return self._obs(pts), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0) * ACTION_SCALE
        q_new = np.clip(self.q + a, -JOINT_LIMIT, JOINT_LIMIT)
        pts_new = fk(q_new)
        ee_new = pts_new[-1]
        dist_new = float(np.linalg.norm(ee_new - self.task["goal"]))
        self.ep_path += float(np.linalg.norm(ee_new - self._prev_pts[-1]))
        self.q = q_new
        self.steps += 1

        clear, _ = self._clearances()
        self.ep_min_clear = min(self.ep_min_clear, clear)

        if self.sparse_reward:
            reward = 0.0
        else:
            # 密集距离惩罚 + 进度塑形 + 控制量惩罚 + 时间惩罚(鼓励尽快到达)
            reward = (-dist_new
                      + (self._prev_dist - dist_new)
                      - 0.01 * float(np.linalg.norm(a))
                      - 0.01)
        self._prev_dist = dist_new
        self._prev_pts = pts_new

        terminated = False
        if clear < LINK_RADIUS:
            reward += -5.0
            terminated = True
        elif dist_new < GOAL_TOL:
            reward += 10.0
            terminated = True
        truncated = (not terminated) and self.steps >= MAX_STEPS
        info = {
            "success": bool(dist_new < GOAL_TOL),
            "collision": bool(clear < LINK_RADIUS),
            "dist": dist_new,
            "clearance": self.ep_min_clear,
            "path_length": self.ep_path,
            "steps": self.steps,
        }
        return self._obs(pts_new), float(reward), terminated, truncated, info


