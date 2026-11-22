"""工业 AGV 栅格路径规划环境 (Gymnasium 接口)。

将 AGV 在仓库栅格地图中的导航问题建模为马尔可夫决策过程 (MDP):
    状态 s : AGV 归一化坐标(2) + 目标归一化坐标(2) + 八方向局部障碍占位(8)
    动作 a : 四邻域移动 (可选八邻域)
    奖励 r : 每步 -1, 撞障碍 -10 (原地不动), 到达目标 +100
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 四邻域: 上, 下, 左, 右
DELTA_4 = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
# 八邻域: 追加四个斜向
DELTA_8 = np.array(
    [[-1, 0], [1, 0], [0, -1], [0, 1],
     [-1, -1], [-1, 1], [1, -1], [1, 1]],
    dtype=np.int64,
)
# 局部障碍探测方向 (始终用八方向, 便于跨地图泛化)
PROBE = DELTA_8


class GridNavEnv(gym.Env):
    """栅格地图上的 AGV 最短路径导航环境。"""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 10,
        obstacle_ratio: float = 0.15,
        seed: int = 0,
        grid: np.ndarray | None = None,
        diagonal: bool = False,
        max_steps: int | None = None,
        reward_step: float = -1.0,
        reward_goal: float = 100.0,
        reward_obstacle: float = -10.0,
        random_start: bool = False,
        n_movers: int = 0,
        mover_slots: int = 0,
    ) -> None:
        super().__init__()
        self.obstacle_ratio = float(obstacle_ratio)
        self.diagonal = bool(diagonal)
        self.reward_step = float(reward_step)
        self.reward_goal = float(reward_goal)
        self.reward_obstacle = float(reward_obstacle)
        self.random_start = bool(random_start)
        self.n_movers = int(n_movers)
        # 观测中预留的移动障碍槽位数, 固定观测维度以便跨配置对照
        self.mover_slots = max(int(mover_slots), self.n_movers)

        self.deltas = DELTA_8 if self.diagonal else DELTA_4
        self.n_actions = len(self.deltas)
        self.action_space = spaces.Discrete(self.n_actions)
        # 观测 = 位置2 + 目标2 + 八方向probe8 + 每槽移动障碍相对位置2
        self.obs_dim = 4 + len(PROBE) + 2 * self.mover_slots
        # 移动障碍相对位置可为负, 故观测范围放宽到 [-1, 1]
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        if grid is not None:
            self.grid = np.asarray(grid, dtype=np.int8)
            self.grid_size = int(self.grid.shape[0])
        else:
            self.grid_size = int(grid_size)
            self.grid = self._random_grid(seed)

        self.max_steps = int(max_steps) if max_steps else 4 * self.grid_size ** 2
        self.start = np.array([0, 0], dtype=np.int64)
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int64)
        self.agent = self.start.copy()
        self.steps = 0
        self.moves = 0
        self.path: list[np.ndarray] = [self.agent.copy()]
        self.movers: list[np.ndarray] = []
        self._rng = np.random.default_rng(seed)

    # ---------------- 地图生成 ----------------
    def _random_grid(self, seed: int) -> np.ndarray:
        """随机生成障碍地图, 并保证起点到终点连通。"""
        rng = np.random.default_rng(seed)
        n = self.grid_size
        fallback = np.zeros((n, n), dtype=np.int8)
        for _ in range(500):
            cand = (rng.random((n, n)) < self.obstacle_ratio).astype(np.int8)
            cand[0, 0] = 0
            cand[n - 1, n - 1] = 0
            if self._reachable(cand, (0, 0), (n - 1, n - 1)):
                return cand
            fallback = cand
        return fallback

    def _reachable(self, grid: np.ndarray, src, dst) -> bool:
        """BFS 判断在当前移动模型下 src 到 dst 是否连通。"""
        n = grid.shape[0]
        seen = {src}
        queue = deque([src])
        while queue:
            r, c = queue.popleft()
            if (r, c) == dst:
                return True
            for dr, dc in self.deltas:
                nr, nc = r + int(dr), c + int(dc)
                if (
                    0 <= nr < n
                    and 0 <= nc < n
                    and grid[nr, nc] == 0
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # ---------------- 观测与交互 ----------------
    def _blocked(self, pos: np.ndarray) -> bool:
        r, c = int(pos[0]), int(pos[1])
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return True
        if self.grid[r, c] == 1:
            return True
        return any(r == int(m[0]) and c == int(m[1]) for m in self.movers)

    def _free_for_mover(self, pos: np.ndarray, exclude_idx: int) -> bool:
        """移动障碍可进入的格子: 不越界、非静态障碍、非 agent、非终点、非其它移动障碍。"""
        r, c = int(pos[0]), int(pos[1])
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        if self.grid[r, c] == 1:
            return False
        if (r, c) == (int(self.agent[0]), int(self.agent[1])):
            return False
        if (r, c) == (int(self.goal[0]), int(self.goal[1])):
            return False
        for j, om in enumerate(self.movers):
            if j != exclude_idx and (r, c) == (int(om[0]), int(om[1])):
                return False
        return True

    def _place_movers(self) -> list[np.ndarray]:
        if self.n_movers <= 0:
            return []
        free = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0
            and (r, c) != (int(self.start[0]), int(self.start[1]))
            and (r, c) != (int(self.goal[0]), int(self.goal[1]))
        ]
        k = min(self.n_movers, len(free))
        picks = self._rng.choice(len(free), size=k, replace=False)
        return [np.array(free[int(i)], dtype=np.int64) for i in picks]

    def _move_movers(self) -> None:
        """完全随机游走: 每步均匀随机选一个四邻域方向, 合法则移动, 否则原地。"""
        for i, m in enumerate(self.movers):
            d = DELTA_4[int(self._rng.integers(len(DELTA_4)))]
            nxt = m + d
            if self._free_for_mover(nxt, i):
                self.movers[i] = nxt

    def _obs(self) -> np.ndarray:
        denom = float(max(self.grid_size - 1, 1))
        pos = self.agent / denom
        goal = self.goal / denom
        probes = np.array(
            [1.0 if self._blocked(self.agent + d) else 0.0 for d in PROBE],
            dtype=np.float32,
        )
        parts = [pos, goal, probes]
        for i in range(self.mover_slots):
            if i < len(self.movers):
                parts.append((self.movers[i] - self.agent) / denom)
            else:
                parts.append(np.zeros(2, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.random_start:
            free = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if self.grid[r, c] == 0
                and (r, c) != (int(self.goal[0]), int(self.goal[1]))
            ]
            r, c = free[int(self._rng.integers(len(free)))]
            self.start = np.array([r, c], dtype=np.int64)
        self.agent = self.start.copy()
        self.steps = 0
        self.moves = 0
        self.path = [self.agent.copy()]
        self.movers = self._place_movers()
        return self._obs(), {}

    def step(self, action: int):
        nxt = self.agent + self.deltas[int(action)]
        if self._blocked(nxt):
            reward = self.reward_obstacle
        else:
            self.agent = nxt
            self.moves += 1
            reward = self.reward_step

        self.steps += 1
        self.path.append(self.agent.copy())
        self._move_movers()

        terminated = bool(np.array_equal(self.agent, self.goal))
        if terminated:
            reward = self.reward_goal
        truncated = (not terminated) and self.steps >= self.max_steps

        info = {"success": terminated, "steps": self.steps, "path_length": self.moves}
        return self._obs(), float(reward), terminated, truncated, info

    # ---------------- 可视化 ----------------
    def render(self) -> np.ndarray:
        """渲染为 RGB 数组: 障碍深灰, 轨迹蓝, 起点绿, 终点红。"""
        n, cell = self.grid_size, 32
        img = np.full((n * cell, n * cell, 3), 255, dtype=np.uint8)
        for r in range(n):
            for c in range(n):
                if self.grid[r, c] == 1:
                    img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (45, 45, 45)
        for p in self.path:
            r, c = int(p[0]), int(p[1])
            img[
                r * cell + cell // 4:(r + 1) * cell - cell // 4,
                c * cell + cell // 4:(c + 1) * cell - cell // 4,
            ] = (90, 140, 225)
        for p, color in ((self.start, (40, 170, 80)), (self.goal, (210, 60, 60))):
            r, c = int(p[0]), int(p[1])
            img[r * cell + 4:(r + 1) * cell - 4, c * cell + 4:(c + 1) * cell - 4] = color
        # 移动障碍最后画(最上层), 避免与起点/终点色块重叠时被遮挡
        for m in self.movers:
            r, c = int(m[0]), int(m[1])
            img[r * cell + 6:(r + 1) * cell - 6, c * cell + 6:(c + 1) * cell - 6] = (255, 140, 0)
        return img

    def free_cells(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0
        ]

    def obs_at(self, pos) -> np.ndarray:
        """返回 AGV 位于 pos 时的观测, 不改变环境状态。用于绘制策略图。"""
        backup = self.agent.copy()
        self.agent = np.asarray(pos, dtype=np.int64)
        obs = self._obs()
        self.agent = backup
        return obs
