"""A* 全局路径规划, 作为深度强化学习方法的对照基线。"""
from __future__ import annotations

import heapq

import numpy as np

DELTA_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DELTA_8 = DELTA_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    diagonal: bool = False,
) -> tuple[list[tuple[int, int]] | None, float]:
    """返回 (路径, 代价)。路径为坐标列表 (含起点终点), 不可达时返回 (None, inf)。"""
    grid = np.asarray(grid)
    n = grid.shape[0]
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    deltas = DELTA_8 if diagonal else DELTA_4

    def h(node: tuple[int, int]) -> float:
        dr = abs(node[0] - goal[0])
        dc = abs(node[1] - goal[1])
        # 八邻域用 Chebyshev, 四邻域用 Manhattan
        return float(max(dr, dc)) if diagonal else float(dr + dc)

    g_score = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap = [(h(start), 0.0, start)]
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, cost, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1], cost
        closed.add(cur)
        for dr, dc in deltas:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n):
                continue
            if grid[nxt[0], nxt[1]] == 1 or nxt in closed:
                continue
            step_cost = 1.0 if (dr == 0 or dc == 0) else float(np.sqrt(2.0))
            tentative = cost + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative
                came_from[nxt] = cur
                heapq.heappush(open_heap, (tentative + h(nxt), tentative, nxt))

    return None, float("inf")
