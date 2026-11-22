"""评估与出图脚本: 训练曲线、路径对比、策略可视化、基线对比。

用法:
    python evaluate.py --run results/main
    python evaluate.py --run results/main --compare results/tabular
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astar import astar
from dqn import DQNAgent
from env_grid import GridNavEnv

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
})


def load_log(run_dir: Path) -> list[dict]:
    with open(run_dir / "train_log.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("reward", "epsilon", "loss", "success_rate"):
            r[k] = float(r[k])
        for k in ("episode", "steps", "path_length", "success"):
            r[k] = int(r[k])
        for k in ("eval_success", "eval_path_length", "eval_reward"):
            r[k] = float(r[k]) if r[k] != "" else None
    return rows


def load_run(run_dir: Path):
    """从结果目录重建环境和已训练的 DQN 智能体。"""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    grid = np.load(run_dir / "grid.npy")
    env = GridNavEnv(grid=grid, diagonal=summary.get("diagonal", False))
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden=summary.get("hidden", 128),
        double_dqn=summary.get("double_dqn", False),
    )
    agent.load(run_dir / "dqn_model.pt")
    return env, agent, summary


def greedy_path(env: GridNavEnv, agent: DQNAgent) -> list[tuple[int, int]]:
    """跑一个纯贪心回合, 返回去重后的路径坐标列表。"""
    env.reset(seed=999)
    done = False
    while not done:
        action = agent.act(env._obs(), greedy=True)
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
    path = [(int(p[0]), int(p[1])) for p in env.path]
    return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def draw_grid(ax, grid: np.ndarray) -> None:
    """画障碍底图: 1 为障碍(深灰), 0 为可通行(白)。"""
    cmap = matplotlib.colors.ListedColormap(["#ffffff", "#2d2d2d"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="upper")
    n = grid.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)


def plot_path(ax, path, color, label, marker="o"):
    if not path:
        return
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    ax.plot(cols, rows, color=color, linewidth=2.0, marker=marker,
            markersize=4, label=label, alpha=0.9)


def fig_training_curve(rows: list[dict], summary: dict, out: Path) -> None:
    eps = [r["episode"] for r in rows]
    reward = [r["reward"] for r in rows]
    sr = [r["success_rate"] for r in rows]
    eval_pts = [(r["episode"], r["eval_path_length"]) for r in rows
                if r["eval_path_length"] is not None]
    optimal = summary.get("astar_path_length", -1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    axes[0].plot(eps, reward, color="#9ecae1", linewidth=0.9, label="episode reward")
    sm = smooth(reward, 20)
    axes[0].plot(np.arange(len(sm)) + 20 - 1, sm, color="#1f4e79",
                 linewidth=2.0, label="moving avg (20)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("(a) Training reward")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(eps, sr, color="#2c7fb8", linewidth=1.8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("(b) Success rate (sliding window)")
    axes[1].grid(alpha=0.3)

    if eval_pts:
        axes[2].plot([p[0] for p in eval_pts], [p[1] for p in eval_pts],
                     color="#d94801", linewidth=1.8, marker="o", markersize=3,
                     label="greedy path length")
    if optimal > 0:
        axes[2].axhline(optimal, color="#31a354", linestyle="--", linewidth=1.6,
                        label=f"A* optimal = {optimal}")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Path length (steps)")
    axes[2].set_title("(c) Greedy path length vs A*")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.savefig(out)
    plt.close(fig)


def fig_path_compare(env: GridNavEnv, agent: DQNAgent, summary: dict,
                     out: Path) -> dict:
    dqn_path = greedy_path(env, agent)
    a_path, a_cost = astar(env.grid, tuple(env.start.tolist()),
                           tuple(env.goal.tolist()), env.diagonal)
    a_list = [(int(r), int(c)) for r, c in (a_path or [])]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3))
    for ax, path, color, title in (
        (axes[0], a_list, "#e6550d", "A* (global planning)"),
        (axes[1], dqn_path, "#3182bd", "DQN (learned policy)"),
    ):
        draw_grid(ax, env.grid)
        plot_path(ax, path, color, "path")
        ax.scatter([env.start[1]], [env.start[0]], s=140, color="#28a745",
                   marker="s", edgecolor="black", zorder=5, label="start")
        ax.scatter([env.goal[1]], [env.goal[0]], s=140, color="#dc3545",
                   marker="*", edgecolor="black", zorder=5, label="goal")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=3, fontsize=8, frameon=False)

    fig.savefig(out)
    plt.close(fig)

    dqn_len = len(dqn_path) - 1
    a_len = len(a_list) - 1 if a_list else -1
    return {
        "astar_length": a_len,
        "dqn_length": dqn_len,
        "suboptimal_ratio": round(dqn_len / a_len, 4) if a_len > 0 else None,
    }


def fig_policy(env: GridNavEnv, agent: DQNAgent, out: Path) -> None:
    """把学到的 Q 函数画成策略箭头图 + 价值热力图。"""
    n = env.grid_size
    arrows_u = np.full((n, n), np.nan)
    arrows_v = np.full((n, n), np.nan)
    value = np.full((n, n), np.nan)

    for r, c in env.free_cells():
        q = agent.q_values(env.obs_at((r, c)))
        a = int(np.argmax(q))
        dr, dc = env.deltas[a]
        arrows_u[r, c] = dc
        # imshow(origin="upper") 使 y 轴反向, quiver 的 +V 指向屏幕上方,
        # 而网格行号增大是"向下", 故垂直分量取负才能画对方向
        arrows_v[r, c] = -dr
        value[r, c] = float(np.max(q))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))

    masked = np.ma.masked_invalid(value)
    im = axes[0].imshow(masked, cmap="YlGnBu", origin="upper")
    obs = np.ma.masked_where(env.grid == 0, env.grid.astype(float))
    axes[0].imshow(obs, cmap=matplotlib.colors.ListedColormap(["#2d2d2d"]),
                   origin="upper", alpha=0.85)
    axes[0].scatter([env.start[1]], [env.start[0]], s=120, color="#28a745",
                    marker="s", edgecolor="black", zorder=5)
    axes[0].scatter([env.goal[1]], [env.goal[0]], s=150, color="#dc3545",
                    marker="*", edgecolor="black", zorder=5)
    axes[0].set_title("(a) State value max_a Q(s,a)", fontsize=11)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    draw_grid(axes[1], env.grid)
    rows, cols = [], []
    for r, c in env.free_cells():
        rows.append(r)
        cols.append(c)
    axes[1].quiver(np.array(cols), np.array(rows),
                   arrows_u[rows, cols], arrows_v[rows, cols],
                   value[rows, cols], cmap="viridis",
                   scale=22, width=0.006, pivot="middle")
    axes[1].scatter([env.start[1]], [env.start[0]], s=120, color="#28a745",
                    marker="s", edgecolor="black", zorder=5)
    axes[1].scatter([env.goal[1]], [env.goal[0]], s=150, color="#dc3545",
                    marker="*", edgecolor="black", zorder=5)
    axes[1].set_title("(b) Greedy policy field", fontsize=11)

    fig.savefig(out)
    plt.close(fig)


def fig_compare(runs: dict[str, list[dict]], out: Path) -> None:
    """多个方法的学习曲线对比 (DQN vs 表格 Q-learning 等)。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    colors = ["#3182bd", "#e6550d", "#31a354", "#756bb1", "#d62728"]
    for i, (name, rows) in enumerate(runs.items()):
        color = colors[i % len(colors)]
        eps = [r["episode"] for r in rows]
        sm = smooth([r["reward"] for r in rows], 20)
        # 滑动平均用 valid 卷积, 长度会短 window-1, 横轴需对齐到窗口末端
        axes[0].plot(np.arange(len(sm)) + 20 - 1, sm,
                     color=color, linewidth=1.9, label=name)
        axes[1].plot(eps, [r["success_rate"] for r in rows],
                     color=color, linewidth=1.9, label=name)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward (moving avg)")
    axes[0].set_title("(a) Learning curve")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("(b) Success rate")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate and plot AGV DQN results")
    p.add_argument("--run", default="results/main", help="DQN 结果目录")
    p.add_argument("--compare", nargs="*", default=[],
                   help="其它结果目录, 用于画学习曲线对比图")
    p.add_argument("--labels", nargs="*", default=[],
                   help="与 --compare 对应的图例名称")
    args = p.parse_args()

    run_dir = Path(args.run)
    rows = load_log(run_dir)
    env, agent, summary = load_run(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_training_curve(rows, summary, fig_dir / "training_curve.png")
    metrics = fig_path_compare(env, agent, summary, fig_dir / "path_compare.png")
    fig_policy(env, agent, fig_dir / "policy_map.png")
    print(f"[{summary['tag']}] figures -> {fig_dir}")
    print(f"  A* length {metrics['astar_length']} | "
          f"DQN length {metrics['dqn_length']} | "
          f"ratio {metrics['suboptimal_ratio']}")

    if args.compare:
        runs = {"DQN (proposed)": rows}
        for i, other in enumerate(args.compare):
            label = args.labels[i] if i < len(args.labels) else Path(other).name
            runs[label] = load_log(Path(other))
        fig_compare(runs, fig_dir / "method_compare.png")
        print(f"  method_compare.png written with {len(runs)} curves")

    merged = dict(summary)
    merged.update(metrics)
    (run_dir / "eval_metrics.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
