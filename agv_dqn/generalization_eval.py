"""最小版泛化对比: 不重训, 直接用已训好的 DQN 与表格 Q, 在训练图与未见地图上评估。

产出:
    results/generalization/generalization.png   对比图
    results/generalization/generalization.csv   逐地图明细
    results/generalization/generalization.json  汇总

用法:
    python generalization_eval.py
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
from env_grid import GridNavEnv
from evaluate import load_run

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})


def greedy_rollout_dqn(env, agent, max_steps=None):
    obs, _ = env.reset(seed=7)
    done = False
    steps = 0
    limit = max_steps or env.max_steps
    info = {"success": False, "path_length": 0}
    while not done and steps < limit:
        action = agent.act(obs, greedy=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1
    return bool(info["success"]), info["path_length"]


def greedy_rollout_tabular(env, q_table, max_steps=None):
    env.reset(seed=7)
    done = False
    steps = 0
    limit = max_steps or env.max_steps
    info = {"success": False, "path_length": 0}
    while not done and steps < limit:
        cell = int(env.agent[0]) * env.grid_size + int(env.agent[1])
        action = int(np.argmax(q_table[cell]))
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1
    return bool(info["success"]), info["path_length"]


def evaluate_map(grid, dqn_agent, q_table, diagonal=False):
    """在一张地图上分别评估 DQN / 表格 Q / A*, 返回明细 dict。"""
    env_d = GridNavEnv(grid=grid, diagonal=diagonal)
    env_t = GridNavEnv(grid=grid, diagonal=diagonal)
    d_ok, d_len = greedy_rollout_dqn(env_d, dqn_agent)
    t_ok, t_len = greedy_rollout_tabular(env_t, q_table)
    a_path, _ = astar(grid, (0, 0), (grid.shape[0] - 1, grid.shape[0] - 1), diagonal)
    a_len = len(a_path) - 1 if a_path else -1
    return {
        "dqn_success": int(d_ok), "dqn_len": d_len,
        "tab_success": int(t_ok), "tab_len": t_len,
        "astar_len": a_len,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dqn-run", default="results/main")
    p.add_argument("--tab-run", default="results/tabular")
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--obstacles", type=float, default=0.15)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--n-unseen", type=int, default=10)
    p.add_argument("--unseen-start", type=int, default=1)
    p.add_argument("--out", default="results/generalization")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_proto, dqn_agent, dqn_summary = load_run(args.dqn_run)
    q_table = np.load(Path(args.tab_run) / "q_table.npy")

    seeds = [args.train_seed] + list(
        range(args.unseen_start, args.unseen_start + args.n_unseen)
    )
    rows = []
    for i, seed in enumerate(seeds):
        is_train = seed == args.train_seed
        grid = GridNavEnv(grid_size=args.grid, obstacle_ratio=args.obstacles,
                          seed=seed).grid
        rec = evaluate_map(grid, dqn_agent, q_table)
        rec.update({"map": ("train" if is_train else f"unseen{i}"),
                    "seed": seed, "is_train": int(is_train)})
        rows.append(rec)
        tag = "TRAIN" if is_train else "unseen"
        print(f"{tag:7s} seed={seed:3d} | DQN {rec['dqn_success']} len {rec['dqn_len']:3d} | "
              f"Tab {rec['tab_success']} len {rec['tab_len']:3d} | A* {rec['astar_len']}")

    train_rows = [r for r in rows if r["is_train"]]
    unseen_rows = [r for r in rows if not r["is_train"]]
    summary = {
        "dqn_train_success": float(np.mean([r["dqn_success"] for r in train_rows])),
        "dqn_unseen_success": float(np.mean([r["dqn_success"] for r in unseen_rows])),
        "tab_train_success": float(np.mean([r["tab_success"] for r in train_rows])),
        "tab_unseen_success": float(np.mean([r["tab_success"] for r in unseen_rows])),
        "n_unseen": len(unseen_rows),
    }

    fieldnames = ["map", "seed", "is_train", "dqn_success", "dqn_len",
                  "tab_success", "tab_len", "astar_len"]
    with open(out_dir / "generalization.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "generalization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 画图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0))
    methods = ["DQN", "Tabular Q"]
    train_rate = [summary["dqn_train_success"], summary["tab_train_success"]]
    unseen_rate = [summary["dqn_unseen_success"], summary["tab_unseen_success"]]
    x = np.arange(len(methods))
    width = 0.35
    axes[0].bar(x - width / 2, train_rate, width, label="train map", color="#74c476")
    axes[0].bar(x + width / 2, unseen_rate, width, label="unseen maps", color="#fd8d3c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("(a) Generalization: train vs unseen")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    for xi, (tr, un) in enumerate(zip(train_rate, unseen_rate)):
        axes[0].text(xi - width / 2, tr + 0.02, f"{tr:.2f}", ha="center", fontsize=8)
        axes[0].text(xi + width / 2, un + 0.02, f"{un:.2f}", ha="center", fontsize=8)

    mat = np.array([[r["dqn_success"] for r in rows],
                    [r["tab_success"] for r in rows]], dtype=float)
    im = axes[1].imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[1].set_yticks([0, 1], methods)
    axes[1].set_xticks(range(len(rows)), [r["map"] for r in rows],
                       rotation=45, ha="right", fontsize=7)
    axes[1].set_title("(b) Per-map success (green=success)")
    for r_i in range(mat.shape[0]):
        for c_i in range(mat.shape[1]):
            axes[1].text(c_i, r_i, int(mat[r_i, c_i]), ha="center", va="center",
                         fontsize=7, color="black")

    labels = [r["map"] for r in rows]
    d_len = [r["dqn_len"] if r["dqn_success"] else np.nan for r in rows]
    a_len = [r["astar_len"] for r in rows]
    xi = np.arange(len(rows))
    axes[2].bar(xi - width / 2, d_len, width, label="DQN path len", color="#3182bd")
    axes[2].plot(xi + width / 2, a_len, "o--", color="#31a354", label="A* optimal")
    axes[2].set_xticks(xi, labels, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("Path length (steps)")
    axes[2].set_title("(c) Path length vs A* (DQN, solved maps)")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.savefig(out_dir / "generalization.png")
    plt.close(fig)

    print("\nsummary:", json.dumps(summary, ensure_ascii=False))
    print(f"figure -> {out_dir / 'generalization.png'}")


if __name__ == "__main__":
    main()
