"""论文数据图(学术样式): 训练过程曲线、轨迹、关节、成功率柱状、clearance对比。
Times New Roman, 英文轴标, 灰度配色, 无图内标题。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import make_agent, make_env
from env3link import REACH_MAX, fk, make_tasks

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _log(tag):
    rows = list(csv.DictReader(open(HERE / "runs" / tag / "train_log.csv",
                                    encoding="utf-8")))
    return (np.array([int(r["step"]) for r in rows]),
            np.array([float(r["mean_ep_reward"]) for r in rows]),
            np.array([float(r["eval_success"]) for r in rows]),
            np.array([float(r["eval_collision"]) for r in rows]))


def fig_curves():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    dense = [_log(t) for t in ("sac_s0", "sac_s1")]
    sparse = _log("sac_sparse_s0")
    # dense 两种子画成两条细线, 稀疏一条虚线
    for i, (x, rw, sc, co) in enumerate(dense):
        axes[0].plot(x, rw, color=("#08519c" if i == 0 else "#6baed6"),
                     lw=1.0 if i else 1.4,
                     label="dense, seed %d" % i)
        axes[1].plot(x, sc, color=("#08519c" if i == 0 else "#6baed6"),
                     lw=1.0 if i else 1.4,
                     label="dense, seed %d" % i)
        axes[2].plot(x, co, color=("#08519c" if i == 0 else "#6baed6"),
                     lw=1.0 if i else 1.4,
                     label="dense, seed %d" % i)
    axes[0].plot(sparse[0], sparse[1], color="#fd8d3c", lw=1.4, ls="--",
                 label="sparse")
    axes[1].plot(sparse[0], sparse[2], color="#fd8d3c", lw=1.4, ls="--",
                 label="sparse")
    axes[2].plot(sparse[0], sparse[3], color="#fd8d3c", lw=1.4, ls="--",
                 label="sparse")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Episode reward")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Success rate")
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Collision rate")
    for ax, tag in zip(axes, "abc"):
        ax.text(-0.18, 1.04, "(%s)" % tag, transform=ax.transAxes, fontsize=11)
        ax.grid(alpha=0.3, lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "f1_curves.png")
    plt.close(fig)


def _recorded_rollout(agent, env, task):
    obs, _ = env.reset(task=task)
    ee_path = [fk(env.q)[-1].copy()]
    qs = [env.q.copy()]
    done = False
    info = {}
    while not done:
        a = agent.act(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        ee_path.append(fk(env.q)[-1].copy())
        qs.append(env.q.copy())
    return np.array(ee_path), np.array(qs), info


def fig_schematic():
    cfg = {"n_obstacles": 2, "obstacle_radius": 0.08,
           "include_obstacle_obs": True, "sparse_reward": False}
    agent = make_agent("sac", seed=0)
    agent.load(HERE / "runs" / "sac_s0" / "model.pt")
    env = make_env(cfg)
    task = make_tasks(1, seed=1000, n_obstacles=2, ob_r=0.08)[0]
    obs, _ = env.reset(task=task)
    qs = [env.q.copy()]
    done = False
    while not done:
        a = agent.act(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        qs.append(env.q.copy())
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for c in task["obstacles"]:
        ax.add_patch(plt.Circle(c, 0.08, color="#555555", alpha=0.4))
    idx = [0, len(qs) // 2, len(qs) - 1]
    colors = ["#08519c", "#6baed6", "#fd8d3c"]
    for k, i in enumerate(idx):
        pts = fk(qs[min(i, len(qs) - 1)])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=colors[k], lw=2.2, markersize=4)
    ax.scatter([0], [0], s=55, marker="s", color="k", zorder=6)
    ax.scatter([task["goal"][0]], [task["goal"][1]], s=75, marker="*",
               color="#d7191c", edgecolor="k", zorder=6)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "f0_schematic.png")
    plt.close(fig)


def fig_trajectories_and_joints():
    cfg = {"n_obstacles": 2, "obstacle_radius": 0.08,
           "include_obstacle_obs": True, "sparse_reward": False}
    agent = make_agent("sac", seed=0)
    agent.load(HERE / "runs" / "sac_s0" / "model.pt")
    env = make_env(cfg)
    tasks = make_tasks(8, seed=1000, n_obstacles=2, ob_r=0.08)

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    rec = None
    for t in tasks[:6]:
        path, qs, info = _recorded_rollout(agent, env, t)
        color = "#2c7fb8" if info["success"] else "#fd8d3c"
        ax.plot(path[:, 0], path[:, 1], "-", color=color, lw=1.1, alpha=0.9)
        for c in t["obstacles"]:
            ax.add_patch(plt.Circle(c, 0.08, color="#555555", alpha=0.45))
        ax.scatter([t["goal"][0]], [t["goal"][1]], s=40, marker="*",
                   color="#d7191c", edgecolor="k", zorder=5)
        if rec is None and info["success"]:
            rec = qs
    ax.scatter([0], [0], s=60, marker="s", color="k", zorder=6)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "f2_trajectories.png")
    plt.close(fig)

    if rec is None:
        _, rec, info = _recorded_rollout(agent, env, tasks[0])
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for j in range(3):
        ax.plot(rec[:, j], lw=1.2, color=["#08519c", "#41ab5d", "#fd8d3c"][j],
                label="q%d" % (j + 1))
    ax.set_xlabel("Step")
    ax.set_ylabel("Joint angle / rad")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "f3_joints.png")
    plt.close(fig)


def read_summary():
    rows = list(csv.DictReader(open(HERE / "results_eval" / "summary.csv",
                                    encoding="utf-8")))
    for r in rows:
        for k in ("success_rate", "collision_rate", "mean_steps",
                  "mean_path", "mean_clearance"):
            r[k] = float(r[k])
    return rows


def _agg(rows, key, set_name):
    sel = [r for r in rows if r["model"] == key and r["set"] == set_name]
    return {k: float(np.mean([r[k] for r in sel])) for k in
            ("success_rate", "collision_rate", "mean_steps",
             "mean_path", "mean_clearance")} if sel else None


def _agg_sac(rows, set_name):
    vals = [a for a in (_agg(rows, m, set_name) for m in ("sac_s0", "sac_s1")) if a]
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]} if vals else None


def _group(rows, set_name):
    return {
        "SAC": _agg_sac(rows, set_name),
        "Sparse": _agg(rows, "sac_sparse_s0", set_name),
        "NoObs": _agg(rows, "sac_noobsfeat_s0", set_name),
        "DLS": _agg(rows, "DLS(reference)", set_name),
    }


def fig_bars(rows):
    labels = ["SAC", "Sparse", "NoObs", "DLS"]
    colors = ["#08519c", "#fd8d3c", "#41ab5d", "#969696"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, set_name, tag in ((axes[0], "train", "a"),
                              (axes[1], "unseen", "b")):
        g = _group(rows, set_name)
        vals = [g[m]["success_rate"] if g[m] else 0.0 for m in labels]
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Success rate")
        ax.text(-0.18, 1.04, "(%s)" % tag, transform=ax.transAxes, fontsize=11)
        for xi, v in enumerate(vals):
            ax.text(xi, v + 0.02, "%.2f" % v, ha="center", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "f4_bars.png")
    plt.close(fig)


def fig_clearance(rows):
    labels = ["SAC", "Sparse", "NoObs", "DLS"]
    colors = ["#08519c", "#fd8d3c", "#41ab5d", "#969696"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, key, ylabel, tag in ((axes[0], "mean_clearance",
                                  "Mean min clearance / m", "a"),
                                 (axes[1], "mean_path",
                                  "Mean EE path length / m", "b")):
        g = _group(rows, "unseen")
        vals = [g[m][key] if g[m] and np.isfinite(g[m][key]) else 0.0 for m in labels]
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.text(-0.18, 1.04, "(%s)" % tag, transform=ax.transAxes, fontsize=11)
        for xi, v in enumerate(vals):
            ax.text(xi, v + 0.01 * max(vals), "%.2f" % v, ha="center", fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "f5_clearance.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_schematic()
    fig_curves()
    fig_trajectories_and_joints()
    rows = read_summary()
    fig_bars(rows)
    fig_clearance(rows)
    print("figures ->", FIG)
