"""额外过程图: 环境交互闭环、SAC训练循环框图、完整训练曲线(三面板)。"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def _box(ax, xy, w, h, text, fc="#eef4fb", ec="#2c7fb8", fs=8):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, linewidth=1.0))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p0, p1, color="#444444"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.1, color=color))


def fig_interaction():
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    _box(ax, (0.3, 1.6), 1.9, 1.1, "观测 $o_t$\n关节角/末端/目标/障碍", fc="#eef4fb")
    _box(ax, (3.0, 1.6), 2.0, 1.1, "策略 $\\pi_\\theta$\nSAC actor", fc="#e6f2e6", ec="#31a354")
    _box(ax, (5.7, 1.6), 2.0, 1.1, "动作 $a_t$\n关节增量 $\\Delta q$", fc="#eef4fb")
    _box(ax, (8.2, 1.6), 1.6, 1.1, "环境\n正运动学/碰撞", fc="#fdf1e7", ec="#e6550d")
    _arrow(ax, (2.2, 2.15), (3.0, 2.15))
    _arrow(ax, (5.0, 2.15), (5.7, 2.15))
    _arrow(ax, (7.7, 2.15), (8.2, 2.15))
    # 反馈环
    _arrow(ax, (7.7, 1.8), (7.7, 0.6))
    _arrow(ax, (7.7, 0.6), (0.3 + 0.95, 0.6))
    _arrow(ax, (0.3 + 0.95, 0.6), (0.3 + 0.95, 1.6))
    ax.text(4.6, 0.45, "$o_{t+1},\\ r_t$（状态转移与奖励）", ha="center", fontsize=8)
    ax.set_title("环境交互闭环", fontsize=10)
    fig.savefig(FIG / "f2_loop.png")
    plt.close(fig)


def fig_training_flow():
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.axis("off")
    steps = [
        (1.0, 7.2, "策略采样 $a_t \\sim \\pi_\\theta(\\cdot|o_t)$"),
        (1.0, 6.0, "环境执行 $o_{t+1}, r_t$"),
        (1.0, 4.8, "存入经验回放池 $D$"),
        (1.0, 3.6, "采样 batch，更新双 Q 网络"),
        (1.0, 2.4, "更新策略与温度 $\\alpha$"),
        (1.0, 1.2, "软更新目标网络 $\\theta^- \\leftarrow \\tau\\theta + (1-\\tau)\\theta^-$"),
    ]
    for x, y, s in steps:
        _box(ax, (x, y), 6.0, 0.9, s, fc="#eef4fb")
    for y0, y1 in [(7.2, 6.0), (6.0, 4.8), (4.8, 3.6), (3.6, 2.4), (2.4, 1.2)]:
        _arrow(ax, (4.0, y0), (4.0, y1 + 0.9))
    _arrow(ax, (4.0, 1.2), (4.0, 0.35))
    _arrow(ax, (4.0, 0.35), (0.9, 0.35))
    _arrow(ax, (0.9, 0.35), (0.9, 7.65))
    ax.text(4.0, 0.15, "直到收敛（重复 $N$ 个环境步）", ha="center", fontsize=8)
    ax.set_title("SAC 训练循环", fontsize=10)
    fig.savefig(FIG / "f3_flow.png")
    plt.close(fig)


def fig_curves():
    def load(tag):
        rows = list(csv.DictReader(open(HERE / "runs" / tag / "train_log.csv",
                                        encoding="utf-8")))
        return (np.array([int(r["step"]) for r in rows]),
                np.array([float(r["mean_ep_reward"]) for r in rows]),
                np.array([float(r["eval_success"]) for r in rows]),
                np.array([float(r["eval_collision"]) for r in rows]))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for tag, label, color in (("sac_s0", "dense", "#3182bd"),
                              ("sac_sparse_s0", "sparse", "#8c8c8c")):
        x, rw, sc, co = load(tag)
        axes[0].plot(x, rw, color=color, lw=1.6, label=label)
        axes[1].plot(x, sc, color=color, lw=1.6, label=label)
        axes[2].plot(x, co, color=color, lw=1.6, label=label)
    for ax, yl, tt in ((axes[0], "Episode reward", "(a) Reward"),
                       (axes[1], "Success rate", "(b) Success rate"),
                       (axes[2], "Collision rate", "(c) Collision rate")):
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(yl)
        ax.set_title(tt)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylim(-0.05, 1.05)
    fig.savefig(FIG / "f4_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_interaction()
    fig_training_flow()
    fig_curves()
    print("extra figs ->", FIG)
