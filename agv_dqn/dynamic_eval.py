"""动态障碍效果评估: 静态训练 vs 动态训练, 在静态/动态环境下的 2x2 成功率对照,
并输出移动障碍避障的连续帧图。

用法:
    python dynamic_eval.py --static-run results/dyn_static_ctrl --dynamic-run results/dyn
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

from dqn import DQNAgent
from env_grid import GridNavEnv

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})


def load_model(run_dir: Path, mover_slots: int, grid_size: int,
               obstacle_ratio: float, map_seed: int):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    grid = np.load(run_dir / "grid.npy")
    env = GridNavEnv(grid=grid, mover_slots=mover_slots)
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden=summary.get("hidden", 128),
        double_dqn=summary.get("double_dqn", False),
    )
    agent.load(run_dir / "dqn_model.pt")
    return env, agent, summary


def rollout(env, agent, seed, capture_every=None, max_frames=6):
    obs, _ = env.reset(seed=seed)
    frames = []
    done = False
    t = 0
    if capture_every is not None:
        frames.append((0, env.render()))
    info = {"success": False, "path_length": 0}
    while not done:
        action = agent.act(obs, greedy=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        t += 1
        if capture_every is not None and t % capture_every == 0 and len(frames) < max_frames:
            frames.append((t, env.render()))
    if capture_every is not None and len(frames) < max_frames and frames[-1][0] != t:
        frames.append((t, env.render()))
    return bool(info["success"]), info["path_length"], frames


def eval_pair(env_proto_grid, agent, mover_slots, movers, n_episodes, diagonal=False):
    successes, lengths = [], []
    frames_out = None
    for i in range(n_episodes):
        env = GridNavEnv(grid=env_proto_grid.copy(), mover_slots=mover_slots,
                         n_movers=movers, diagonal=diagonal)
        cap = 6 if i == 0 and movers > 0 else None
        ok, ln, frames = rollout(env, agent, seed=1000 + i, capture_every=cap)
        successes.append(int(ok))
        lengths.append(ln)
        if frames and frames_out is None:
            frames_out = frames
    return float(np.mean(successes)), float(np.mean(lengths)), frames_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--static-run", default="results/dyn_static_ctrl")
    p.add_argument("--dynamic-run", default="results/dyn")
    p.add_argument("--movers", type=int, default=2)
    p.add_argument("--mover-slots", type=int, default=2)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--out", default="results/dynamic")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_env, s_agent, s_sum = load_model(args.static_run, args.mover_slots, 0, 0, 0)
    d_env, d_agent, d_sum = load_model(args.dynamic_run, args.mover_slots, 0, 0, 0)
    grid = s_env.grid

    rows = []
    frames = None
    for model_name, agent in (("static-trained", s_agent), ("dynamic-trained", d_agent)):
        for env_name, movers in (("static env", 0), ("dynamic env", args.movers)):
            sr, ml, fr = eval_pair(grid, agent, args.mover_slots, movers, args.episodes)
            rows.append({"model": model_name, "env": env_name,
                         "success_rate": round(sr, 3), "mean_path_length": round(ml, 1)})
            print(f"{model_name:16s} | {env_name:12s} | success {sr:.2f} | mean len {ml:.1f}")
            if fr is not None and frames is None and model_name == "dynamic-trained":
                frames = fr

    with open(out_dir / "dynamic_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "env", "success_rate", "mean_path_length"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / "dynamic_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2x2 对照: (a) 成功率 (b) 平均路径长度(效率)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    models = ["static-trained", "dynamic-trained"]
    static_rate = [next(r["success_rate"] for r in rows
                        if r["model"] == m and r["env"] == "static env") for m in models]
    dynamic_rate = [next(r["success_rate"] for r in rows
                         if r["model"] == m and r["env"] == "dynamic env") for m in models]
    static_len = [next(r["mean_path_length"] for r in rows
                       if r["model"] == m and r["env"] == "static env") for m in models]
    dynamic_len = [next(r["mean_path_length"] for r in rows
                        if r["model"] == m and r["env"] == "dynamic env") for m in models]
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, static_rate, w, label="static env", color="#74c476")
    axes[0].bar(x + w / 2, dynamic_rate, w, label=f"dynamic env ({args.movers} movers)",
                color="#fd8d3c")
    axes[0].set_xticks(x, models)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("(a) Success rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    for xi, (a, b) in enumerate(zip(static_rate, dynamic_rate)):
        axes[0].text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        axes[0].text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    axes[1].bar(x - w / 2, static_len, w, label="static env", color="#74c476")
    axes[1].bar(x + w / 2, dynamic_len, w, label=f"dynamic env ({args.movers} movers)",
                color="#fd8d3c")
    axes[1].axhline(18, color="#31a354", linestyle="--", linewidth=1.4,
                    label="Static-map lower bound = 18")
    axes[1].set_xticks(x, models)
    axes[1].set_ylabel("Mean path length (steps)")
    axes[1].set_title("(b) Efficiency under moving obstacles")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    for xi, (a, b) in enumerate(zip(static_len, dynamic_len)):
        axes[1].text(xi - w / 2, a + 0.4, f"{a:.1f}", ha="center", fontsize=8)
        axes[1].text(xi + w / 2, b + 0.4, f"{b:.1f}", ha="center", fontsize=8)
    fig.savefig(out_dir / "dynamic_robustness.png")
    plt.close(fig)

    # 连续帧图
    if frames:
        n = len(frames)
        cols = min(3, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 4.0 * rows_n))
        axes = np.atleast_1d(axes).ravel()
        for i, (step, fr) in enumerate(frames):
            axes[i].imshow(fr)
            axes[i].set_title(f"t = {step}", fontsize=9)
            axes[i].axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.savefig(out_dir / "dynamic_frames.png")
        plt.close(fig)
        print(f"frames -> {out_dir / 'dynamic_frames.png'}")

    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
