"""训练脚本: 用 DQN 求解 AGV 栅格路径规划, 输出日志、模型与地图。

用法示例:
    python train.py --episodes 400 --tag main
    python train.py --double-dqn --tag double
    python train.py --grid 15 --obstacles 0.2 --episodes 800 --tag big
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from astar import astar
from dqn import DQNAgent
from env_grid import GridNavEnv


def greedy_rollout(env: GridNavEnv, agent: DQNAgent, seed: int = 999):
    """用贪心策略 (不探索) 跑一个回合, 返回累计奖励与 info。"""
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    info = {"success": False, "steps": 0, "path_length": 0}
    while not done:
        action = agent.act(obs, greedy=True)
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        done = term or trunc
    return total, info


def build_env(cfg) -> GridNavEnv:
    return GridNavEnv(
        grid_size=cfg.grid,
        obstacle_ratio=cfg.obstacles,
        seed=cfg.map_seed,
        diagonal=cfg.diagonal,
        n_movers=cfg.movers,
        mover_slots=cfg.mover_slots,
    )


def train(cfg) -> dict:
    env = build_env(cfg)
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden=cfg.hidden,
        lr=cfg.lr,
        gamma=cfg.gamma,
        buffer_capacity=cfg.buffer,
        batch_size=cfg.batch,
        target_update=cfg.target_update,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay_episodes=cfg.eps_decay,
        double_dqn=cfg.double_dqn,
        seed=cfg.seed,
    )

    out_dir = Path(cfg.out_root) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # A* 最优路径长度, 作为论文里的对照基准
    a_star_path, a_star_cost = astar(
        env.grid, tuple(env.start.tolist()), tuple(env.goal.tolist()), env.diagonal
    )
    optimal_len = len(a_star_path) - 1 if a_star_path else -1

    fieldnames = [
        "episode", "reward", "steps", "path_length", "success",
        "success_rate", "epsilon", "loss",
        "eval_reward", "eval_success", "eval_path_length",
    ]
    rows: list[dict] = []
    window: list[int] = []
    t0 = time.time()

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed * 100_000 + ep)
        ep_reward, losses = 0.0, []
        term = trunc = False
        info = {"success": False, "steps": 0, "path_length": 0}
        while not (term or trunc):
            action = agent.act(obs)
            nxt, reward, term, trunc, info = env.step(action)
            agent.observe(obs, action, reward, nxt, term)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            obs = nxt
            ep_reward += reward
        agent.end_episode()

        window.append(int(info["success"]))
        if len(window) > cfg.eval_window:
            window.pop(0)

        row = {
            "episode": ep,
            "reward": round(ep_reward, 2),
            "steps": info["steps"],
            "path_length": info["path_length"],
            "success": int(info["success"]),
            "success_rate": round(float(np.mean(window)), 4),
            "epsilon": round(agent.epsilon(), 4),
            "loss": round(float(np.mean(losses)), 5) if losses else 0.0,
            "eval_reward": "",
            "eval_success": "",
            "eval_path_length": "",
        }

        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            eval_reward, eval_info = greedy_rollout(env, agent)
            row["eval_reward"] = round(eval_reward, 2)
            row["eval_success"] = int(eval_info["success"])
            row["eval_path_length"] = eval_info["path_length"]
            print(
                f"ep {ep:4d} | reward {ep_reward:8.1f} | "
                f"sr(window) {row['success_rate']:.2f} | eps {row['epsilon']:.3f} | "
                f"greedy {'OK ' if eval_info['success'] else 'FAIL'} "
                f"len {eval_info['path_length']:3d} (A* {optimal_len})"
            )
        rows.append(row)

    elapsed = time.time() - t0

    with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.save(out_dir / "grid.npy", env.grid)
    agent.save(out_dir / "dqn_model.pt")

    tail = rows[-cfg.eval_window:]
    summary = {
        "tag": cfg.tag,
        "episodes": cfg.episodes,
        "train_seconds": round(elapsed, 1),
        "grid_size": cfg.grid,
        "obstacle_ratio": cfg.obstacles,
        "map_seed": cfg.map_seed,
        "seed": cfg.seed,
        "double_dqn": bool(cfg.double_dqn),
        "diagonal": bool(cfg.diagonal),
        "movers": int(cfg.movers),
        "mover_slots": int(cfg.mover_slots),
        "hidden": cfg.hidden,
        "lr": cfg.lr,
        "gamma": cfg.gamma,
        "batch_size": cfg.batch,
        "buffer": cfg.buffer,
        "target_update": cfg.target_update,
        "eps_start": cfg.eps_start,
        "eps_end": cfg.eps_end,
        "eps_decay_episodes": cfg.eps_decay,
        "obs_dim": env.obs_dim,
        "n_actions": env.n_actions,
        "astar_path_length": optimal_len,
        "final_success_rate": round(float(np.mean([r["success"] for r in tail])), 4),
        "final_greedy_success": rows[-1]["eval_success"],
        "final_greedy_path_length": rows[-1]["eval_path_length"],
        "total_learn_steps": agent.learn_steps,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[{cfg.tag}] done in {elapsed:.1f}s")
    print(f"  window success rate : {summary['final_success_rate']:.2%}")
    print(f"  greedy path length  : {summary['final_greedy_path_length']}"
          f" (A* optimal {optimal_len})")
    print(f"  output dir          : {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN for AGV grid navigation")
    p.add_argument("--grid", type=int, default=10, help="栅格边长")
    p.add_argument("--obstacles", type=float, default=0.15, help="障碍占比")
    p.add_argument("--map-seed", type=int, default=0, help="地图随机种子")
    p.add_argument("--seed", type=int, default=0, help="训练随机种子")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--target-update", type=int, default=200)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-window", type=int, default=50)
    p.add_argument("--diagonal", action="store_true", help="启用八邻域动作")
    p.add_argument("--double-dqn", action="store_true", help="启用 Double DQN")
    p.add_argument("--movers", type=int, default=0, help="移动障碍数量, 0=静态")
    p.add_argument("--mover-slots", type=int, default=0,
                   help="观测中预留移动障碍槽位, 固定观测维度")
    p.add_argument("--out-root", default="results")
    p.add_argument("--tag", default="main")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
