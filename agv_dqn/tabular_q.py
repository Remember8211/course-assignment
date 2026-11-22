"""表格型 Q-learning 基线, 用于和 DQN 对照。
状态取栅格单元编号, 更新式 Q(s,a) <- Q(s,a) + alpha*[r + gamma*max Q(s',a') - Q(s,a)]。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from astar import astar
from env_grid import GridNavEnv


class TabularQLearner:
    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=250, seed=0):
        self.q = np.zeros((int(n_states), int(n_actions)), dtype=np.float64)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_episodes = int(eps_decay_episodes)
        self.n_actions = int(n_actions)
        self.episode = 0
        self.rng = np.random.default_rng(seed)

    def epsilon(self):
        frac = min(1.0, self.episode / max(1, self.eps_decay_episodes))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[int(state)]))

    def update(self, s, a, r, s_next, done):
        bootstrap = 0.0 if done else self.gamma * float(np.max(self.q[int(s_next)]))
        self.q[int(s), int(a)] += self.alpha * (r + bootstrap - self.q[int(s), int(a)])

    def end_episode(self):
        self.episode += 1


def cell_index(env):
    return int(env.agent[0]) * env.grid_size + int(env.agent[1])


def greedy_rollout(env, agent, seed=999):
    env.reset(seed=seed)
    done = False
    info = {"success": False, "path_length": 0}
    while not done:
        action = agent.act(cell_index(env), greedy=True)
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info["path_length"], bool(info["success"])


def train(cfg):
    env = GridNavEnv(grid_size=cfg.grid, obstacle_ratio=cfg.obstacles,
                     seed=cfg.map_seed, diagonal=cfg.diagonal,
                     n_movers=cfg.movers)
    agent = TabularQLearner(n_states=env.grid_size ** 2, n_actions=env.n_actions,
                            alpha=cfg.alpha, gamma=cfg.gamma,
                            eps_decay_episodes=cfg.eps_decay, seed=cfg.seed)
    out_dir = Path(cfg.out_root) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    a_path, _ = astar(env.grid, tuple(env.start.tolist()),
                      tuple(env.goal.tolist()), env.diagonal)
    optimal_len = len(a_path) - 1 if a_path else -1

    rows, window = [], []
    t0 = time.time()
    for ep in range(1, cfg.episodes + 1):
        env.reset(seed=cfg.seed * 100_000 + ep)
        ep_reward, term, trunc = 0.0, False, False
        info = {"success": False, "steps": 0, "path_length": 0}
        while not (term or trunc):
            s = cell_index(env)
            action = agent.act(s)
            _, reward, term, trunc, info = env.step(action)
            agent.update(s, action, reward, cell_index(env), term)
            ep_reward += reward
        agent.end_episode()

        window.append(int(info["success"]))
        if len(window) > cfg.eval_window:
            window.pop(0)

        greedy_len, greedy_ok = greedy_rollout(env, agent)
        rows.append({
            "episode": ep,
            "reward": round(ep_reward, 2),
            "steps": info["steps"],
            "path_length": info["path_length"],
            "success": int(info["success"]),
            "success_rate": round(float(np.mean(window)), 4),
            "epsilon": round(agent.epsilon(), 4),
            "loss": 0.0,
            "eval_reward": "",
            "eval_success": int(greedy_ok),
            "eval_path_length": greedy_len,
        })
        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            print(f"ep {ep:4d} | reward {ep_reward:8.1f} | "
                  f"sr(window) {rows[-1]['success_rate']:.2f} | "
                  f"greedy {'OK ' if greedy_ok else 'FAIL'} len {greedy_len} "
                  f"(A* {optimal_len})")

    elapsed = time.time() - t0
    with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.save(out_dir / "q_table.npy", agent.q)
    np.save(out_dir / "grid.npy", env.grid)

    tail = rows[-cfg.eval_window:]
    summary = {
        "tag": cfg.tag,
        "method": "tabular_q",
        "episodes": cfg.episodes,
        "train_seconds": round(elapsed, 1),
        "grid_size": cfg.grid,
        "obstacle_ratio": cfg.obstacles,
        "map_seed": cfg.map_seed,
        "seed": cfg.seed,
        "alpha": cfg.alpha,
        "gamma": cfg.gamma,
        "movers": int(cfg.movers),
        "eps_decay_episodes": cfg.eps_decay,
        "astar_path_length": optimal_len,
        "final_success_rate": round(float(np.mean([r["success"] for r in tail])), 4),
        "final_greedy_success": rows[-1]["eval_success"],
        "final_greedy_path_length": rows[-1]["eval_path_length"],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[{cfg.tag}] done in {elapsed:.1f}s")
    print(f"  window success rate : {summary['final_success_rate']:.2%}")
    print(f"  greedy path length  : {summary['final_greedy_path_length']} "
          f"(A* optimal {optimal_len})")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Tabular Q-learning baseline")
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--obstacles", type=float, default=0.15)
    p.add_argument("--map-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-decay", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-window", type=int, default=50)
    p.add_argument("--diagonal", action="store_true")
    p.add_argument("--movers", type=int, default=0, help="移动障碍数量, 0=静态")
    p.add_argument("--out-root", default="results")
    p.add_argument("--tag", default="tabular")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
