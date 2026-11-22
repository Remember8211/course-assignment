"""共享工具: 环境/智能体构造与确定性评估。"""
from __future__ import annotations

import numpy as np

from env3link import ArmEnv3Link, make_tasks
from ppo import PPO
from sac import SAC


def make_env(cfg: dict, seed: int | None = None) -> ArmEnv3Link:
    return ArmEnv3Link(
        n_obstacles=cfg.get("n_obstacles", 2),
        obstacle_radius=cfg.get("obstacle_radius", 0.08),
        include_obstacle_obs=cfg.get("include_obstacle_obs", True),
        sparse_reward=cfg.get("sparse_reward", False),
        seed=seed,
    )


def make_agent(algo: str, seed: int, device=None, **kw):
    if algo == "ppo":
        return PPO(13, 3, seed=seed, device=device, **kw)
    if algo == "sac":
        return SAC(13, 3, seed=seed, device=device, **kw)
    raise ValueError(algo)


def rollout(agent, env: ArmEnv3Link, task: dict) -> dict:
    obs, _ = env.reset(task=task)
    done = False
    info = {"success": False, "steps": 0, "clearance": float("inf"),
            "path_length": 0.0, "collision": False}
    while not done:
        a = agent.act(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    return info


def evaluate(agent, tasks: list, cfg: dict) -> dict:
    env = make_env(cfg)
    infos = [rollout(agent, env, t) for t in tasks]
    n = len(infos)
    succ = [i["success"] for i in infos]
    return {
        "n": n,
        "success_rate": float(np.mean(succ)),
        "collision_rate": float(np.mean([i["collision"] for i in infos])),
        "mean_steps": float(np.mean([i["steps"] for i in infos])),
        "mean_path": float(np.mean([i["path_length"] for i in infos])),
        "mean_clearance": float(np.mean([i["clearance"] for i in infos if i["success"]] or [np.nan])),
    }


def eval_task_sets(seed_base: int = 0):
    return {
        "train": make_tasks(100, seed=1000 + seed_base, n_obstacles=2, ob_r=0.08,
                            goal_range=(0.55, 0.95)),
        "unseen": make_tasks(100, seed=2000 + seed_base, n_obstacles=3, ob_r=0.10,
                             goal_range=(0.75, 0.95), perturb_q0=0.3),
    }
