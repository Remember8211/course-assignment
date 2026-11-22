"""训练入口: PPO/SAC 在平面三连杆避障环境上训练, 周期确定性评估, 存模型与日志。"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from common import evaluate, make_agent, make_env
from env3link import make_tasks


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="sac", choices=["ppo", "sac"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--no-obstacles", action="store_true")
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--no-obs-feat", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse()
    cfg = {
        "n_obstacles": 0 if args.no_obstacles else 2,
        "obstacle_radius": 0.08,
        "include_obstacle_obs": not args.no_obs_feat,
        "sparse_reward": args.sparse,
    }
    tag = args.tag or args.algo + ("_smoke" if args.no_obstacles else "") + \
        ("_sparse" if args.sparse else "") + ("_noobsfeat" if args.no_obs_feat else "") + \
        f"_s{args.seed}"
    out = Path(args.out_root) / tag
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    kw = {"lr": args.lr}
    if args.algo == "ppo":
        kw["epochs"] = args.epochs
        kw["ent_coef"] = args.ent_coef
    agent = make_agent(args.algo, seed=args.seed, **kw)
    env = make_env(cfg, seed=args.seed * 31 + 7)
    eval_tasks = make_tasks(20, seed=999, n_obstacles=cfg["n_obstacles"],
                            ob_r=cfg["obstacle_radius"])

    rows = []
    recent = deque(maxlen=20)
    obs, _ = env.reset()
    ep_reward = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if args.algo == "ppo":
            a, logp, v = agent.act_train(obs)
            nxt, r, term, trunc, info = env.step(a)
            agent.store(obs, a, logp, r, term, v)
            if agent.buffer_full():
                last = 0.0 if term else agent.value(nxt)
                agent.update(last)
        else:
            a = agent.act(obs)
            nxt, r, term, trunc, info = env.step(a)
            agent.store(obs, a, r, nxt, term)
            agent.update()
        ep_reward += r
        obs = nxt
        if term or trunc:
            recent.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(agent, eval_tasks, cfg)
            rows.append({
                "step": step,
                "mean_ep_reward": round(float(np.mean(recent)), 3) if recent else 0.0,
                "eval_success": round(m["success_rate"], 3),
                "eval_collision": round(m["collision_rate"], 3),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[{tag}] step {step:7d} | epR {rows[-1]['mean_ep_reward']:8.2f} | "
                  f"eval succ {m['success_rate']:.2f} | coll {m['collision_rate']:.2f} | "
                  f"{rows[-1]['seconds']:.0f}s", flush=True)

    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "mean_ep_reward", "eval_success",
                                          "eval_collision", "seconds"])
        w.writeheader(); w.writerows(rows)
    agent.save(out / "model.pt")
    (out / "config.json").write_text(json.dumps({
        "algo": args.algo, "seed": args.seed, "steps": args.steps, "cfg": cfg,
        "tag": tag,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
