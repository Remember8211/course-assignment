"""评估: 对每个训练好的模型在 train/unseen 任务集上确定性评估, 并跑 DLS 参照。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import evaluate, make_agent, make_env
from dls_baseline import dls_solve
from env3link import make_tasks

SETS = {
    "train": dict(n_obstacles=2, obstacle_radius=0.08, goal_range=(0.55, 0.95),
                  perturb_q0=0.0, seed=1000),
    "unseen": dict(n_obstacles=3, obstacle_radius=0.10, goal_range=(0.75, 0.95),
                   perturb_q0=0.3, seed=2000),
}


def build_sets():
    return {name: make_tasks(100, seed=s["seed"], n_obstacles=s["n_obstacles"],
                             ob_r=s["obstacle_radius"], goal_range=s["goal_range"],
                             perturb_q0=s["perturb_q0"])
            for name, s in SETS.items()}


def cfg_for(model_cfg: dict, set_name: str) -> dict:
    s = SETS[set_name]
    return {**model_cfg, "n_obstacles": s["n_obstacles"],
            "obstacle_radius": s["obstacle_radius"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="results_eval")
    p.add_argument("--tags", nargs="*", default=None)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = build_sets()
    rows = []

    run_dirs = sorted(Path(args.runs_root).glob("*"))
    for rd in run_dirs:
        cfgf = rd / "config.json"
        if not cfgf.exists():
            continue
        cfg = json.loads(cfgf.read_text(encoding="utf-8"))
        tag = cfg["tag"]
        if args.tags and tag not in args.tags:
            continue
        if cfg["cfg"]["n_obstacles"] == 0:
            continue  # 冒烟跑不参与主评估
        agent = make_agent(cfg["algo"], seed=cfg["seed"])
        agent.load(rd / "model.pt")
        for set_name, tasks in sets.items():
            m = evaluate(agent, tasks, cfg_for(cfg["cfg"], set_name))
            rows.append({"model": tag, "algo": cfg["algo"], "seed": cfg["seed"],
                         "set": set_name, **{k: round(v, 4) for k, v in m.items()}})
            print(f"{tag:24s} {set_name:7s} succ {m['success_rate']:.2f} "
                  f"coll {m['collision_rate']:.2f} steps {m['mean_steps']:.0f} "
                  f"clear {m['mean_clearance']:.3f}", flush=True)

    # DLS 参照
    for set_name, tasks in sets.items():
        s = SETS[set_name]
        env = make_env({"n_obstacles": s["n_obstacles"],
                        "obstacle_radius": s["obstacle_radius"]})
        infos = [dls_solve(env, t) for t in tasks]
        import numpy as np
        m = {
            "n": len(infos),
            "success_rate": float(np.mean([i["success"] for i in infos])),
            "collision_rate": float(np.mean([i["collision"] for i in infos])),
            "mean_steps": float(np.mean([i["steps"] for i in infos])),
            "mean_path": float(np.mean([i["path_length"] for i in infos])),
            "mean_clearance": float(np.mean([i["clearance"] for i in infos
                                             if i["success"]] or [np.nan])),
        }
        rows.append({"model": "DLS(reference)", "algo": "dls", "seed": -1,
                     "set": set_name, **{k: round(v, 4) for k, v in m.items()}})
        print(f"{'DLS(reference)':24s} {set_name:7s} succ {m['success_rate']:.2f} "
              f"coll {m['collision_rate']:.2f} steps {m['mean_steps']:.0f} "
              f"clear {m['mean_clearance']:.3f}", flush=True)

    fieldnames = ["model", "algo", "seed", "set", "n", "success_rate",
                  "collision_rate", "mean_steps", "mean_path", "mean_clearance"]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"summary -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
