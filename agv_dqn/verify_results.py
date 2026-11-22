"""Replay the released models on CPU and compare with saved experiment results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

from astar import astar
from dqn import DQNAgent
from env_grid import GridNavEnv


HERE = Path(__file__).resolve().parent


def load_model(tag: str, slots: int):
    directory = HERE / 'results' / tag
    grid = np.load(directory / 'grid.npy')
    config = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    env = GridNavEnv(grid=grid, mover_slots=slots)
    agent = DQNAgent(env.obs_dim, env.n_actions, hidden=config['hidden'],
                     device=torch.device('cpu'))
    agent.load(directory / 'dqn_model.pt')
    return grid, agent


def rollout(env, agent, seed):
    obs, _ = env.reset(seed=seed)
    blocked = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs, greedy=True))
        blocked += int(reward == env.reward_obstacle)
        if terminated or truncated:
            return dict(success=int(info['success']), moves=info['path_length'],
                        steps=info['steps'], blocked=blocked)


def check_logs(tag):
    with (HERE / 'results' / tag / 'train_log.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    summary = json.loads((HERE / 'results' / tag / 'summary.json').read_text())
    assert len(rows) == summary['episodes'], (tag, 'episode count')
    for i, row in enumerate(rows):
        steps, moves, success = (int(row[k]) for k in ['steps', 'path_length', 'success'])
        expected = -moves - 10 * (steps - moves) + 101 * success
        assert float(row['reward']) == expected, (tag, i + 1, 'reward')
        window = rows[max(0, i - 49):i + 1]
        rate = np.mean([int(r['success']) for r in window])
        assert abs(float(row['success_rate']) - rate) <= 0.0001, (tag, i + 1, 'rate')
    return dict(episodes=len(rows), final50_success=sum(int(r['success']) for r in rows[-50:]))


def verify():
    torch.set_num_threads(1)
    grid, agent = load_model('main', 0)
    main = rollout(GridNavEnv(grid=grid), agent, 999)
    assert main == dict(success=1, moves=18, steps=18, blocked=0), main
    unseen = []
    with (HERE / 'results/generalization/generalization.csv').open(encoding='utf-8') as f:
        saved_maps = list(csv.DictReader(f))
    for row in saved_maps:
        seed = int(row['seed'])
        env = GridNavEnv(seed=seed)
        rec = rollout(env, agent, 7)
        optimal, _ = astar(env.grid, (0, 0), (9, 9))
        rec.update(seed=seed, astar_length=len(optimal) - 1)
        assert rec['success'] == int(row['dqn_success']), rec
        assert rec['moves'] == int(row['dqn_len']), rec
        assert rec['astar_length'] == int(row['astar_len']), rec
        unseen.append(rec)
    assert sum(r['success'] for r in unseen if r['seed'] != 0) == 8
    dynamic = []
    with (HERE / 'results/dynamic/dynamic_summary.csv').open(encoding='utf-8') as f:
        saved_dynamic = list(csv.DictReader(f))
    for tag, model in [('dyn_static_ctrl', 'static-trained'), ('dyn', 'dynamic-trained')]:
        grid, agent = load_model(tag, 2)
        for movers, env_name in [(0, 'static env'), (2, 'dynamic env')]:
            episodes = [rollout(GridNavEnv(grid=grid, mover_slots=2, n_movers=movers),
                                agent, 1000 + i) for i in range(30)]
            rec = dict(model=model, env=env_name, successes=sum(r['success'] for r in episodes),
                       mean_moves=float(np.mean([r['moves'] for r in episodes])),
                       mean_steps=float(np.mean([r['steps'] for r in episodes])),
                       blocked_total=sum(r['blocked'] for r in episodes), episodes=episodes)
            saved = next(r for r in saved_dynamic if r['model'] == model and r['env'] == env_name)
            assert rec['successes'] / 30 == float(saved['success_rate']), rec
            assert round(rec['mean_moves'], 1) == float(saved['mean_path_length']), rec
            repeated = rollout(GridNavEnv(grid=grid, mover_slots=2, n_movers=movers), agent, 1000)
            assert repeated == episodes[0], (model, env_name, 'determinism')
            dynamic.append(rec)
    hashes = {str(p.relative_to(HERE)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((HERE / 'results').rglob('*')) if p.suffix in ['.pt', '.npy']}
    return dict(status='passed', python=platform.python_version(), device='cpu', threads=1,
                dependencies={n: importlib.metadata.version(n)
                              for n in ['numpy', 'gymnasium', 'matplotlib', 'torch']},
                main=main, maps=unseen, dynamic=dynamic,
                logs={tag: check_logs(tag) for tag in ['main', 'dyn', 'dyn_static_ctrl']},
                model_and_map_hashes=hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE.parent / 'reproduction/verification.json')
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Verification passed: training map 18 moves, unseen maps 8/10.')
    for rec in result['dynamic']:
        print(rec['model'], rec['env'], f"{rec['successes']}/30", f"moves={rec['mean_moves']:.4f}")
    print('Details:', args.output)
