"""Independent paired-task audit of saved SAC policies; never trains a model.

Run from any directory: python -B arm_rl/validate_avoidance.py
Task generation finishes and is hashed before any policy is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from env3link import (ArmEnv3Link, ACTION_SCALE, JOINT_LIMIT, LINK_LEN,
                     LINK_RADIUS, MAX_STEPS, Q0, fk, seg_point_dist)
from dls_baseline import dls_action

HERE = Path(__file__).resolve().parent
SEED = 20261107
OB_RADIUS = 0.08
REFERENCE_MARGIN = 0.01
METHODS = ('sac_s0', 'sac_s1', 'dls_noavoid', 'dls_avoid')
TOLERANCES = (0.08, 0.02)
BACK_OBSTACLES = np.array([[-0.6, 0.3], [-0.6, -0.3]])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return float(value) if np.isfinite(value) else None


def batch_fk(qs):
    qs = np.asarray(qs, dtype=float).reshape(-1, 3)
    theta = np.cumsum(qs, axis=1)
    increments = LINK_LEN[None, :, None] * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    return np.concatenate((np.zeros((len(qs), 1, 2)), np.cumsum(increments, axis=1)), axis=1)


def surface_clearances(qs, obstacles, radius=OB_RADIUS):
    """Minimum capsule-to-circle surface distance for each configuration."""
    pts = batch_fk(qs)
    if len(obstacles) == 0:
        return np.full(len(pts), np.inf)
    centers = np.asarray(obstacles, float)
    a = pts[:, :-1, None, :]
    v = pts[:, 1:, None, :] - a
    w = centers[None, None, :, :] - a
    t = np.clip(np.sum(w*v, axis=-1) / np.sum(v*v, axis=-1), 0.0, 1.0)
    dist = np.linalg.norm(w - t[..., None]*v, axis=-1)
    return np.min(dist, axis=(1, 2)) - radius - LINK_RADIUS


def motion_samples(q0, q1, resolution=0.001):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = max(1, int(np.ceil(np.max(np.abs(q1-q0))/resolution)))
    return np.linspace(q0, q1, n+1)


def inspect_motion(q0, q1, obstacles, radius=OB_RADIUS, resolution=0.001):
    qs = motion_samples(q0, q1, resolution)
    clear = surface_clearances(qs, obstacles, radius)
    hits = np.flatnonzero(clear <= 0.0)
    collision = len(hits) > 0
    if collision:
        qs = qs[:int(hits[0])+1]
        clear = clear[:len(qs)]
    ee = batch_fk(qs)[:, -1]
    return {'q_end': qs[-1], 'min_clearance': float(clear.min()),
            'collision': bool(collision),
            'path_length': float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum()),
            'samples': qs}


class AuditEnv(ArmEnv3Link):
    """Keep the original policy observation; audit geometry/termination only."""
    def __init__(self, tolerance=0.08, resolution=0.001):
        super().__init__(obstacle_radius=OB_RADIUS)
        self.tolerance = float(tolerance)
        self.resolution = float(resolution)

    def reset(self, seed=None, task=None):
        if task is not None:
            task = {**task, 'q0': np.asarray(task['q0'], float),
                    'goal': np.asarray(task['goal'], float),
                    'obstacles': [np.asarray(c, float) for c in task['obstacles']]}
            self.ob_r = float(task.get('obstacle_radius', OB_RADIUS))
        obs, info = super().reset(seed=seed, task=task)
        self.ep_min_clear = float(surface_clearances([self.q], self.task['obstacles'], self.ob_r)[0])
        return obs, info

    def advance(self, action):
        delta = np.clip(np.asarray(action, float), -1.0, 1.0)*ACTION_SCALE
        requested = np.clip(self.q+delta, -JOINT_LIMIT, JOINT_LIMIT)
        check = inspect_motion(self.q, requested, self.task['obstacles'], self.ob_r, self.resolution)
        self.q = check['q_end'].copy()
        self.steps += 1
        self.ep_min_clear = min(self.ep_min_clear, check['min_clearance'])
        self.ep_path += check['path_length']
        pts = fk(self.q)
        dist = float(np.linalg.norm(pts[-1]-self.task['goal']))
        if check['collision']:
            status = 'collision'
        elif dist < self.tolerance:
            status = 'success'
        elif self.steps >= MAX_STEPS:
            status = 'timeout'
        else:
            status = None
        return self._obs(pts), status, {**check, 'dist': dist}


def edge_valid(a, b, obstacles, margin=REFERENCE_MARGIN, resolution=0.005):
    # Planner edges use a conservative displacement bound between grid samples.
    qs = motion_samples(a, b, resolution)
    clearance = surface_clearances(qs, obstacles)
    bound = float(np.dot([1.0, 0.6, 0.3], np.abs(qs[1]-qs[0])))/2
    return bool(np.min(clearance) >= margin+bound)


def rrt_connect(start, goal, obstacles, rng, max_extensions=10000):
    trees = [([np.array(start, float)], [-1]), ([np.array(goal, float)], [-1])]
    used = 0

    def extend(tree, target):
        nonlocal used
        nodes, parents = tree
        arr = np.asarray(nodes)
        j = int(np.argmin(np.sum((arr-target)**2, axis=1)))
        distance = float(np.linalg.norm(target-nodes[j]))
        if distance < 1e-10:
            return j, True
        candidate = nodes[j] + (target-nodes[j])*min(1.0, 0.2/distance)
        used += 1
        if not edge_valid(nodes[j], candidate, obstacles):
            return None, False
        nodes.append(candidate); parents.append(j)
        return len(nodes)-1, distance <= 0.2

    def backtrack(tree, j):
        path = []
        while j != -1:
            path.append(tree[0][j]); j = tree[1][j]
        return path[::-1]

    while used < max_extensions:
        side = used % 2
        sample = (trees[1-side][0][0] if rng.random() < 0.1 else
                  rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, 3))
        ia, _ = extend(trees[side], sample)
        if ia is None:
            continue
        target = trees[side][0][ia]
        while used < max_extensions:
            ib, reached = extend(trees[1-side], target)
            if ib is None:
                break
            if reached:
                pa = backtrack(trees[side], ia)
                pb = backtrack(trees[1-side], ib)
                if side == 0:
                    return pa + pb[-2::-1], used
                return pb + pa[-2::-1], used
    return None, used


def simplify_path(path, obstacles, rng):
    path = list(path)
    for _ in range(100):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j > i+1 and edge_valid(path[i], path[j], obstacles):
            path = path[:i+1] + path[j:]
    return np.asarray(path)


def discretize_path(path):
    result = [np.asarray(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        result.extend(motion_samples(a, b, ACTION_SCALE)[1:])
    return np.asarray(result)


def path_clearance(path, obstacles, resolution=0.001):
    points = [motion_samples(a, b, resolution) for a, b in zip(path[:-1], path[1:])]
    points = np.concatenate(points) if points else np.asarray(path)
    return float(np.min(surface_clearances(points, obstacles)))


def generate_tasks(out, count=10, cap=10000):
    rng = np.random.default_rng(SEED)
    ee0 = fk(Q0)[-1]
    angle0 = np.arctan2(ee0[1], ee0[0])
    tasks, records, rejects = [], [], Counter()
    accepted = 0
    start_time = time.time()
    for candidate in range(1, cap+1):
        q_goal = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, 3)
        goal = fk(q_goal)[-1]
        radius = np.linalg.norm(goal)
        angle = np.arctan2(goal[1], goal[0])
        diff = (angle-angle0+np.pi) % (2*np.pi)-np.pi
        reason = None
        record = {'candidate_index': candidate}
        if not (0.55 <= radius <= 0.95 and abs(diff) <= 0.9):
            reason = 'goal_outside_original_sector'
        elif np.linalg.norm(goal-ee0) < 0.20:
            reason = 'start_goal_too_close'
        else:
            t = float(rng.uniform(0.35, 0.65))
            v = goal-ee0
            normal = np.array([-v[1], v[0]])/np.linalg.norm(v)
            blocker = ee0+t*v+rng.uniform(-0.01, 0.01)*normal
            obstacles_c = np.array([blocker, BACK_OBSTACLES[1]])
            record.update({'q_goal': q_goal.tolist(), 'goal': goal.tolist(),
                           'blocker': blocker.tolist(), 'blocker_t': t})
            if surface_clearances([Q0], obstacles_c)[0] < REFERENCE_MARGIN:
                reason = 'initial_arm_clearance'
            elif np.min(np.linalg.norm(obstacles_c-goal, axis=1)) < OB_RADIUS+LINK_RADIUS+0.08+REFERENCE_MARGIN:
                reason = 'goal_success_region_clearance'
            elif surface_clearances([q_goal], obstacles_c)[0] < REFERENCE_MARGIN:
                reason = 'goal_configuration_clearance'
            elif np.linalg.norm(obstacles_c[0]-obstacles_c[1]) < 2*OB_RADIUS+0.02:
                reason = 'overlapping_obstacles'
            elif not edge_valid(Q0, q_goal, BACK_OBSTACLES):
                reason = 'B_reference_direct_motion_blocked'
            else:
                plan_rng = np.random.default_rng(SEED+candidate*1009)
                path, extensions = rrt_connect(Q0, q_goal, obstacles_c, plan_rng)
                record['planner_extensions'] = extensions
                if path is None:
                    reason = 'no_reference_found_within_budget'
                else:
                    path = discretize_path(simplify_path(path, obstacles_c, plan_rng))
                    if len(path)-1 > MAX_STEPS:
                        reason = 'reference_exceeds_step_limit'
                    else:
                        gap = path_clearance(path, obstacles_c)
                        if gap < REFERENCE_MARGIN:
                            reason = 'reference_margin_failed'
                        else:
                            accepted += 1
                            record.update({'accepted_group': accepted, 'reference_steps':len(path)-1,
                                           'reference_min_clearance':gap})
                            direct = discretize_path([Q0, q_goal])
                            for scenario, obstacles, reference in (
                                ('A', np.empty((0, 2)), direct),
                                ('B', BACK_OBSTACLES, direct),
                                ('C', obstacles_c, path)):
                                tasks.append({'task_id':f'g{accepted:02d}_{scenario}',
                                    'group':accepted,'scenario':scenario,'q0':Q0.tolist(),
                                    'goal':goal.tolist(),'q_goal':q_goal.tolist(),
                                    'obstacles':obstacles.tolist(),'obstacle_radius':OB_RADIUS,
                                    'reference_q':reference.tolist(),'reference_steps':len(reference)-1,
                                    'reference_min_clearance':finite(path_clearance(reference, obstacles)),
                                    'candidate_index':candidate,
                                    'blocker_distance':float(seg_point_dist(ee0, goal, blocker)) if scenario=='C' else None,
                                    'blocker_t':t if scenario=='C' else None})
                            print(f'Accepted group {accepted:02d}: candidate={candidate}, reference={len(path)-1} steps, gap={gap:.4f} m', flush=True)
        record['reason'] = reason or 'accepted'
        records.append(record)
        if reason:
            rejects[reason] += 1
        if accepted >= count:
            break
        if candidate % 100 == 0:
            print(f'Generation candidates={candidate}, accepted={accepted}, rejected={dict(rejects)}', flush=True)
    log = {'seed':SEED,'candidates_seen':candidate,'accepted_groups':accepted,
           'requested_groups':count,'candidate_cap':cap,'rejections':dict(rejects),
           'seconds':time.time()-start_time,'records':records,
           'selection_uses_policy_results':False,
           'goal_proposal':'uniform joint goal in [-pi,pi]^3; FK then sector filtering',
           'planner':'RRT-Connect, step 0.2, 10000 extension cap, 100 shortcuts'}
    package = {'seed':SEED,'groups':accepted,'tasks':tasks}
    write_json(out/'tasks.json', package)
    write_json(out/'generation_log.json', log)
    return package


def rollout(task, method, agent=None, tolerance=0.08, resolution=0.001):
    env = AuditEnv(tolerance, resolution)
    obs, _ = env.reset(task=task)
    qs, ees = [env.q.tolist()], [fk(env.q)[-1].tolist()]
    errors = [float(np.linalg.norm(np.asarray(ees[-1])-env.task['goal']))]
    actions, gaps = [], []
    status = None
    while status is None:
        if method.startswith('sac'):
            action = agent.act(obs, deterministic=True)
        else:
            action = dls_action(env, avoid_gain=0.0 if method=='dls_noavoid' else 1.5)
        actions.append(np.asarray(action, float).tolist())
        obs, status, detail = env.advance(action)
        qs.append(env.q.tolist()); ees.append(fk(env.q)[-1].tolist())
        errors.append(detail['dist']); gaps.append(finite(detail['min_clearance']))
    return {'task_id':task['task_id'],'group':task['group'],'scenario':task['scenario'],
            'method':method,'tolerance':tolerance,'status':status,'steps':env.steps,
            'final_error':errors[-1],'min_clearance':finite(env.ep_min_clear),
            'path_length':env.ep_path,
            'trajectory':{'q':qs,'ee':ees,'actions':actions,
                          'step_min_clearance':gaps,'error':errors}}


def run_matrix(tasks, agents, resolution):
    episodes = []
    for tolerance in TOLERANCES:
        for method in METHODS:
            for task in tasks:
                episodes.append(rollout(task, method, agents.get(method), tolerance, resolution))
            counts = Counter(e['status'] for e in episodes if e['method']==method and e['tolerance']==tolerance)
            print(f'Evaluated tol={tolerance} method={method}: {dict(counts)}', flush=True)
    return episodes


def audit_saved_trajectories(episodes, tasks, resolution):
    by_id = {t['task_id']:t for t in tasks}
    misses, rows = [], []
    for ep in episodes:
        task = by_id[ep['task_id']]
        minimum = path_clearance(ep['trajectory']['q'], task['obstacles'], resolution)
        new_collision = minimum <= 0 and ep['status'] != 'collision'
        key = f"{ep['method']}:{ep['task_id']}:{ep['tolerance']}"
        if new_collision:
            misses.append(key)
        rows.append({'episode':key, 'min_clearance':finite(minimum),
                     'missed_collision':bool(new_collision)})
    return misses, rows


def validate_references(tasks):
    rows = []
    for task in tasks:
        path = np.asarray(task['reference_q'])
        assert len(path)-1 <= MAX_STEPS
        assert np.max(np.abs(np.diff(path,axis=0))) <= ACTION_SCALE+1e-10
        assert np.max(np.abs(path)) <= JOINT_LIMIT+1e-10
        assert np.allclose(path[0], Q0)
        error = float(np.linalg.norm(fk(path[-1])[-1]-task['goal']))
        minimum = path_clearance(path, task['obstacles'], 0.0005)
        assert error < 1e-9 and minimum >= REFERENCE_MARGIN-1e-10
        # Replay the stored per-step increments, bypassing early reaching only for this feasibility check.
        q = Q0.copy()
        for nxt in path[1:]:
            result = inspect_motion(q, nxt, task['obstacles'], resolution=0.0005)
            assert not result['collision']
            q = result['q_end']
        if task['scenario']=='C':
            assert task['blocker_distance'] < 0.015
        rows.append({'task_id':task['task_id'],'steps':len(path)-1,
                     'final_error':error,'min_clearance':finite(minimum),'passed':True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'avoidance_validation_20261107')
    parser.add_argument('--stage', choices=['all','generate','evaluate','report'], default='all')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        from avoidance_report import build_report
        build_report(out); return
    if args.stage in ('all','generate'):
        if (out/'tasks.json').exists():
            raise RuntimeError('Fixed tasks already exist. Use --stage evaluate, or a new output directory.')
        package = generate_tasks(out)
    else:
        package = json.loads((out/'tasks.json').read_text(encoding='utf-8'))
    if args.stage == 'generate':
        return
    tasks = package['tasks']
    reference_checks = validate_references(tasks)
    frozen_hash = sha256(out/'tasks.json')
    # Models are loaded only after tasks have been frozen and reference paths verified.
    import torch
    import matplotlib
    from sac import SAC
    torch.set_num_threads(1)
    agents, model_hashes = {}, {}
    for seed in (0,1):
        name = f'sac_s{seed}'
        model_path = HERE/'runs'/name/'model.pt'
        model_hashes[name] = sha256(model_path)
        agents[name] = SAC(13,3,seed=seed,device='cpu')
        agents[name].load(model_path)
        agents[name].actor.eval()
    config = {'seed':SEED,'groups':package['groups'],'requested_groups':10,
        'methods':list(METHODS),'tolerances':list(TOLERANCES),'max_steps':MAX_STEPS,
        'action_scale':ACTION_SCALE,'link_lengths':LINK_LEN.tolist(),'link_radius':LINK_RADIUS,
        'obstacle_radius':OB_RADIUS,'q0':Q0.tolist(),'reference_margin':REFERENCE_MARGIN,
        'coarse_resolution':0.001,'fine_resolution':0.0005,'model_hashes':model_hashes,
        'tasks_sha256':frozen_hash,'device':'cpu','policy_observation_dim':13,
        'policy_updated':False,'self_collision_checked':False,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                   'torch':torch.__version__,'matplotlib':matplotlib.__version__},
        'source_hashes':{name:sha256(HERE/name) for name in
                         ['env3link.py','sac.py','dls_baseline.py','validate_avoidance.py']}}
    write_json(out/'config.json',config)
    episodes = run_matrix(tasks, agents, 0.001)
    write_json(out/'episodes_coarse.json',episodes)
    misses, refined_rows = audit_saved_trajectories(episodes,tasks,0.0005)
    rerun = bool(misses)
    if rerun:
        print(f'Fine check found {len(misses)} missed collision episodes; re-evaluating ALL at 0.0005.',flush=True)
        episodes = run_matrix(tasks,agents,0.0005)
        remaining, refined_rows = audit_saved_trajectories(episodes,tasks,0.0005)
        assert not remaining
    refinement = {'coarse_resolution':0.001,'fine_resolution':0.0005,
                  'missed_collision_episodes':misses,'rerun_at_fine':rerun,
                  'audited_episodes':len(episodes),'reference_checks':reference_checks,
                  'episode_checks':refined_rows}
    # Fine-grid minima improve reported geometry even when classifications agree.
    refined_map = {r['episode']:r['min_clearance'] for r in refined_rows}
    for ep in episodes:
        key = f"{ep['method']}:{ep['task_id']}:{ep['tolerance']}"
        fine = refined_map[key]
        if fine is not None:
            ep['min_clearance'] = min(ep['min_clearance'],fine)
    write_json(out/'refinement.json',refinement)
    write_json(out/'episodes.json',episodes)
    assert sha256(out/'tasks.json') == frozen_hash
    assert all(sha256(HERE/'runs'/name/'model.pt')==h for name,h in model_hashes.items())
    repeatability=[]
    if tasks:
        for method in METHODS:
            task = next(t for t in tasks if t['scenario']=='C')
            a = rollout(task,method,agents.get(method),0.08,0.0005 if rerun else 0.001)
            b = rollout(task,method,agents.get(method),0.08,0.0005 if rerun else 0.001)
            assert a == b
            repeatability.append({'method':method,'task_id':task['task_id'],'passed':True})
    import unittest
    suite = unittest.defaultTestLoader.discover(str(HERE),pattern='test_avoidance_validation.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    checks={'passed':result.wasSuccessful(),'tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),
            'reference_replay':reference_checks,'repeatability':repeatability,
            'tasks_unchanged':True,'models_unchanged':True,
            'episode_count':len(episodes),'expected_episode_count':len(tasks)*8,
            'outcomes_mutually_exclusive':all(e['status'] in ['success','collision','timeout'] for e in episodes)}
    write_json(out/'checks.json',checks)
    if not result.wasSuccessful() or result.testsRun==0:
        raise RuntimeError('Validation tests did not pass or were not discovered.')
    from avoidance_report import build_report
    build_report(out)
    print(f'Completed verification: {out}',flush=True)


if __name__ == '__main__':
    main()
