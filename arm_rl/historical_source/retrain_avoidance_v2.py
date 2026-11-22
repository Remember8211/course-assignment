"""Fixed-budget retraining protocol. No test-set use during model selection.

python -B arm_rl/retrain_avoidance_v2.py --stage all
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import validate_avoidance as audit
from avoidance_v2 import AvoidanceEnvV2, OBS_VERSION, curriculum
from env3link import Q0, fk, MAX_STEPS, ACTION_SCALE
from dls_baseline import dls_action

HERE=Path(__file__).resolve().parent
ROOT=HERE/'avoidance_retrain_v2'
SPLITS={'train':(1000,20261111),'validation':(50,20261112),'test':(100,20261113)}


def save(path,data):
    audit.write_json(path,data)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def initialize(root):
    root.mkdir(parents=True,exist_ok=True)
    hashes={}
    for split,(groups,seed) in SPLITS.items():
        directory=root/'data'/split; directory.mkdir(parents=True,exist_ok=True)
        if not (directory/'tasks.json').exists():
            previous=audit.SEED
            try:
                audit.SEED=seed
                audit.generate_tasks(directory,groups,300000)
            finally:
                audit.SEED=previous
        package=read(directory/'tasks.json')
        assert package['seed']==seed and package['groups']==groups, 'Incomplete/wrong task bank; no relaxation permitted.'
        hashes[split]=audit.sha256(directory/'tasks.json')
        print(f'Task bank {split}: {groups} groups, hash={hashes[split]}',flush=True)
    # Group IDs intentionally repeat across splits; geometry must never repeat.
    identities={}
    for split in SPLITS:
        signatures={hashlib.sha256(json.dumps({k:t[k] for k in ('q0','goal','obstacles')},sort_keys=True).encode()).hexdigest()
                    for t in read(root/'data'/split/'tasks.json')['tasks']}
        assert len(signatures)==SPLITS[split][0]*3
        for other, old in identities.items():
            assert not signatures & old, (split,other)
        identities[split]=signatures
    config={'version':'avoidance_retrain_v2','observation_version':OBS_VERSION,'obs_dim':21,
      'training_seeds':[0,1,2],'steps_per_seed':300000,'checkpoint_interval':10000,
      'splits':{k:{'groups':v[0],'seed':v[1],'sha256':hashes[k]} for k,v in SPLITS.items()},
      'tolerances':[0.02,0.08],'max_steps':120,'action_scale':0.05,'collision_resolution':0.001,
      'refinement_resolution':0.0005,'device':'cpu','threads':1,
      'reward':{'progress':5.0,'time':0.01,'action_squared':0.001,'safety':0.1,'safety_margin':0.03,
                'success':20.0,'collision':-20.0,'timeout':-5.0},
      'sac':{'hidden':128,'lr':0.0003,'gamma':0.99,'buffer':100000,'batch':256,'warmup':1000,'tau':0.005,'target_entropy':-3},
      'curriculum':[{'through_step':50000,'ABC':[.5,.5,0]},{'through_step':100000,'ABC':[.2,.3,.5]},
                    {'through_step':300000,'ABC':[.1,.2,.7]}],
      'selection':['max_validation_C_success_0.02','max_validation_B_success_0.02','min_validation_C_error','earlier_step'],
      'runtime':{'python':platform.python_version(),'numpy':np.__version__},
      'original_protected_hashes':{str(p.relative_to(HERE.parent)):audit.sha256(p) for p in
        [HERE/'env3link.py',HERE/'sac.py',HERE/'train.py',HERE/'runs/sac_s0/model.pt',HERE/'runs/sac_s1/model.pt',
         HERE.parent/'paper/论文_机械臂版_v3.docx', HERE/'avoidance_validation_20261107/tasks.json',
         HERE/'avoidance_validation_20261107/episodes.json']}}
    save(root/'protocol.json',config)
    return config


def reward_precheck(root):
    tasks=read(root/'data/train/tasks.json')['tasks']
    cases=[]
    # All training C references, not a policy- or reward-selected subset.
    for task in tasks:
        if task['scenario']!='C': continue
        env=AvoidanceEnvV2(); env.reset(task=task)
        ref_return=0.0; discounted=0.0; status=None
        for n,target in enumerate(np.asarray(task['reference_q'])[1:]):
            obs,r,term,trunc,info=env.step((target-env.q)/ACTION_SCALE)
            ref_return+=r; discounted+=0.99**n*r
            if term or trunc:
                status=info['status']; break
        env.reset(task=task); stay=0.0; stay_discounted=0.0
        for n in range(120):
            _,r,term,trunc,info=env.step(np.zeros(3)); stay+=r; stay_discounted+=0.99**n*r
            if term or trunc: break
        # Target-directed geometric controller often collides; record actual outcome.
        env.reset(task=task); collision_return=0.0; collision_status=None
        for n in range(120):
            _,r,term,trunc,info=env.step(dls_action(env,avoid_gain=0.0)); collision_return+=r
            if term or trunc: collision_status=info['status']; break
        passed=status=='success' and ref_return>stay and discounted>stay_discounted
        cases.append({'task_id':task['task_id'],'reference_status':status,'reference_return':ref_return,
          'reference_discounted_return':discounted,'stationary_return':stay,'stationary_discounted_return':stay_discounted,
          'noavoid_status':collision_status,'noavoid_return':collision_return,'passed':passed})
    passed=bool(cases) and all(x['passed'] for x in cases)
    result={'passed':passed,'reference_cases':len(cases),'failures':[x for x in cases if not x['passed']], 'cases':cases}
    save(root/'reward_precheck.json',result)
    print(f'Reward precheck: {sum(x["passed"] for x in cases)}/{len(cases)} references beat stationary behavior.',flush=True)
    if not passed: raise RuntimeError('Reward feasibility gate failed; no training allowed and no automatic retuning.')


def rollout(task,method,agent=None,tolerance=.02,resolution=.001,trajectory=True):
    env=AvoidanceEnvV2(tolerance,resolution); obs,_=env.reset(task=task)
    qs=[env.q.tolist()]; ees=[fk(env.q)[-1].tolist()]; errors=[float(np.linalg.norm(np.asarray(ees[-1])-env.task['goal']))]
    actions=[]; gaps=[]; status=None
    while status is None:
        action=agent.act(obs,deterministic=True) if agent is not None else dls_action(env,avoid_gain=0 if method=='dls_noavoid' else 1.5)
        obs,status,detail=env.advance(action)
        if trajectory:
            actions.append(np.asarray(action).tolist()); qs.append(env.q.tolist()); ees.append(fk(env.q)[-1].tolist())
            errors.append(detail['dist']); gaps.append(audit.finite(detail['min_clearance']))
    result={'task_id':task['task_id'],'group':task['group'],'scenario':task['scenario'],'method':method,
      'tolerance':tolerance,'status':status,'steps':env.steps,'final_error':detail['dist'],
      'min_clearance':audit.finite(env.ep_min_clear),'path_length':env.ep_path}
    if trajectory: result['trajectory']={'q':qs,'ee':ees,'error':errors,'actions':actions,'step_min_clearance':gaps}
    return result


def evaluate(tasks,agent,method,tolerance=.02,trajectory=False,resolution=.001):
    return [rollout(t,method,agent,tolerance,resolution,trajectory) for t in tasks]


def metrics(episodes):
    result={}
    for scenario in 'ABC':
        rows=[e for e in episodes if e['scenario']==scenario]
        result[scenario]={'success':sum(e['status']=='success' for e in rows)/len(rows),
          'collision':sum(e['status']=='collision' for e in rows)/len(rows),
          'timeout':sum(e['status']=='timeout' for e in rows)/len(rows),
          'mean_error':float(np.mean([e['final_error'] for e in rows]))}
    return result


def train(root,seed):
    import torch
    from sac import SAC
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    directory=root/'runs'/f'sac_v2_s{seed}'; directory.mkdir(parents=True,exist_ok=True)
    if (directory/'completed.json').exists():
        print(f'Seed {seed} already complete.',flush=True); return
    if (directory/'train_log.csv').exists():
        raise RuntimeError('Incomplete run exists; refusing an implicit restart. Inspect before selecting a fresh run directory.')
    tasks=read(root/'data/train/tasks.json')['tasks']; validation=read(root/'data/validation/tasks.json')['tasks']
    by_class={s:[t for t in tasks if t['scenario']==s] for s in 'ABC'}
    rng=np.random.default_rng(seed*31+7)
    def choose(step):
        s=str(rng.choice(list('ABC'),p=curriculum(step)))
        return by_class[s][int(rng.integers(len(by_class[s])))]
    agent=SAC(21,3,seed=seed,device='cpu'); env=AvoidanceEnvV2(); task=choose(1); obs,_=env.reset(task=task)
    best=None; best_data=None; started=time.time(); rewards=[]; ep_reward=0.; statuses=Counter(); sampled=Counter({task['scenario']:1})
    log_fields=['step','seconds','mean_recent_return','alpha']+[f'{s}_{m}' for s in 'ABC' for m in ['success','collision','timeout','mean_error']]
    with (directory/'train_log.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=log_fields); writer.writeheader(); stream.flush()
        for step in range(1,300001):
            action=agent.act(obs); nxt,r,term,trunc,info=env.step(action)
            agent.store(obs,action,r,nxt,term); update=agent.update(); obs=nxt; ep_reward+=r
            if term or trunc:
                rewards.append(ep_reward); statuses[info['status']]+=1; ep_reward=0.
                if step < 300000:
                    task=choose(step+1); sampled[task['scenario']]+=1; obs,_=env.reset(task=task)
            if step%1000==0:
                save(directory/'progress.json',{'seed':seed,'step':step,'seconds':time.time()-started,
                    'training_terminal_counts':dict(statuses),'sampled_episodes':dict(sampled)})
            if step%10000==0:
                validation_episodes=evaluate(validation,agent,f'sac_v2_s{seed}')
                score=metrics(validation_episodes)
                key=(score['C']['success'],score['B']['success'],-score['C']['mean_error'],-step)
                checkpoint=directory/f'checkpoint_{step:06d}.pt'
                agent.save(checkpoint)
                # Save optimizer/target/temperature state separately for a complete audit trail.
                torch.save({'step':step,'observation_version':OBS_VERSION,'obs_dim':21,
                    'q1_target':agent.q1t.state_dict(),'q2_target':agent.q2t.state_dict(),
                    'log_alpha':agent.log_alpha.detach(),'actor_optimizer':agent.opt_a.state_dict(),
                    'critic_optimizer':agent.opt_q.state_dict(),'alpha_optimizer':agent.opt_al.state_dict()},
                    directory/f'optimizer_{step:06d}.pt')
                save(directory/f'validation_{step:06d}.json',validation_episodes)
                if best is None or key>best:
                    best=key; best_data={'seed':seed,'step':step,'validation':score,'selection_key':list(key),
                      'observation_version':OBS_VERSION,'obs_dim':21,'checkpoint':checkpoint.name}
                    shutil.copyfile(checkpoint,directory/'model.pt'); save(directory/'selected.json',best_data)
                row={'step':step,'seconds':round(time.time()-started,2),'mean_recent_return':float(np.mean(rewards[-20:])) if rewards else 0.,
                     'alpha':float(agent.log_alpha.exp().detach())}
                row.update({f'{s}_{m}':v for s,values in score.items() for m,v in values.items()})
                writer.writerow(row); stream.flush()
                print(f'Seed {seed}, step {step}: val A/B/C={score["A"]["success"]:.2f}/{score["B"]["success"]:.2f}/{score["C"]["success"]:.2f}, elapsed={time.time()-started:.0f}s',flush=True)
    save(directory/'completed.json',{'seed':seed,'steps':300000,'seconds':time.time()-started,
        'selected':best_data,'model_sha256':audit.sha256(directory/'model.pt'),
        'training_terminal_counts':dict(statuses),'sampled_episodes':dict(sampled)})


def final_evaluate(root):
    import torch
    from sac import SAC
    torch.set_num_threads(1)
    tasks=read(root/'data/test/tasks.json')['tasks']
    diagnostic=read(HERE/'avoidance_validation_20261107/tasks.json')['tasks']
    agents={}
    for seed in (0,1,2):
        folder=root/'runs'/f'sac_v2_s{seed}'
        assert (folder/'completed.json').exists()
        agent=SAC(21,3,seed=seed,device='cpu'); agent.load(folder/'model.pt'); agent.actor.eval(); agents[f'sac_v2_s{seed}']=agent
    out=root/'results'; out.mkdir(exist_ok=True)
    all_ep=[]; diag_ep=[]
    for tol in (.02,.08):
        for method in [*agents,'dls_noavoid','dls_avoid']:
            rows=evaluate(tasks,agents.get(method),method,tol,True)
            all_ep.extend(rows)
            print(f'Final {method}, tol={tol}: {metrics(rows)}',flush=True)
        for method,agent in agents.items():
            diag_ep.extend(evaluate(diagnostic,agent,method,tol,True))
    assert len(all_ep)==3000 and len(diag_ep)==180
    audit_data={}
    for label,episodes,bank in [('test',all_ep,tasks),('diagnostic',diag_ep,diagnostic)]:
        misses,checks=audit.audit_saved_trajectories(episodes,bank,.0005)
        rerun=bool(misses)
        if rerun:
            episodes[:]=[rollout(next(t for t in bank if t['task_id']==e['task_id']),e['method'],agents.get(e['method']),e['tolerance'],.0005) for e in episodes]
            remaining,checks=audit.audit_saved_trajectories(episodes,bank,.0005); assert not remaining
        checkmap={r['episode']:r['min_clearance'] for r in checks}
        for e in episodes:
            key=f"{e['method']}:{e['task_id']}:{e['tolerance']}"
            if checkmap[key] is not None: e['min_clearance']=min(e['min_clearance'],checkmap[key])
        audit_data[label]={'missed_collisions':misses,'rerun':rerun,'checks':checks}
    save(out/'episodes.json',all_ep); save(out/'diagnostic_episodes.json',diag_ep); save(out/'refinement.json',audit_data)
    repeats=[]
    for method,agent in agents.items():
        for scenario in 'ABC':
            task=next(t for t in tasks if t['scenario']==scenario)
            assert rollout(task,method,agent)==rollout(task,method,agent)
            repeats.append({'method':method,'task_id':task['task_id'],'passed':True})
    protocol=read(root/'protocol.json')
    for split,entry in protocol['splits'].items(): assert audit.sha256(root/'data'/split/'tasks.json')==entry['sha256']
    for rel,h in protocol['original_protected_hashes'].items(): assert audit.sha256(HERE.parent/rel)==h
    save(out/'checks.json',{'passed':True,'test_episodes':len(all_ep),'diagnostic_episodes':len(diag_ep),
         'repeatability':repeats,'task_hashes_unchanged':True,'original_artifacts_unchanged':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','prepare','train','evaluate','report'],default='all')
    parser.add_argument('--seed',type=int,choices=[0,1,2]); parser.add_argument('--out',type=Path,default=ROOT)
    args=parser.parse_args(); root=args.out.resolve()
    if args.stage in ('all','prepare'):
        initialize(root); reward_precheck(root)
    if args.stage in ('all','train'):
        assert read(root/'reward_precheck.json')['passed']
        for seed in [args.seed] if args.seed is not None else (0,1,2): train(root,seed)
    if args.stage in ('all','evaluate'): final_evaluate(root)
    if args.stage in ('all','report'):
        from report_avoidance_v2 import build_report
        build_report(root)


if __name__=='__main__': main()
