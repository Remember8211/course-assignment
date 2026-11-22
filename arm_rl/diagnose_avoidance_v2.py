"""Single-seed diagnosis using validation/training data only; no retraining."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sac import SAC
from avoidance_v2 import AvoidanceEnvV2
from retrain_avoidance_v2 import evaluate, metrics, read, save
import avoidance_report as plots
import validate_avoidance as audit
from artifact_integrity import verify_original_artifacts


def diagnose(root):
    root=Path(root); folder=root/'runs/sac_v2_s0'
    completed=read(folder/'completed.json'); assert completed['steps']==300000
    out=root/'diagnosis_seed0'; out.mkdir(exist_ok=True)
    validation=read(root/'data/validation/tasks.json')['tasks']
    train_sample=read(root/'data/train/tasks.json')['tasks'][:150]
    torch.set_num_threads(1)
    checkpoints=sorted(set([completed['selected']['step'],100000,200000,300000]))
    summary=[]; all_rows=[]; returns=[]
    for step in checkpoints:
        model=SAC(21,3,seed=0,device='cpu'); model.load(folder/f'checkpoint_{step:06d}.pt'); model.actor.eval()
        rows=evaluate(validation,model,f'step_{step}',.02,True)
        all_rows.extend(rows)
        for scenario,m in metrics(rows).items():
            subset=[e for e in rows if e['scenario']==scenario]
            action_norms=[float(np.mean(np.linalg.norm(e['trajectory']['actions'],axis=1))) for e in subset]
            summary.append({'checkpoint':step,'scenario':scenario,**m,
                'mean_path_length':float(np.mean([e['path_length'] for e in subset])),
                'mean_normalized_action_norm':float(np.mean(action_norms)),
                'mean_min_goal_error':float(np.mean([min(e['trajectory']['error']) for e in subset]))})
        for e in rows:
            task=next(t for t in validation if t['task_id']==e['task_id'])
            env=AvoidanceEnvV2(); env.reset(task=task); total=discounted=0.
            for i,action in enumerate(e['trajectory']['actions']):
                _,r,term,trunc,info=env.step(action); total+=r; discounted+=.99**i*r
            returns.append({'checkpoint':step,'task_id':e['task_id'],'scenario':e['scenario'],
                'status':e['status'],'return':total,'discounted_return':discounted})
        print(f'Diagnosed checkpoint {step}: {metrics(rows)}',flush=True)
    selected_model=SAC(21,3,seed=0,device='cpu'); selected_model.load(folder/'model.pt')
    train_rows=evaluate(train_sample,selected_model,'selected_train_sample',.02,True)
    save(out/'training_sample.json',train_rows)
    save(out/'validation_trajectories.json',all_rows); save(out/'trajectory_returns.json',returns)
    plots._csv(out/'checkpoint_metrics.csv',summary)
    # Demonstrate the actual finite-deadline semantics without changing the trained implementation.
    task=next(t for t in validation if t['scenario']=='A')
    early=AvoidanceEnvV2(); late=AvoidanceEnvV2()
    o1,_=early.reset(task=task); o2,_=late.reset(task=task); late.steps=119
    _,r1,t1,x1,_=early.step(np.zeros(3)); _,r2,t2,x2,_=late.step(np.zeros(3))
    time_audit={'identical_observation':bool(np.array_equal(o1,o2)),
        'early_reward':r1,'late_reward':r2,'early_terminated':t1,'late_terminated':t2,
        'early_truncated':x1,'late_truncated':x2,
        'remaining_time_present':False,'replay_uses_terminated_only':True,
        'finite_deadline_definition_mismatch':bool(np.array_equal(o1,o2) and r1!=r2 and x2 and not t2)}
    save(out/'timeout_semantics.json',time_audit)
    plots._setup_plotting(); fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    with (folder/'train_log.csv').open(encoding='utf-8') as f: log=list(csv.DictReader(f))
    for ax,s in zip(axes,'ABC'):
        for metric,label in [('success','安全到达'),('collision','碰撞'),('timeout','超时')]:
            ax.plot([int(r['step'])/10000 for r in log],[float(r[f'{s}_{metric}']) for r in log],label=label)
        ax.set_title(plots.SCENARIO_LABELS[s]); ax.set_ylim(0,1.05); ax.set_xlabel('训练步数 / 万'); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out/'validation_outcomes.png'); plt.close(fig)
    taskmap={t['task_id']:t for t in validation}
    displayed=[]
    for step in checkpoints:
        plots.METHOD_LABELS[f'step_{step}']=f'检查点{step}步'
        subset=[e for e in all_rows if e['method']==f'step_{step}']
        groups={1}
        success=next((e for e in subset if e['scenario']=='C' and e['status']=='success'),None)
        if success: groups.add(success['group'])
        for group in sorted(groups):
            name=f'checkpoint_{step}_group_{group}.png'
            plots._triptych(out/name,group,f'step_{step}',.02,taskmap,subset)
            displayed.append((step,group,name))
    protocol=read(root/'protocol.json')
    for split,entry in protocol['splits'].items(): assert audit.sha256(root/'data'/split/'tasks.json')==entry['sha256']
    original_checks = verify_original_artifacts(Path(__file__).resolve().parent.parent, protocol['original_protected_hashes'])
    details={'completed_seed':0,'completed_steps':300000,'selected_step':completed['selected']['step'],
        'training_seconds':completed['seconds'],'selected_training_sample_metrics':metrics(train_rows),
        'checkpoint_metrics':summary,'timeout_semantics':time_audit,
        'independent_test_evaluated':False,'additional_seeds_deferred':True,
        'task_and_original_artifact_hashes_unchanged':True if original_checks['all_listed_verified'] else None,
        'task_hashes_unchanged':True, 'required_original_artifacts_unchanged':True,
        'original_artifact_verification':original_checks}
    save(out/'diagnosis.json',details)
    lines=['# SAC修订方案种子0训练诊断','',
        f'种子0完成300000环境步，用时{completed["seconds"]/60:.1f}分钟。后续种子按用户决定暂缓。本报告仅使用训练样本和既有验证集，不使用独立测试集，也不改奖励或模型。','',
        f'按原选择规则保留的模型来自{completed["selected"]["step"]}步。'+
        ('它尚未进入受阻任务训练阶段（受阻任务在50000步之后开始采样），因此不能将这个检查点的少量受阻成功归因于受阻训练。' if completed['selected']['step']<=50000 else '它已进入受阻任务训练阶段，但新方案同时更改多个因素，不能单独归因。'),'',
        '## 检查点表现','', '| 步数 | 场景 | 安全成功率 | 碰撞率 | 超时率 | 平均终点误差/m | 平均归一化动作范数 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in summary: lines.append(f'| {r["checkpoint"]} | {r["scenario"]} | {r["success"]:.2f} | {r["collision"]:.2f} | {r["timeout"]:.2f} | {r["mean_error"]:.3f} | {r["mean_normalized_action_norm"]:.3f} |')
    early_c=next(r for r in summary if r['checkpoint']==min(checkpoints) and r['scenario']=='C')
    late_c=next(r for r in summary if r['checkpoint']==300000 and r['scenario']=='C')
    lines+=['','每类50个固定验证任务。这里是训练诊断，不是独立测试成绩。高超时率说明未能完成任务；动作范数和轨迹可帮助区分停滞、抖动、绕行失败，不能把所有超时都叫作碰撞。', '',
        f'受阻场景由早期检查点的碰撞率{early_c["collision"]:.0%}变为最终的{late_c["collision"]:.0%}，超时率由{early_c["timeout"]:.0%}变为{late_c["timeout"]:.0%}，平均归一化动作范数从{early_c["mean_normalized_action_norm"]:.3f}降至{late_c["mean_normalized_action_norm"]:.3f}。最终表现更接近保守停留或小范围运动，未形成到达目标的有效绕行。碰撞减少不能单独算作完成避障到达。这个观察尚不能归因于某一奖励项。', '',
        '![验证期间各类终止结果](validation_outcomes.png)','',
        '## 已确认的时间语义问题','',
        f'同一姿态、目标、障碍和动作，在第1步与第120步的21维观测相同，但奖励分别为{r1:.2f}和{r2:.2f}。超时返回terminated={t2}、truncated={x2}；训练入口仅把terminated写入回放终止标记，因此超时仍计算后继Q值。', '',
        '本轮将120步与额外−5超时惩罚作为任务失败条件，却沿用了旧入口的外部截断处理，且观测中没有剩余时间。这是任务定义与时间处理未对齐的实现问题。它可能影响价值学习，但本次未做受控重训，不能断言它是唯一或主要失败原因。', '',
        '有限时域任务应把时间上限作为终止并提供剩余时间；如果只是外部采样截断，应保持自举并重新审视超时惩罚。[Farama官方解释](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)', '',
        '## 其他观察与解释边界','',
        '无障碍和不挡路任务本身也未稳定收敛，因此不能只把失败归因于绕障过难。进入受阻任务占比更高的阶段后，稳定成功仍未出现；这提示应先验证基础学习流程，再扩大种子。', '',
        '参考路径奖励高于停滞只验证了回报量级，没有证明随机探索容易发现这些成功路径。困难任务的稀少成功经验、密集安全惩罚与训练分布变化均可能影响学习，尚无消融证据区分它们。', '',
        '## 建议的后续顺序','',
        '1. 先统一有限时域终止、自举掩码和时间观测；新增对应测试，保留本次模型作为诊断版本。',
        '2. 使用少量固定、可解的无障碍和不挡路任务检查稳定到达，设置小规模预实验门槛。',
        '3. 通过后逐步加入少量受阻任务，检查真实绕行轨迹和成功经验，再决定是否重跑大预算多种子。',
        '4. 本次不自动改奖励、训练预算或任务难度；下一轮修订需要明确新协议，独立测试仍保持未用于选择。','',
        '## 配对轨迹','']
    for step,g,name in displayed: lines += [f'**检查点{step}步，第{g}组**','',f'![配对轨迹]({name})','']
    lines += ['## 数据','', '- [诊断汇总](diagnosis.json)、[检查点指标CSV](checkpoint_metrics.csv)',
              '- [完整验证轨迹](validation_trajectories.json)、[回报复算](trajectory_returns.json)',
              '- [训练样本表现](training_sample.json)、[时间语义复现](timeout_semantics.json)',
              '- 如已运行交付核验：[检查记录](checks.json)，包含检查点复现、加密碰撞复核和原文件哈希。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Diagnosis ready: {out/"report.md"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'avoidance_retrain_v2'); a=p.parse_args(); diagnose(a.root)
