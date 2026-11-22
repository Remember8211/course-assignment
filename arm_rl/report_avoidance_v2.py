"""Derived reports only; never changes tasks, policies or raw trajectories."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import avoidance_report as plots


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def build_report(root):
    root=Path(root); out=root/'results'; protocol=read(root/'protocol.json')
    episodes=read(out/'episodes.json'); diagnostic=read(out/'diagnostic_episodes.json')
    tasks={t['task_id']:t for t in read(root/'data/test/tasks.json')['tasks']}
    checks=read(out/'checks.json'); refinement=read(out/'refinement.json')
    selections={f'sac_v2_s{s}':read(root/'runs'/f'sac_v2_s{s}'/'selected.json') for s in (0,1,2)}
    for s in (0,1,2): plots.METHOD_LABELS[f'sac_v2_s{s}']=f'新SAC 种子{s}'
    plots._setup_plotting()
    summary=plots.summarize(episodes); diag_summary=plots.summarize(diagnostic); paired=plots.paired_comparisons(episodes)
    plots._csv(out/'summary.csv',summary); plots._csv(out/'diagnostic_summary.csv',diag_summary); plots._csv(out/'paired.csv',paired)
    plots._csv(out/'episodes.csv',[{k:v for k,v in e.items() if k!='trajectory'} for e in episodes])
    seed_stats=[]
    for tolerance in (.02,.08):
        for scenario in 'ABC':
            rows=[r for r in summary if r['tolerance_m']==tolerance and r['scenario']==scenario and r['method'].startswith('sac')]
            values=[r['safe_success_rate'] for r in rows]
            seed_stats.append({'tolerance':tolerance,'scenario':scenario,'seeds':3,
              'mean_safe_success_rate':float(np.mean(values)),'std_between_seeds':float(np.std(values,ddof=1)),
              'per_seed_rates':values})
    (out/'seed_statistics.json').write_text(json.dumps(seed_stats,ensure_ascii=False,indent=2),encoding='utf-8')
    figures=out/'figures'; figures.mkdir(exist_ok=True)
    methods=[f'sac_v2_s{s}' for s in (0,1,2)]+['dls_noavoid','dls_avoid']
    colors=['#235789','#57a0c5','#8c6bb1','#bc8045','#54895d']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,tolerance in zip(axes,(.02,.08)):
        for i,(method,color) in enumerate(zip(methods,colors)):
            rows=[next(r for r in summary if r['method']==method and r['scenario']==s and r['tolerance_m']==tolerance) for s in 'ABC']
            x=np.arange(3)+(i-2)*.15; values=[r['safe_success_rate'] for r in rows]
            ax.bar(x,values,.15,color=color,label=plots.METHOD_LABELS[method])
            for xi,v,r in zip(x,values,rows): ax.text(xi,v+.015,f"{r['safe_success']}/{r['n']}",ha='center',va='bottom',rotation=90,fontsize=7)
        ax.set_xticks(np.arange(3),[plots.SCENARIO_LABELS[s] for s in 'ABC']); ax.set_ylim(0,1.25)
        ax.set_title(f'到达容差 {tolerance:.2f} m'); ax.set_ylabel('安全到达率'); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False)
    fig.subplots_adjust(bottom=.18,top=.89,wspace=.22); fig.savefig(figures/'success_overview.png'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for s in (0,1,2):
        with (root/'runs'/f'sac_v2_s{s}'/'train_log.csv').open(encoding='utf-8') as f: log=list(csv.DictReader(f))
        for ax,scenario in zip(axes,'ABC'):
            ax.plot([int(r['step']) for r in log],[float(r[f'{scenario}_success']) for r in log],label=f'种子{s}')
            ax.set_title(f'验证集 {plots.SCENARIO_LABELS[scenario]}'); ax.set_ylim(0,1.05); ax.set_xlabel('环境交互步数'); ax.set_ylabel('2厘米安全成功率'); ax.grid(alpha=.2)
    axes[-1].legend(); fig.tight_layout(); fig.savefig(figures/'validation_curves.png'); plt.close(fig)
    fixed=[]; failures=[]
    for g in (1,50,100):
        for method in methods[:3]:
            for tolerance in (.02,.08):
                filename=f'fixed_g{g:03d}_{method}_tol{tolerance:.2f}.png'
                plots._triptych(figures/filename,g,method,tolerance,tasks,episodes)
                fixed.append((g,method,tolerance,filename))
    for method in methods:
        for tolerance in (.02,.08):
            bad=sorted([e for e in episodes if e['method']==method and e['tolerance']==tolerance and e['status']!='success'],key=lambda e:(e['group'],e['scenario']))
            if bad:
                e=bad[0]; filename=f'first_failure_g{e["group"]:03d}_{method}_tol{tolerance:.2f}.png'
                plots._triptych(figures/filename,e['group'],method,tolerance,tasks,episodes)
                failures.append((e,filename))
    lines=['# 重新训练SAC后的有限绕障验证','',
      '本实验从头训练三个21维完整几何观测SAC模型，采用固定任务库、课程安排、奖励与300000步预算。模型仅按验证集选择，独立测试集不参与训练或选择。', '',
      '## 主要结果','',
      '| 模型 | 所选训练步数 | A无障碍（2厘米） | B不挡路（2厘米） | C受阻（2厘米） | C受阻（8厘米） |',
      '|---|---:|---:|---:|---:|---:|']
    for method in methods[:3]:
        counts=[]
        for scenario,tol in [('A',.02),('B',.02),('C',.02),('C',.08)]:
            row=next(r for r in summary if r['method']==method and r['scenario']==scenario and r['tolerance_m']==tol)
            counts.append(f"{row['safe_success']}/{row['n']}")
        lines.append('| '+plots.METHOD_LABELS[method]+' | '+str(selections[method]['step'])+' | '+' | '.join(counts)+' |')
    c2=[r for r in summary if r['scenario']=='C' and r['tolerance_m']==.02 and r['method'].startswith('sac')]
    total=sum(r['safe_success'] for r in c2)
    lines+=['',f'三个种子在独立C类任务的2厘米安全成功回合合计为{total}/300；这包含同一批100个任务在三个模型上的重复评估，不能当成300个独立任务。']
    if total:
        lines+=['这些安全成功回合说明修订方案在部分直达受阻且可解的测试任务中表现出一定绕障能力。成功范围应结合逐种子成绩阅读，不能推广为任意环境的安全保证。']
    else:
        lines+=['本次仍未得到2厘米条件下安全绕障的成功证据，应如实报告失败；不能为了获得正面结论继续调整测试任务或训练预算。']
    lines+=['','新方案同时改变任务分布、观测与奖励，不能把变化归因于某一个因素，也不据此声称SAC优于经典方法。', '',
      '![独立测试结果](figures/success_overview.png)','',
      '## 可复现设置','',
      '机械臂长度为[0.4,0.3,0.3] m、连杆半径0.02 m，固定初始关节角[0.2,0.5,−0.6] rad，单步各关节最多变化0.05 rad，关节限幅±π，最多120步。主训练容差0.02 m，另以0.08 m评估。成功、碰撞、超时互斥，碰撞优先。', '',
      '训练/验证/测试分别1000/50/100组，每组包含同起点目标的A/B/C任务；种子分别20261111/20261112/20261113。目标由均匀随机关节姿态经正运动学及区域筛选生成，不是工作空间面积均匀分布。C类具备覆盖8厘米成功区域的直达阻挡和120步内至少1厘米间隙的参考关节路径。参考姿态及路径不输入策略、不作为模仿数据。', '',
      '完整21维观测包括归一化的9维机械臂/目标信息及三个障碍槽，每槽为相对圆心二维坐标、半径和有效标记。任务中最多两个障碍，剩余槽置零；训练包含无障碍任务。假设障碍几何已知，未模拟传感器。', '',
      '奖励为5倍归一化距离进度−0.01时间惩罚−0.001动作平方范数−0.1近障碍惩罚；近障碍项为max(0,(0.03−运动段最小表面间隙)/0.03)²，无障碍时为零。安全到达/碰撞/超时分别额外+20/−20/−5。', '',
      'SAC为两层128神经元，Adam学习率3e−4，折扣0.99，回放100000，批大小256，预热1000步，软更新0.005，自动熵温度、目标熵−3。CPU单线程，种子0/1/2分别控制初始化、探索、回放与任务抽样。', '',
      '课程按环境步数固定：1～50000步A/B各50%；50001～100000步A/B/C为20%/30%/50%；之后为10%/20%/70%。每10000步评估固定验证集；依次按C类成功率、B类成功率、C类误差、较早步数选择模型。', '',
      '![验证集训练曲线](figures/validation_curves.png)','',
      '## 检查与统计边界','',
      f'完成{len(episodes)}个独立测试回合及{len(diagnostic)}个旧诊断集回归回合。任务与原始受保护文件哈希检查通过：{checks["passed"]}。旧10组任务已公开，仅作回归诊断。', '',
      f'0.0005 rad加密复核发现测试漏碰撞回合{len(refinement["test"]["missed_collisions"])}个，诊断集{len(refinement["diagnostic"]["missed_collisions"])}个；如有漏判已统一细化重评。表面间隙扣除障碍及连杆半径，统计包括初始姿态和运动段。检查不涵盖自碰撞、动力学及执行器误差，不构成数学连续安全证明。', '',
      '每类测试100个任务，每个种子单独报告；跨种子标准差为样本标准差（ddof=1）。终点误差统计全部任务；路径和步数均值仅统计安全成功任务，并附成功样本数。无障碍间隙为不适用。', '',
      '| 容差/m | 场景 | 三种子平均成功率 | 种子间标准差 |','|---|---|---:|---:|']
    for r in seed_stats: lines.append(f'| {r["tolerance"]} | {r["scenario"]} | {r["mean_safe_success_rate"]:.3f} | {r["std_between_seeds"]:.3f} |')
    for tol in (.02,.08):
        lines+=['',f'## 独立测试详细结果：容差{tol:.2f} m','']
        lines+=plots._result_table([r for r in summary if r['tolerance_m']==tol])
    lines+=['','## 旧诊断集回归','', '此前公开的30个任务不参与训练和模型选择，以下结果仅作为回归检查，不能代替新的独立测试。','']
    for tol in (.02,.08): lines+=['',f'### 容差{tol:.2f} m','']+plots._result_table([r for r in diag_summary if r['tolerance_m']==tol])
    lines+=['','## 固定配对轨迹','', '以下固定展示第1、50、100组，不按结果挑选。灰虚线仅表示已核验参考路径的末端投影，参考关节轨迹保存在任务文件中。','']
    for g,m,t,f in fixed: lines += [f'**第{g}组，{plots.METHOD_LABELS[m]}，容差{t:.2f} m**','',f'![配对轨迹](figures/{f})','']
    lines+=['## 首个失败案例','', '每种方法和容差按组号、A/B/C顺序选取首个失败，保留全部结果。','']
    for e,f in failures: lines += [f'**{plots.METHOD_LABELS[e["method"]]}：第{e["group"]}组{e["scenario"]}，容差{e["tolerance"]:.2f} m**','',f'![失败案例](figures/{f})','']
    lines+=['## 复现与数据','', '在项目根目录运行；报告阶段只读取已保存结果，不重训：','',
      '```powershell', '$env:MPLCONFIGDIR = Join-Path (Get-Location) "arm_rl/avoidance_retrain_v2/.mplconfig"',
      'python -B arm_rl/retrain_avoidance_v2.py --stage report',
      '# 在全新目录重做完整协议：',
      'python -B arm_rl/retrain_avoidance_v2.py --out arm_rl/avoidance_retrain_v2_reproduction --stage all', '```','',
      '- [实验协议](../protocol.json)、[奖励预检](../reward_precheck.json)',
      '- [完整测试轨迹](episodes.json)、[诊断轨迹](diagnostic_episodes.json)',
      '- [逐回合CSV](episodes.csv)、[汇总CSV](summary.csv)、[配对CSV](paired.csv)',
      '- [加密复核](refinement.json)、[完整性检查](checks.json)、[三种子统计](seed_statistics.json)',
      '- 任务、规划记录在上一级data目录，训练日志、验证轨迹、检查点及所选模型在runs目录。','',
      '论文修订应报告实际成绩与失败条件，明确已知几何、固定起点、静态圆障碍和有限运动学模型等限制。本文档提供修订依据，不自动修改原论文。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Report ready: {out/"report.md"}',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args(); build_report(a.root)
