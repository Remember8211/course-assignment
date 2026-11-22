# SAC有限绕障重新训练实验

本目录保存版本化协议的任务库、奖励预检、种子0训练记录、模型与验证集诊断；独立最终评估尚未完成。原13维模型及原论文不覆盖。

## 当前状态

用户在训练过程中选择先诊断当前种子，再决定另外两个。种子0已完成300000步，用时约51.9分钟，按验证规则选中50000步检查点。种子1和2尚未开始训练，独立测试集尚未运行控制器评估。完整三种子协议暂缓，当前诊断交付见 `diagnosis_seed0/report.md`；不要将本目录当作已完成的三种子实验。

已复现的时间处理问题和建议记录在诊断报告中。本次仅诊断，没有自动修复训练逻辑或修改奖励，也没有继续训练。

## 运行方式

在项目根目录执行：

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'arm_rl/avoidance_retrain_v2/.mplconfig'
python -B arm_rl/retrain_avoidance_v2.py --stage prepare
python -B -m unittest discover -s arm_rl -p 'test_avoidance*.py' -v
python -B arm_rl/retrain_avoidance_v2.py --stage train
python -B arm_rl/retrain_avoidance_v2.py --stage evaluate
python -B arm_rl/retrain_avoidance_v2.py --stage report
```

完整重做应选择全新输出目录：

```powershell
python -B arm_rl/retrain_avoidance_v2.py --out arm_rl/avoidance_retrain_v2_reproduction --stage all
```

已完成的种子不会重复训练；存在未完成日志时脚本停止，避免静默覆盖。每个检查点保存模型、优化器/目标网络/温度状态与验证结果，但未保存回放池和全部随机数状态，因此不能声称支持精确中途续训。

## 固定约定

- 数据：训练/验证/测试1000/50/100组，每组A无障碍、B障碍不挡路、C直达受阻，种子20261111/20261112/20261113。
- 模型：21维已知圆几何观测、三维动作，两层128神经元SAC；训练种子0/1/2，每个300000环境步，不输入参考路径。
- 训练课程：前50000步A/B各50%，随后50000步A/B/C为20%/30%/50%，其余为10%/20%/70%。类别在回合重置时按当前步数采样，跨阶段回合继续至终止。
- 选择：每10000步在固定验证集评估，以C安全成功率、B安全成功率、C误差、较早步数依次排序；测试集不参与选择。
- 几何：训练/评估动作按关节最大子步0.001 rad检查，测试轨迹以0.0005 rad复核；仅外部障碍，无动力学及自碰撞。
- 结论：比较修订方案整体的可行性，不把改善归因于某一单独改动；保留全部失败，不根据测试结果延长预算或调整任务。

## 产物

- `protocol.json`：冻结参数、任务哈希及受保护的原始文件哈希。
- `data/*/tasks.json`：完整任务几何、参考姿态和关节路径；`generation_log.json`记录筛选。
- `reward_precheck.json`：全部1000条训练C参考路径与停滞/无避障控制的回报比较。
- `runs/sac_v2_s*/`：训练日志、进度、每10000步检查点和验证轨迹，`selected.json`记录选择理由，`model.pt`是所选模型。
- `results/report.md`：训练完成后生成的中文实验报告；同目录包含原始轨迹、CSV、复核记录及图表。

## 发布副本说明

上面的完整协议命令供未来另行运行，不代表已全部执行。请优先阅读上级README中的分阶段命令，使用全新输出目录。原论文不随代码分发，发布入口将其设为明确报告的可选校验，其他受保护资料仍必须匹配；历史协议与报告不作回写。
