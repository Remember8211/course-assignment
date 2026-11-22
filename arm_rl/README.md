# SAC机械臂到达控制与避障能力评估

这是工业人工智能课程的平面三连杆机械臂实验，包含原13维SAC实验、固定配对避障验证、21维修订训练及单种子诊断。环境是NumPy运动学模型，不依赖ROS、Gazebo或PyBullet。它不包含动力学、执行器误差或自碰撞。

## 三个阶段与已有证据

| 阶段 | 材料入口 | 状态与结论边界 |
|---|---|---|
| 原SAC随机任务实验 | [原汇总](results_eval/summary.csv)、[主模型种子0](runs/sac_s0/)、[种子1](runs/sac_s1/) | 两个主种子各15万步；两项消融各10万步、一个种子。平均到达标记比例96.5%/80.5%，未见布局平均步数21.24。 |
| 固定配对避障验证 | [报告](avoidance_validation_20261107/report.md)、[逐回合CSV](avoidance_validation_20261107/episodes.csv)、[完整轨迹](avoidance_validation_20261107/episodes.json) | 10组A/B/C、30个固定任务、240回合；参考路径已验证可行。8厘米容差下两个SAC在B类均10/10安全成功，C类均0/10。 |
| 修订训练及诊断 | [状态说明](avoidance_retrain_v2/README.md)、[诊断报告](avoidance_retrain_v2/diagnosis_seed0/report.md)、[模型选择](avoidance_retrain_v2/runs/sac_v2_s0/selected.json) | 仅种子0完成30万步；种子1、2未训练，独立测试未评估。最终检查点C类50个验证任务全部超时。 |

原评估的`success`只判断末端距离，可与碰撞重叠；只检查动作终点姿态。原间隙未扣除连杆半径，原随机评估只保留汇总，不能补算严格安全成功率。配对验证使用运动段检查和互斥的安全成功/碰撞/超时，不能与原百分数直接相减。

C类失败同时包含碰撞和超时：8厘米容差下种子0为5次碰撞、5次超时，种子1为8次碰撞、2次超时。无障碍A类还改变了四维障碍输入的分布，不能仅凭A类成绩认定基础到达能力。去障碍观测消融和稀疏奖励消融的种子数、预算有限，不能证明障碍信息可省略或密集奖励必要。

21维版本尚有已确认的时间处理不一致：120步有额外超时惩罚、观测无剩余时间、截断仍自举。发布保留这一历史实现，不把诊断解释为已验证的因果结论，也不把碰撞减少而全部超时称为成功避障。

## 环境与安装

历史记录环境为Windows、Python 3.11.9、NumPy 2.4.6、Matplotlib 3.11.1、PyTorch 2.13.0+cu126；配对验证及v2训练使用CPU单线程。默认安装不要求CUDA，跨平台数值一致性未经本次发布测试。

在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r arm_rl/requirements.txt
```

下文`python`指已安装这些依赖的解释器；Linux/macOS可使用`.venv/bin/python`。字体缺失可能影响中文图显示，可安装宋体或兼容中文字体。

## 只读取现有结果

直接打开上述CSV、JSON和Markdown即可阅读，不需要执行模型。逐文件SHA-256见仓库根目录`MANIFEST.sha256`，校验清单自身不列入清单。

下面命令只从保存的任务和回合记录重新生成配对报告与图表，不运行控制器。它会重写该结果目录的报告、汇总和图片；需要保留工作副本时，请先复制整个结果目录再对副本运行。

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'reproduction/sac_plot_cache'
python -B arm_rl/avoidance_report.py arm_rl/avoidance_validation_20261107
```

`figures/paper_revision/`保存纠错版论文的五幅图，来自现有日志及第1、5、10组配对记录。上层六幅`f*.png`为历史原图，历史图中的“success”应按原到达标记解释；多任务轨迹叠加图中的障碍也来自多个不同任务。

## 另行重新评估或训练

以下是供读者另行复现的命令，**本次发布没有执行**。所有输出使用新的`reproduction`目录。

原13维模型评估会重新构造固定种子任务并执行策略：

```powershell
python -B arm_rl/eval.py --runs-root arm_rl/runs --out reproduction/sac_original_eval --tags sac_s0 sac_s1 sac_sparse_s0 sac_noobsfeat_s0
```

重新做配对验证会生成任务、执行240回合并生成报告：

```powershell
python -B arm_rl/validate_avoidance.py --out reproduction/sac_avoidance_validation --stage all
```

从头训练原主模型（种子1将`--seed`改为1；切勿覆盖已保存目录）：

```powershell
python -B arm_rl/train.py --algo sac --seed 0 --steps 150000 --out-root reproduction/sac_runs
```

重做21维方案请使用全新输出目录，以下只指定种子0；`prepare`包含任务生成和奖励行为检查，`train`还会周期执行验证策略：

```powershell
python -B arm_rl/retrain_avoidance_v2.py --out reproduction/sac_v2 --stage prepare
python -B arm_rl/retrain_avoidance_v2.py --out reproduction/sac_v2 --stage train --seed 0
```

默认不执行`--stage all`，因为该入口会继续训练三个种子并进行独立测试。现有发布不包含三种子最终测试报告；`report_avoidance_v2.py`需要未来完整评估的结果文件，不能对当前未完成状态直接运行。`diagnose_avoidance_v2.py`会执行模型和环境，属于重新诊断，不是只读取记录。历史`make_figs.py`也会执行策略，且按旧口径作图。

## 文件组织与发布适配

- `env3link.py`、`sac.py`、`train.py`、`eval.py`：原环境、算法和训练评估。`common.py`依赖`ppo.py`，因此保留PPO实现，但未发布PPO调参结果，也未完成同预算算法比较。
- `validate_avoidance.py`、`avoidance_report.py`：几何、参考路径、运动段检查及只读取记录的报告入口。
- `avoidance_v2.py`、`retrain_avoidance_v2.py`、`diagnose_avoidance_v2.py`：修订环境、训练和诊断；未修复历史时间语义。
- `test_*.py`：已有环境与几何检查代码，运行它们会推进环境，不属于本次发布所执行的检查。
- `historical_source/`：两份因发布适配而调整的脚本原始字节快照；不是新的运行入口。
- `artifact_integrity.py`：只处理历史相对路径和文件哈希，不依赖模型或环境。缺少不随代码发布的原论文时明确报告未校验；缺少代码/模型/任务或哈希不符时仍报错。原论文存在时也必须匹配。未完整校验的总标记为`null`，不伪报全部通过。
- [发布来源清单](PUBLICATION.json)：逐文件来源、历史哈希及适配说明。旧报告、协议、模型和固定任务保持原始字节；其历史校验结果不冒充本次重新实验的结果。

`.gitattributes`对本目录禁用换行转换，确保克隆后任务、模型和源码仍可匹配历史SHA-256。缓存、虚拟环境、无关试跑、重复粗检查轨迹副本、论文正文和期刊模板不随本目录发布。检查点虽保存优化器等状态，但缺少回放池和完整随机数状态，不能声称支持精确中途续训。
