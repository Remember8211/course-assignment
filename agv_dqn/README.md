# DQN栅格导航实验

本目录保存论文对应的简化栅格实验。机器人在四邻域中选择上、下、左、右动作，静态观测为自身坐标、目标坐标和八个相邻格的占据标记，共12维。动态版本额外输入两个移动障碍的相对坐标，共16维，未限制其探测距离。

## 环境与安装

本次核验使用Windows、Python 3.11.9、NumPy 2.4.6、Gymnasium 1.4.0、Matplotlib 3.11.1和PyTorch 2.13.0（本机安装为cu126构建）。以下模型重放核验采用CPU、单线程。

在仓库根目录执行以下命令。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r agv_dqn/requirements.txt
cd agv_dqn
..\.venv\Scripts\python verify_results.py --output ../reproduction/verification.json
```

Linux/macOS将解释器路径换为`.venv/bin/python`及`../.venv/bin/python`。后文用`python`表示已安装上述依赖的解释器。

## 文件与实验记录

| 文件或目录 | 内容 |
|---|---|
| `env_grid.py` | 状态、观测、移动障碍、动作及奖励 |
| `dqn.py` | 价值网络、经验回放、目标网络及训练更新 |
| `astar.py`、`tabular_q.py` | A*参照和表格Q-learning |
| `train.py` | DQN训练入口 |
| `evaluate.py` | 主实验路径、学习曲线和策略图 |
| `generalization_eval.py` | 十张未见地图评估 |
| `dynamic_eval.py` | 静态/动态训练与静态/动态环境的配对评估 |
| `verify_results.py` | CPU确定性重放、日志一致性及受阻动作核验 |
| `results/main`、`results/tabular` | 静态主实验及表格Q-learning结果 |
| `results/dyn_static_ctrl`、`results/dyn` | 16维静态对照及动态训练结果 |
| `results/generalization`、`results/dynamic` | 原始评估汇总及图表 |

主实验训练400回合，动态实验600回合，均为种子0。两个DQN隐藏层各128单元，动作输出维数4，批大小64，回放池50000，目标网络每200次学习更新同步。

## 结果重放与重新出图

以下命令在`agv_dqn`目录执行。输出写入`../reproduction/`，保留随仓库发布的原始结果。

```powershell
python verify_results.py --output ../reproduction/verification.json
python generalization_eval.py --out ../reproduction/generalization
python dynamic_eval.py --episodes 30 --out ../reproduction/dynamic
```

主实验出图入口会在指定运行目录中写入图片，因此先复制主实验到独立目录。

```powershell
python -c "import shutil; shutil.copytree('results/main', '../reproduction/main', dirs_exist_ok=True)"
python evaluate.py --run ../reproduction/main --compare results/tabular --labels "Tabular Q-learning"
```

`verify_results.py`显式使用CPU；原始评估入口会在可用时自动选择CUDA。平台、依赖和设备差异可能影响数值或训练结果，核验脚本会报告与保存成绩的差异。

重新出图时，动态连续帧使用实际交互步号，并将18步虚线标为静态底图下界。`results/dynamic/`中的历史图片保留了当时的标注；本文所用修订图按上述口径解释。

## 从头训练

以下为保存配置对应的训练命令，输出到独立目录。从头训练的结果仍需单独检查。

```powershell
python train.py --episodes 400 --seed 0 --map-seed 0 --out-root ../reproduction/training --tag main
python tabular_q.py --episodes 400 --seed 0 --map-seed 0 --out-root ../reproduction/training --tag tabular
python train.py --episodes 400 --seed 0 --map-seed 0 --movers 0 --mover-slots 2 --out-root ../reproduction/training --tag dyn_static_ctrl
python train.py --episodes 600 --seed 0 --map-seed 0 --movers 2 --mover-slots 2 --out-root ../reproduction/training --tag dyn
```

历史汇总中的`train_seconds`是当时运行记录，未保存完整设备信息。当前环境的重放核验不验证历史训练耗时，也不代表重新执行了完整训练。

## 指标与结论边界

`final_success_rate`对应训练最后50回合的到达比例，`final_greedy_success`为固定任务上的最终贪心评估。未见地图的80%为8/10，每张地图一次评估，起终点保持不变。

`path_length`只累计有效移动，`steps`包含受阻后原地不动的动作。机器人尝试进入占据格会被阻止并受罚，回合仍可继续到达。核验中，静态训练模型在动态环境有1个回合出现1次受阻；动态训练模型30回合未出现受阻。

两个未见地图失败样本表现为局部循环。静态底图的18步最短路径仅提供动态实验中的空间长度下界。单种子结果、不同训练回合数以及静态对照在测试时出现非零移动障碍槽位，限制了能力归因。尚无去除局部占据观测的DQN消融结果。
