"""Read-only analysis and figures for the independent avoidance validation.

This module never selects tasks or changes a controller. ``build_report`` reads
the frozen validation artifacts and writes derived CSVs, figures and a report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np

from env3link import LINK_LEN, LINK_RADIUS, fk


METHOD_LABELS = {
    "sac_s0": "SAC 种子0",
    "sac_s1": "SAC 种子1",
    "dls_noavoid": "DLS 无避障项",
    "dls_avoid": "DLS 有避障项",
    "dls_no_avoid": "DLS 无避障项",
    "dls": "DLS 有避障项",
}
SCENARIO_LABELS = {"A": "A 无障碍", "B": "B 障碍不挡路", "C": "C 直达受阻"}
STATUS_LABELS = {
    "success": "安全到达",
    "safe_success": "安全到达",
    "collision": "碰撞",
    "timeout": "超时",
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values):
    clean = [v for x in values if (v := _finite(x)) is not None]
    return float(np.mean(clean)) if clean else None


def _fmt(value, digits=3):
    value = _finite(value)
    return "—" if value is None else f"{value:.{digits}f}"


def _csv(path: Path, rows: list[dict], fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _success(episode):
    return episode["status"] in ("success", "safe_success")


def summarize(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["tolerance"], episode["scenario"], episode["method"])].append(episode)
    result = []
    for (tol, scenario, method), items in sorted(groups.items()):
        successes = [e for e in items if _success(e)]
        gaps = [v for e in items if (v := _finite(e.get("min_clearance"))) is not None]
        result.append({
            "tolerance_m": tol,
            "scenario": scenario,
            "method": method,
            "n": len(items),
            "safe_success": len(successes),
            "collision": sum(e["status"] == "collision" for e in items),
            "timeout": sum(e["status"] == "timeout" for e in items),
            "safe_success_rate": len(successes) / len(items),
            "mean_final_error_all_m": _mean(e["final_error"] for e in items),
            "mean_steps_success": _mean(e["steps"] for e in successes),
            "mean_path_success_m": _mean(e["path_length"] for e in successes),
            "mean_min_clearance_success_m": _mean(e.get("min_clearance") for e in successes),
            "min_clearance_all_m": min(gaps) if gaps else None,
        })
    return result


def _trajectory(episode):
    """Return the complete stored joint-state trajectory, including q0."""
    return np.asarray(episode["trajectory"]["q"], dtype=float)


def _ee_path(episode):
    return np.asarray([fk(q)[-1] for q in _trajectory(episode)])


def _rms_path_difference(a, b):
    pa, pb = _ee_path(a), _ee_path(b)
    n = min(len(pa), len(pb))
    return float(np.sqrt(np.mean(np.sum((pa[:n] - pb[:n]) ** 2, axis=1))))


def paired_comparisons(episodes):
    groups = defaultdict(dict)
    for e in episodes:
        groups[(e["group"], e["method"], e["tolerance"])][e["scenario"]] = e
    rows = []
    for (group, method, tol), cases in sorted(groups.items()):
        if not all(s in cases for s in "ABC"):
            continue
        a, b, c = (cases[s] for s in "ABC")
        rows.append({
            "group": group, "method": method, "tolerance_m": tol,
            "A_status": a["status"], "B_status": b["status"], "C_status": c["status"],
            "A_C_ee_rms_common_prefix_m": _rms_path_difference(a, c),
            "B_C_ee_rms_common_prefix_m": _rms_path_difference(b, c),
            "C_final_error_m": c["final_error"],
            "C_min_clearance_m": c.get("min_clearance"),
        })
    return rows


def _setup_plotting():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 180,
    })


def _arm(ax, q, color, alpha=1.0, zorder=3):
    """Draw actual-radius link capsules instead of arbitrary-width line segments."""
    pts = fk(q)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        v = p1 - p0
        normal = LINK_RADIUS * np.array([-v[1], v[0]]) / np.linalg.norm(v)
        polygon = [p0 + normal, p1 + normal, p1 - normal, p0 - normal]
        ax.add_patch(Polygon(polygon, facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))
    for p in pts:
        ax.add_patch(Circle(p, LINK_RADIUS, facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))


def _draw_case(ax, task, episode):
    goal = np.asarray(task["goal"], dtype=float)
    radius = float(task.get("obstacle_radius", 0.08))
    for center in task["obstacles"]:
        ax.add_patch(Circle(center, radius, facecolor="#626970", edgecolor="#383d42", alpha=0.65, zorder=2))
    ax.add_patch(Circle(goal, episode["tolerance"], facecolor="#62b58c", edgecolor="#248359", ls="--", lw=1.0, alpha=0.25, zorder=2))
    reference = task.get("reference_q")
    if reference is not None and episode["scenario"] == "C":
        rp = np.asarray([fk(q)[-1] for q in reference])
        ax.plot(rp[:, 0], rp[:, 1], "--", color="#8c8c8c", lw=1.0, zorder=2)
    qs = _trajectory(episode)
    indices = sorted(set([0, (len(qs) - 1) // 3, 2 * (len(qs) - 1) // 3, len(qs) - 1]))
    for idx in indices:
        final = idx == len(qs) - 1
        color = "#ce563f" if final else "#507ca6"
        _arm(ax, qs[idx], color, alpha=0.90 if final else 0.27, zorder=4 if final else 3)
    path = _ee_path(episode)
    ax.plot(path[:, 0], path[:, 1], color="#145d91", lw=1.6, zorder=5)
    ax.scatter(path[0, 0], path[0, 1], s=30, facecolor="white", edgecolor="#145d91", zorder=6)
    ax.scatter(goal[0], goal[1], s=75, marker="*", color="#218b52", edgecolor="#124f30", linewidth=0.5, zorder=7)
    if episode["status"] == "collision":
        ax.scatter(path[-1, 0], path[-1, 1], marker="x", s=48, color="#b52b2b", linewidth=1.8, zorder=7)
    ax.scatter([0], [0], marker="s", s=28, color="#273342", zorder=6)
    gap = _fmt(episode.get("min_clearance"), 3)
    ax.set_title(f"{SCENARIO_LABELS.get(episode['scenario'], episode['scenario'])} · {STATUS_LABELS.get(episode['status'], episode['status'])}\n"
                 f"误差 {episode['final_error']:.3f} m；步数 {episode['steps']}；最小间隙 {gap} m")
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), xlabel="x / m", ylabel="y / m")
    ax.set_aspect("equal")
    ax.grid(alpha=0.14, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _triptych(path, group, method, tol, tasks, episodes):
    by_case = {e["scenario"]: e for e in episodes if e["group"] == group and e["method"] == method and abs(e["tolerance"] - tol) < 1e-10}
    if not all(s in by_case for s in "ABC"):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    for ax, scenario in zip(axes, "ABC"):
        e = by_case[scenario]
        _draw_case(ax, tasks[e["task_id"]], e)
    fig.suptitle(f"任务组 {group:02d} · {METHOD_LABELS.get(method, method)} · 到达容差 {tol:.2f} m", fontsize=13, y=0.98)
    handles = [
        Line2D([], [], color="#145d91", lw=1.6, label="实际末端轨迹"),
        Line2D([], [], color="#507ca6", lw=5, alpha=.4, label="起始/中间姿态"),
        Line2D([], [], color="#ce563f", lw=5, label="结束姿态"),
        Line2D([], [], color="#8c8c8c", ls="--", label="C类可行参考末端轨迹"),
        Line2D([], [], marker="*", ls="none", color="#218b52", markersize=9, label="目标及成功圆"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.055, right=.99, top=.81, bottom=.17, wspace=.25)
    fig.savefig(path)
    plt.close(fig)
    return True


def _overview(path, summary):
    tolerances = sorted(set(row["tolerance_m"] for row in summary), reverse=True)
    methods = sorted(set(row["method"] for row in summary), key=lambda m: (not m.startswith("sac"), m))
    fig, axes = plt.subplots(1, len(tolerances), figsize=(12, 4.5), squeeze=False)
    colors = ["#245c8d", "#5d92bc", "#b97842", "#518968"]
    width = .18
    for ax, tol in zip(axes[0], tolerances):
        for i, method in enumerate(methods):
            x = np.arange(3) + (i - (len(methods) - 1) / 2) * width
            rows = [next((r for r in summary if r["method"] == method and r["scenario"] == s and r["tolerance_m"] == tol), None) for s in "ABC"]
            vals = [r["safe_success_rate"] if r else 0 for r in rows]
            ax.bar(x, vals, width, color=colors[i % len(colors)], label=METHOD_LABELS.get(method, method))
            for xi, value, row in zip(x, vals, rows):
                if row:
                    ax.text(xi, value + .025, f"{row['safe_success']}/{row['n']}",
                            ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(np.arange(3), [SCENARIO_LABELS[s] for s in "ABC"])
        ax.set(ylim=(0, 1.15), ylabel="安全到达率", title=f"到达容差 {tol:.2f} m")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.19, top=.88, left=.06, right=.99, wspace=.20)
    fig.savefig(path)
    plt.close(fig)


def _result_table(rows):
    lines = [
        "| 场景 | 方法 | 安全到达 | 碰撞 | 超时 | 平均终点误差/m | 成功回合平均步数 | 成功回合平均路径/m |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {name: i for i, name in enumerate(("sac_s0", "sac_s1", "dls_noavoid", "dls_avoid"))}
    for row in sorted(rows, key=lambda r: (r["scenario"], order.get(r["method"], 99))):
        lines.append(
            f"| {SCENARIO_LABELS.get(row['scenario'], row['scenario'])} "
            f"| {METHOD_LABELS.get(row['method'], row['method'])} "
            f"| {row['safe_success']}/{row['n']} | {row['collision']} | {row['timeout']} "
            f"| {_fmt(row['mean_final_error_all_m'])} | {_fmt(row['mean_steps_success'], 1)} "
            f"| {_fmt(row['mean_path_success_m'])} |"
        )
    return lines


def _conclusion_lines(summary, paired):
    lines = []
    c_rows = [r for r in summary if r["scenario"] == "C" and r["method"].startswith("sac")]
    for tol in sorted({r["tolerance_m"] for r in c_rows}, reverse=True):
        selected = sorted([r for r in c_rows if r["tolerance_m"] == tol], key=lambda r: r["method"])
        wording = "；".join(f"{METHOD_LABELS.get(r['method'], r['method'])} 为 {r['safe_success']}/{r['n']}" for r in selected)
        n_success = sum(r["safe_success"] for r in selected)
        if n_success:
            lines.append(f"在 {tol:.2f} m 容差的 C 类直达受阻任务中，{wording}。这些安全成功回合提供了现有策略在相应任务中绕过阻挡的有限证据。该结论针对本次构造任务，不代表任意障碍环境中的成功保证。")
        else:
            lines.append(f"在 {tol:.2f} m 容差的 C 类直达受阻任务中，{wording}。本次评估未获得该容差下 SAC 安全绕障到达的成功案例，不能据此声称现有模型已通过这组主动绕障验证；也不能由此推断所有其他任务都无法绕障。")
    for method in sorted({r["method"] for r in c_rows}):
        pair8 = [p for p in paired if p["method"] == method and abs(p["tolerance_m"] - .08) < 1e-10]
        if pair8:
            b_ok = sum(p["B_status"] in ("success", "safe_success") for p in pair8)
            c_ok = sum(p["C_status"] in ("success", "safe_success") for p in pair8)
            b_ok_c_failed = sum(p["B_status"] in ("success", "safe_success") and p["C_status"] not in ("success", "safe_success") for p in pair8)
            lines.append(f"主要配对证据：{METHOD_LABELS.get(method, method)} 在 0.08 m 容差下的 B 类成功 {b_ok}/{len(pair8)}、C 类成功 {c_ok}/{len(pair8)}；其中 {b_ok_c_failed} 组从障碍不挡路时成功，变为直达受阻后失败。每组起点、目标和障碍数量相同，仅移动一个障碍，因此 B→C 是本次判断阻挡影响的主要对照。原始 A→C 结果仍保存在配对数据中。")
    lines.append("A 类无障碍测试需要单独解读：原观测构造会把四维障碍特征全部置零，而现有模型主要在含两个障碍的环境中训练；移除障碍也改变了观测分布。A 类成绩较低可能与这种分布变化有关，这一解释尚未经因果验证，不能仅凭 A 类失败认定模型缺乏基础到达能力。B 类保留了两个障碍，更适合作为本次配对到达参照。")
    dls_c = [r for r in summary if r["scenario"] == "C" and r["method"].startswith("dls")]
    if dls_c and all(r["safe_success"] == 0 for r in dls_c):
        lines.append("DLS有、无避障项两个版本也均未通过本次C类任务。每个任务已有满足步数及碰撞检查条件的参考路径，因而这些失败不能简单解释为任务无解；结论限于所测控制器及构造场景，不能推广为SAC算法普遍无法避障。")
    lines.append("0.02 m 测试同时提高了到达精度要求。现有模型原本以 0.08 m 容差训练，因此严格测试失败须结合碰撞、超时和最终误差分析，不能全部归因于避障能力。")
    lines.append("A/B/C 轨迹随布局发生变化可补充说明障碍适应，但路径变化本身不等于正确避障。无避障 DLS 在 C 类的表现是对照事实，不是筛选任务的条件；任务和参考路径均在控制器评估前固定。")
    return lines


def build_report(out_dir):
    """Create all derived presentation artifacts from the frozen result files.

    Required schema: tasks.json wraps a ``tasks`` list; episodes.json is the
    per-episode list and each episode stores ``trajectory.q``. File references
    in report.md are relative to its own output directory.
    """
    out_dir = Path(out_dir)
    config = _read(out_dir / "config.json")
    task_data = _read(out_dir / "tasks.json")
    generation = _read(out_dir / "generation_log.json")
    episodes = _read(out_dir / "episodes.json")
    refinement = _read(out_dir / "refinement.json")
    checks = _read(out_dir / "checks.json")
    tasks = {task["task_id"]: task for task in task_data["tasks"]}
    groups = sorted({task["group"] for task in tasks.values()})

    for e in episodes:
        if e["task_id"] not in tasks:
            raise ValueError(f"Episode references absent task: {e['task_id']}")
        if e["status"] not in ("success", "safe_success", "collision", "timeout"):
            raise ValueError(f"Unexpected terminal status: {e['status']}")
    summary = summarize(episodes)
    paired = paired_comparisons(episodes)
    flat = [{
        "task_id": e["task_id"], "group": e["group"], "scenario": e["scenario"],
        "method": e["method"], "tolerance_m": e["tolerance"], "status": e["status"],
        "safe_success": int(_success(e)), "collision": int(e["status"] == "collision"),
        "timeout": int(e["status"] == "timeout"), "steps": e["steps"],
        "final_error_m": e["final_error"], "path_length_m": e["path_length"],
        "min_surface_clearance_m": _finite(e.get("min_clearance")),
    } for e in episodes]
    _csv(out_dir / "episodes.csv", flat)
    _csv(out_dir / "summary.csv", summary)
    _csv(out_dir / "paired.csv", paired)

    _setup_plotting()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fixed_figures = []
    tolerances = sorted({e["tolerance"] for e in episodes}, reverse=True)
    methods = sorted({e["method"] for e in episodes})
    if summary:
        _overview(fig_dir / "success_overview.png", summary)
    for group in (1, 5, 10):
        if group not in groups:
            continue
        for method in ("sac_s0", "sac_s1"):
            if method not in methods:
                continue
            for tol in tolerances:
                filename = f"fixed_g{group:02d}_{method}_tol{tol:.2f}.png"
                if _triptych(fig_dir / filename, group, method, tol, tasks, episodes):
                    fixed_figures.append((group, method, tol, filename))
    failure_figures = []
    for method in methods:
        for tol in tolerances:
            failures = sorted((e for e in episodes if e["method"] == method and e["tolerance"] == tol and not _success(e)), key=lambda e: (e["group"], e["scenario"]))
            if not failures:
                continue
            failure = failures[0]
            group = failure["group"]
            filename = f"first_failure_g{group:02d}_{method}_tol{tol:.2f}.png"
            if _triptych(fig_dir / filename, group, method, tol, tasks, episodes):
                failure_figures.append((group, method, tol, failure["scenario"], filename))

    passed = checks.get("passed") is True
    expected_episodes = len(tasks) * 4 * 2
    missing_groups = [group for group in (1, 5, 10) if group not in groups]
    lines = [
        "# 现有SAC机械臂模型的有限绕障验证",
        "",
        f"本报告评估已有两个 SAC 模型在配对障碍场景中的安全到达表现。共固定 {len(groups)} 组、{len(tasks)} 个任务，记录 {len(episodes)} 个评估回合。模型权重和原 13 维观测保持原样，未重新训练；本报告不修改原论文。",
        "",
        "## 主要结果",
        "",
    ]
    for paragraph in _conclusion_lines(summary, paired):
        lines.extend([paragraph, ""])
    if summary:
        lines.extend(["![各场景安全到达率及成功任务计数](figures/success_overview.png)", ""])
    lines.extend([
        "## 任务构造和信息条件",
        "",
        f"生成种子为 {task_data.get('seed', config.get('seed', 20261107))}。每组 A/B/C 使用同一起始关节角与目标；初始关节角为 [0.2, 0.5, −0.6] rad，基座在原点，连杆长 [0.40, 0.30, 0.30] m，连杆半径 0.02 m。目标位于距基座 0.55～0.95 m、初始末端方向 ±0.9 rad 的区域，起始末端到目标至少 0.20 m。",
        "",
        "| 场景 | 定义 | 核验用途 |",
        "|---|---|---|",
        "| A 无障碍 | 移除障碍，保持同组起点和目标，四维障碍输入变为零 | 无障碍输入条件下的到达表现，含观测分布变化 |",
        "| B 障碍不挡路 | 两个半径 0.08 m 的圆形障碍位于后方，并验证参考路径不被阻挡 | 普通障碍场景 |",
        "| C 直达受阻 | 仅移动 B 的一个障碍至直接到达通道，另一个保持原位 | 检验应对实际阻挡的能力 |",
        "",
        "C 类移动障碍的圆心位于初始末端—目标线段内部附近，到线段距离小于 0.015 m。该条件把 0.02 m 连杆半径、整个 0.08 m 成功圆以及 0.005 m 阻挡裕量计入判定，避免只挡目标中心而仍允许直达成功圆边缘。每组另外保存无碰撞的目标关节姿态与参考路径，验证在相同动作限幅和 120 步限制下可以到达，参考路径要求至少 0.01 m 表面间隙。",
        "",
        "可行参考路径仅用于证明这些测试任务具有可行解及辅助画图，不提供给 SAC。B 类“不挡路”指障碍不阻挡已验证的参考路线，不保证任意控制器产生的轨迹都安全。候选接受与拒绝只根据几何和参考路径，不依据 SAC 或 DLS 成绩。",
        "",
        f"生成过程检查 {generation.get('candidates_seen', '见记录')} 个候选，接受 {generation.get('accepted_groups', len(groups))} 组。拒绝原因计数：`{json.dumps(generation.get('rejections', {}), ensure_ascii=False)}`。完整记录见 [generation_log.json](generation_log.json)。",
        "",
        "候选目标并非按照原论文的半径与角度均匀采样：本次先在 [-π,π]³ 内均匀采样目标关节姿态，再通过正运动学得到末端目标，筛选扇区、起终距离、阻挡和参考路径条件。即使坐标范围相同，目标分布也不同。可行性筛选和固定后方障碍进一步形成构造测试的选择偏差，因此本测试不能估计原论文任务分布或一般随机场景的成功率。",
        "",
    ])
    if len(groups) < 10:
        lines.extend([f"**任务生成不足：实际仅得到 {len(groups)} 组，未自动放宽条件或补入容易任务。**因此结果分母以实际任务数为准。", ""])
    lines.extend([
        "## 控制器、奖励与评估口径",
        "",
        "四种方法为 SAC 种子0、SAC 种子1、DLS关闭避障项和DLS保留避障项。SAC输入仍是三个关节角、末端坐标、目标坐标、目标位移和四维障碍距离信息，评估使用确定性动作；DLS直接使用环境几何。两者的信息条件不同，本实验不据此声称某算法普遍优越。",
        "",
        "每次关节动作限于 ±0.05 rad，关节限幅 ±π，最多120步。两次容差评估使用完全相同的任务和模型，仅分别以 0.08 m 与 0.02 m 为末端到达阈值。评估不更新模型，也不以奖励高低判定结果。原模型训练中的奖励和网络权重未改动。",
        "",
        "安全到达、碰撞和超时互斥，碰撞优先。间隙为连杆中心线到障碍圆心距离减去障碍半径与连杆半径；接触或重叠判为碰撞。无障碍任务的间隙为不适用，CSV留空而不是写成0。最小间隙包括初始姿态与动作中的检查点。",
        "",
        "每个动作按最大关节变化不超过0.001 rad细分检查，首次检测到碰撞立即终止；保存轨迹再以0.0005 rad复核。该检查是有限分辨率的运动段检查，不是数学连续安全证明；不涵盖动力学、执行器误差或自碰撞。",
        "",
        "以下平均终点误差包含全部回合；平均步数和末端路径长度只统计安全成功回合。若没有成功回合，相应均值标为“—”。两个SAC种子分别报告，未将同一批任务在两个模型上的结果视为独立的新任务样本。",
        "",
    ])
    for tol in tolerances:
        lines.extend([f"### 到达容差 {tol:.2f} m", ""])
        lines.extend(_result_table([r for r in summary if r["tolerance_m"] == tol]))
        lines.append("")

    missed = refinement.get("missed_collision_episodes", [])
    lines.extend([
        "## 检查和复核记录",
        "",
        f"自动检查总体状态：**{'通过' if passed else '未通过或尚未确认'}**。预期回合数 {expected_episodes}，实际 {len(episodes)}。详细的中途碰撞、终点与碰撞同时发生、关节限幅、容差、参考路径重放和重复运行检查见 [checks.json](checks.json)。",
        "",
        f"轨迹加密复核记录数：{refinement.get('audited_episodes', '见原始记录')}；检测到粗检查漏碰撞回合数：{len(missed)}；是否以更细分辨率重新评估：{'是' if refinement.get('rerun_at_fine') else '否'}。详细结果见 [refinement.json](refinement.json)。若检查未通过，结果应视为诊断材料，不作为已验证结论。",
        "",
        "模型文件SHA-256与运行环境记录见 [config.json](config.json)，固定任务和参考路径见 [tasks.json](tasks.json)。",
        "",
        "## 固定展示的配对轨迹",
        "",
        "下列任务组在评估前指定为第1、5、10组，包含两个SAC模型与两种到达容差，不按结果挑选。蓝线为实际末端轨迹，浅蓝为起始/中间整臂姿态，橙色为结束姿态，灰色圆为实际障碍；C类灰虚线为可行参考末端轨迹。连杆按0.02 m物理半径绘制。成功圆大小随评估容差变化。",
        "",
    ])
    if missing_groups:
        lines.extend([f"实际任务中没有预定展示组 {missing_groups}，未用其他成功组替换。", ""])
    for group, method, tol, filename in fixed_figures:
        caption = f"第{group}组，{METHOD_LABELS.get(method, method)}，容差{tol:.2f} m"
        lines.extend([f"**{caption}**", "", f"![{caption}](figures/{filename})", ""])
    lines.extend([
        "## 按编号选择的首个失败案例",
        "",
        "对每个方法和容差组合，按任务组编号、再按A/B/C顺序选取首个失败，展示其完整配对场景。相同任务可能也出现在固定展示中。没有失败的组合不人为补例。",
        "",
    ])
    if not failure_figures:
        lines.extend(["本次记录没有失败回合。", ""])
    for group, method, tol, scenario, filename in failure_figures:
        caption = f"{METHOD_LABELS.get(method, method)}，容差{tol:.2f} m：首个失败为第{group}组{scenario}类"
        lines.extend([f"**{caption}**", "", f"![{caption}](figures/{filename})", ""])
    lines.extend([
        "## 结论适用范围与数据文件",
        "",
        "本次是小规模、针对几何阻挡的构造测试，不能估计任意随机环境的总体成功概率。固定初始姿态、静态圆形障碍、运动学模型和压缩障碍观测均限制了外推范围。新任务的障碍位置及更严格的运动段碰撞检查与原论文随机评估不同，不能直接以两次成功率差值解释模型进步或退步。",
        "",
        "路径长度和步数均值存在成功样本选择差异；应与成功任务数一起阅读。配对轨迹RMS差异只是相同步数前缀上的描述量，不是显著性检验，也不单独证明安全避障。",
        "",
        "## 复现命令",
        "",
        "在项目根目录运行以下命令，仅从固定JSON重新生成报告、CSV和图表，不运行控制器，不修改任务、模型或逐回合结果：",
        "",
        "```powershell",
        "$env:MPLCONFIGDIR = Join-Path (Get-Location) 'arm_rl/avoidance_validation_20261107/.mplconfig'",
        "python -B arm_rl/avoidance_report.py arm_rl/avoidance_validation_20261107",
        "```",
        "",
        "如需另行核验原评估脚本，下面命令会在新目录重新生成任务并运行240回合，不训练模型；该操作不是本次报告收尾所执行的内容：",
        "",
        "```powershell",
        "python -B arm_rl/validate_avoidance.py --out arm_rl/avoidance_validation_reproduction --stage all",
        "```",
        "",
        "新目录应不存在。固定种子和已保存配置用于复现；跨运行环境的数值一致性应另行核对。原论文、环境与训练代码、模型权重均未因本次报告修订而修改。",
        "",
        "- [episodes.csv](episodes.csv)：每个方法、任务、容差的最终状态和指标。",
        "- [summary.csv](summary.csv)：按场景、方法、容差汇总，包含成功样本数与间隙指标。",
        "- [paired.csv](paired.csv)：同组A/B/C结果和轨迹差异。",
        "- [episodes.json](episodes.json)：完整逐步关节状态、动作、末端位置、误差和检查间隙。",
        "- [tasks.json](tasks.json)：固定任务坐标、目标关节姿态、参考路径和阻挡几何。",
        "- [generation_log.json](generation_log.json)：任务生成与拒绝原因。",
        "- [checks.json](checks.json)、[refinement.json](refinement.json)：自动检查与轨迹加密复核。",
        "- [config.json](config.json)：配置、运行环境与模型哈希。",
        "",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(out_dir / "report.md"), "summary_rows": len(summary), "episodes": len(episodes), "fixed_figures": len(fixed_figures), "failure_figures": len(failure_figures)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation report and plots from frozen JSON artifacts.")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.out_dir), ensure_ascii=False, indent=2))

