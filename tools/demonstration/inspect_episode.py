"""Read a v0.1 recording and prepare a v0.2 reference timeline; no physics/imports.

The output is a legacy reference export, not a complete v0.2 episode or a replay.
Uses only the Python standard library and never writes into the input recording.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from portable_paths import output_path, contained_path, relative_source


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numbers(row, prefix):
    values = [float(row[f"{prefix}_{j}"]) for j in range(9)]
    require(all(math.isfinite(v) for v in values), f"Nonfinite {prefix}")
    return values


def inspect(run, output):
    run, output = output_path(run), output_path(output, exists=False)
    require(run.is_dir() and not output.is_relative_to(run), 'Use a separate outputs directory')
    require(not output.exists(), 'Output already exists; use a fresh directory')
    summary = read_json(contained_path(run, "summary.json"))
    config = read_json(contained_path(run, "configs/config.json"))
    frozen = read_json(contained_path(run, "configs/frozen_context.json"))
    initial = read_json(contained_path(run, "manual_initialization.json"))
    sources = {p.relative_to(run).as_posix(): digest(contained_path(run,p.relative_to(run)))
               for p in sorted(run.rglob("*")) if p.is_file()}
    checks = []

    def check(name, condition):
        require(condition, name)
        checks.append(name)

    check("original_csv_hash", sources["manual.csv"] == summary["manual_csv_sha256"])
    check("original_jog_hash", sources["jog_references.jsonl"] == summary["jog_references_sha256"])
    for item in summary["snapshots"]:
        check("snapshot:" + item["path"], sources.get(item["path"]) == item["sha256"])
    dt = float(summary["dt_s"])
    total = int(summary["steps"])
    check("positive_clock", math.isfinite(dt) and dt > 0 and total > 0)
    check("clock_matches_frozen_physics", dt == frozen["effective_solver_options"]["timestep"] ==
          summary["frozen_physics_options"]["timestep"])
    check("nine_joint_order", len(config["joint_order"]) == 9)
    for key in ("initial_qpos", "initial_qvel"):
        check(key + "_matches_frozen", summary[key] == frozen[key])
        check(key + "_finite", all(math.isfinite(float(v)) for v in frozen[key]))
    for key in ("initial_ctrl", "initial_qacc_warmstart"):
        check(key + "_available", len(frozen[key]) > 0 and
              all(math.isfinite(float(v)) for v in frozen[key]))
    for key in ("kp", "kd"):
        check(key + "_matches_frozen", summary[key] == frozen[key] and len(summary[key]) == 9)
    check("caps_match_frozen", summary["software_caps"] == frozen["cap"])

    output.mkdir(parents=True, exist_ok=False)
    reference_file = output / "references.csv"
    columns = ["step_id", "sim_s", "command_id", "phase"] + [
        f"{prefix}_{j}" for prefix in ("q_ref", "qd_ref") for j in range(9)]
    max_clock_error = 0.0
    row_count = 0
    first_motion = None
    with (run / "manual.csv").open(encoding="utf-8-sig", newline="") as source, \
            reference_file.open("x", encoding="utf-8", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for k, row in enumerate(reader):
            row_count += 1
            require(k <= total, "More rows than summary steps + 1")
            t = float(row["sim_s"])
            require(math.isfinite(t), f"Invalid time at row {k}")
            error = abs(t - k * dt)
            max_clock_error = max(max_clock_error, error)
            require(error < 1e-7, f"Missing/repeated/time-shifted row at step {k}")
            require(int(row["terminal_state"]) == int(k == total), f"Invalid terminal row {k}")
            qref, actual, velocity = numbers(row, "q_ref"), numbers(row, "q"), numbers(row, "qd")
            require(abs(qref[7] - float(row["finger_ref_m"])) < 1e-12 and
                    abs(qref[8] - float(row["finger_ref_m"])) < 1e-12,
                    f"Finger reference mismatch at step {k}")
            if k == 0:
                addresses = config["joint_qpos_addresses"]
                check("first_actual_matches_initial_qpos", all(
                    abs(actual[j] - summary["initial_qpos"][addresses[j]]) < 1e-12 for j in range(9)))
                check("first_reference_matches_initialization", qref == initial["q_ref9"])
            if k < total:
                qdref = numbers(row, "qd_ref")
                if first_motion is None and any(abs(v) > 1e-10 for v in qdref):
                    first_motion = {"step_id": k, "sim_s": t, "command_id": row["command_id"]}
                writer.writerow({**{name: row[name] for name in columns if name != "step_id"}, "step_id": k})
            else:
                check("terminal_has_no_future_reference_velocity", all(row[f"qd_ref_{j}"] == "" for j in range(9)))
                check("terminal_time_matches_summary", abs(t - summary["simulation_s"]) < 1e-8)
                terminal = {"step_id": k, "sim_s": t, "source_row_zero_based": k,
                            "has_next_action": False, "phase": row["phase"]}
    check("samples_equal_steps_plus_one", row_count == total + 1)
    check("samples_match_saved_summary", row_count == summary["data_saved_samples"])
    checks.extend(["all_step_times_contiguous", "all_action_references_finite",
                   "all_measured_joint_states_finite", "all_finger_references_consistent"])

    events = []
    with (run / "events.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                events.append(json.loads(line))
    event_groups = {}
    for item in events:
        key = (item.get("stage"), item.get("reason"), item.get("accepted"))
        event_groups[key] = event_groups.get(key, 0) + 1
    for relative, before in sources.items():
        require(digest(run / relative) == before, "Source changed while reading: " + relative)
    checks.append("source_hashes_unchanged_after_read")
    gaps = ["旧CSV没有逐步工具完整朝向、方块四元数、方块线/角速度向量；只有位置、部分标量误差与速率。",
            "完整MuJoCo积分状态维度及未用状态、初始ctrl/warmstart的运行时恢复语义，需回放入口逐项核对。",
            "控制器、输入/参考生成器和任务观察器的软件初态尚未序列化为统一v0.2格式；需按冻结源码明确重建。"]
    manifest = {
        "format": "g1.legacy_reference_export.v0.2-draft1",
        "result": "REFERENCE_TIMELINE_CHECK_PASSED",
        "source_run": relative_source(output, run), "episode_id": run.name,
        "created_at_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "interpreter": sys.executable, "python_version": sys.version.split()[0],
        "inspector_sha256": digest(Path(__file__)),
        "dt_s": dt, "state_count": row_count, "action_count": total,
        "first_action_step": 0, "last_action_step": total - 1,
        "terminal": terminal, "joint_order": config["joint_order"],
        "units": {"joints_0_to_6": "rad, rad/s", "joints_7_to_8": "m, m/s"},
        "semantics": "row k measures s_k and references for [k*dt,(k+1)*dt); terminal row N has no next action",
        "first_motion": first_motion, "max_clock_grid_error_s": max_clock_error,
        "checks": checks, "snapshot_count": len(summary["snapshots"]),
        "source_files_sha256": sources,
        "references_sha256": digest(reference_file),
        "historical_result_from_summary": {key: summary[key] for key in (
            "status", "end_reason", "manual_task_completed", "grasp_hold_passed", "transfer_completed",
            "release_started", "placement_passed", "tracking_thresholds_passed")},
        "historical_milestones_from_summary": summary["manual_task_milestones"],
        "event_groups": [{"stage": key[0], "reason": key[1], "accepted": key[2], "count": count}
                         for key, count in event_groups.items()],
        "gaps": gaps, "complete_v02_episode": False,
        "dynamic_replay_executed": False, "replay_passed": None,
    }
    (output / "episode-index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = {"manual_close": "闭爪参考开始", "grip_baseline": "建立持物基准",
              "lift_observed": "观察到抬升", "transfer_observed": "观察到运输",
              "grasp_hold": "抓持评分通过", "manual_open": "开爪参考开始", "placement": "落盘评分通过"}
    table = "\n".join(f"| {labels.get(key, key)} | {value['step']} | {value['sim_s']:.3f} |"
                      for key, value in sorted(summary["manual_task_milestones"].items(), key=lambda pair: pair[1]["step"]))
    report = f"""# v0.2 第一步：一条旧示范的参考时间线检查

回合：`{run.name}`。检查通过，**尚未进行动力学回放**。

这一步回答：旧记录是否保存了每个物理步应该执行的关节参考。输出为兼容旧记录的参考导出，尚非完整v0.2采集格式。

| 检查项 | 实际结果 |
|---|---|
| 仿真时长 | {summary['simulation_s']:.3f} s |
| 物理步长 | {dt:.3f} s（{1/dt:.0f} Hz） |
| 状态行数 | {row_count}，含初态和终态 |
| 有下一步动作的参考行数 | {total}，step_id为0至{total-1} |
| 时间连续性 | 无缺步/重复步，最大网格误差{max_clock_error:.3g} s |
| 参考/实测关节数据 | 9维；所有动作行位置与速度参考均为有限数 |
| 原始CSV/方向参考哈希 | 与原summary一致 |
| 冻结快照哈希 | {len(summary['snapshots'])}项一致 |
| 输入原件 | {len(sources)}文件读取前后SHA-256一致 |

## 怎么读这一回合

第k行先测量当前状态s_k，用该行q_ref/qd_ref计算反馈力矩，再推进一次物理步得到s_(k+1)。最后一行只记录终态，不能再执行一个动作。

q_ref是控制器要跟踪的关节参考；q是动力学产生的实际关节位置。0—6是手臂转动关节（rad），7—8是夹爪移动关节（m）。关节顺序见episode-index.json。

仿真步决定回放顺序；wall_s包含人工等待和规划等现实开销。回放要重新算PD与重力补偿，不用原力矩代替反馈，不逐帧设置实际关节或方块位置。

## 这次人工示范的事件

以下时间来自旧summary，仅说明原人工回合的记录；不是新回放的结果。“开合参考开始”不等于夹爪已经完成开合。

| 事件 | 物理步 | 仿真秒 |
|---|---:|---:|
{table}

## 还需要补什么

""" + "\n".join("- " + gap for gap in gaps) + """

旧数据的缺失字段不填造。现有参考已支持继续开发回放入口；能否动力学重现，要等新进程实际执行后判断。完整新格式将在采集入口补齐，后续至少采一条验证新增字段。

## 本次产物与下一步

- references.csv：每个物理步的9维位置/速度参考，未包含终态的虚构动作。
- episode-index.json：来源哈希、关节顺序、初态文件关联、事件分类、缺失项和终态语义。
- 本报告：用于理解与检查。原始记录仍保存在原目录。

下一步准备独立回放入口和初始化核对，由用户启动一次真实动力学回放，再将原示范与回放曲线叠加比较。
"""
    (output / "检查报告.md").write_text(report, encoding="utf-8-sig")
    print("REFERENCE TIMELINE: PASS")
    print(f"States: {row_count}; actions: {total}; simulation: {summary['simulation_s']:.3f} s; dt: {dt:.3f} s")
    print(f"Snapshot hashes: {len(summary['snapshots'])} matched; input files unchanged: {len(sources)}")
    print("Complete v0.2 episode: NOT YET; dynamic replay: NOT RUN")
    print("EPISODE CHECK: " + str(output))
    print("REPORT: " + str(output / "检查报告.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = output_path(args.run), output_path(args.output, exists=False)
    project = Path(__file__).resolve().parents[2]
    require(run.is_relative_to(project / "outputs"), "Input must be inside project outputs")
    require(output.is_relative_to(project / "outputs") and not output.is_relative_to(run),
            "Output must be a separate project outputs directory")
    require(not output.exists(), "Output already exists; use a fresh directory")
    try:
        inspect(run, output)
    except Exception as exc:
        if output.exists():
            (output / "FAILED.txt").write_text(f"Inspection failed; partial export must not be used.\n{exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
