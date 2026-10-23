"""Compare saved demonstration/replay states, with no physics or model imports.

Statistics use all aligned samples. Plots retain actual simulation times; they
never stretch a partial run or interpolate missing states into a success.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime"))
from lesson_common import guard
from portable_paths import resolve_source

MAX_BYTES = 256 * 1024 * 1024
MAX_ROWS = 250001
MAX_PLOT = 8000
XYZ = tuple(f"{p}_{a}_m" for p in ("actual", "cube") for a in "xyz")
CONTINUOUS = ("finger1_q_m", "finger2_q_m", "position_error_m",
              "orientation_error_rad", "finger1_normal_n", "finger2_normal_n",
              "placement_continuous_s")
DISCRETE = ("finger1_contact_count", "finger2_contact_count",
            "cube_table_contact_count", "cube_tray_base_contact_count",
            "cube_tray_wall_contact_count", "unexpected_contact_count",
            "release_active", "placement_valid", "grasp_hold_passed",
            "transfer_completed", "placement_passed")
GATES = ("manual_task_completed", "replay_completed", "sequence_completed",
         "grasp_hold_passed", "transfer_completed", "release_started",
         "placement_passed", "tracking_thresholds_passed")
REFERENCE = tuple(f"{p}_{j}" for p in ("q_ref", "qd_ref") for j in range(9))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def safe_path(path, exists=True):
    result = Path(path).absolute()
    for part in (result, *result.parents):
        require(not part.is_symlink() and not part.is_junction(), "Reparse paths are not accepted.")
    result = result.resolve()
    require(result.is_relative_to((ROOT / "outputs").resolve()), "Inputs/outputs must be under project outputs.")
    if exists:
        require(result.is_dir(), f"Run directory is missing: {result}")
    return result


def regular(path):
    require(path.is_file() and not path.is_symlink() and not path.is_junction(), f"Missing regular file: {path}")
    require(path.stat().st_size <= MAX_BYTES, "Input exceeds 256 MiB bound.")


def number(value, name):
    require(value is not None and value != "" and not isinstance(value, bool), f"Missing numeric field {name}")
    result = float(value)
    require(math.isfinite(result), f"Nonfinite field {name}")
    return result


def stats(values):
    if not values:
        return None
    n = len(values)
    return dict(samples=n, mean=sum(values)/n,
                rmse=math.sqrt(sum(v*v for v in values)/n),
                max_abs=max(abs(v) for v in values),
                minimum=min(values), maximum=max(values), final=values[-1])


def read_run(run):
    summary_path, csv_path = run / "summary.json", run / "manual.csv"
    regular(summary_path)
    regular(csv_path)
    hashes = {"summary.json": sha(summary_path), "manual.csv": sha(csv_path)}
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    expected = summary.get("manual_csv_sha256")
    if expected:
        require(str(expected).lower() == hashes["manual.csv"], "CSV hash differs from saved summary.")
    dt = number(summary.get("dt_s", summary.get("model_timestep_s")), "dt_s")
    require(dt > 0, "Timestep must be positive.")
    steps = number(summary.get("steps"), "steps")
    require(steps >= 0 and steps.is_integer(), "Invalid summary step count.")
    rows, issues = [], []
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ("sim_s", "terminal_state", *XYZ, *CONTINUOUS, *DISCRETE, *REFERENCE)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Missing/duplicate CSV header.")
        require(all(k in reader.fieldnames for k in required), "Missing required comparison columns.")
        for raw in reader:
            require(len(rows) < MAX_ROWS, "Sample count exceeds comparison bound.")
            terminal = number(raw["terminal_state"], "terminal_state")
            row = {k: (None if terminal == 1 and k.startswith("qd_ref_") and raw[k] == ""
                       else number(raw[k], k)) for k in required}
            t = row["sim_s"]
            step = round(t / dt)
            require(step >= 0 and abs(t-step*dt) <= max(1e-8, dt*1e-4), "Sample time is not on the simulation step grid.")
            require(step == len(rows), "State timeline must start at step zero and have no missing/duplicate steps.")
            if "step_id" in raw:
                require(number(raw["step_id"], "step_id") == step, "Saved step_id disagrees with simulation time.")
            require(row["terminal_state"] in (0, 1), "Invalid terminal flag.")
            require(not rows or not rows[-1]["terminal_state"], "Samples after terminal state.")
            for k in DISCRETE:
                require(row[k] >= 0 and row[k].is_integer(), f"Invalid discrete measurement {k}")
            for k in ("position_error_m", "orientation_error_rad", "finger1_normal_n", "finger2_normal_n", "placement_continuous_s"):
                require(row[k] >= 0, f"Negative nonnegative measurement {k}")
            row["step_id"] = step
            rows.append(row)
    if len(rows) != int(steps)+1:
        issues.append("Saved samples differ from summary steps + 1; no rows were synthesized.")
    if not rows or not rows[-1]["terminal_state"]:
        issues.append("No saved terminal row.")
    duration = number(summary.get("simulation_s"), "simulation_s")
    if not rows or abs(rows[-1]["sim_s"]-duration) > 1e-7:
        issues.append("Saved final time differs from summary simulation_s.")
    if summary.get("data_saved_samples") is not None and int(summary["data_saved_samples"]) != len(rows):
        issues.append("Saved sample count differs from summary data_saved_samples.")
    if summary.get("status") == "completed":
        require(not issues, "Completed summary has incomplete state evidence: " + "; ".join(issues))
    return dict(path=run, summary=summary, hashes=hashes, rows=rows, dt_s=dt,
                recording_issues=issues, csv_hash_declared=bool(expected))


def run_label(record):
    summary = record["summary"]
    return dict(path=str(record["path"]), hashes=record["hashes"],
                csv_hash_verified_against_summary=record["csv_hash_declared"],
                status=summary.get("status"), end_reason=summary.get("end_reason"),
                steps=summary.get("steps"), saved_samples=len(record["rows"]),
                simulation_s=summary.get("simulation_s"),
                physics_loop_wall_s=summary.get("physics_loop_wall_s"),
                gates={k: summary.get(k) for k in GATES},
                recording_issues=record["recording_issues"])


def event_times(rows):
    return {key: next((r["sim_s"] for r in rows if r[key] > 0), None)
            for key in ("grasp_hold_passed", "transfer_completed", "release_active", "placement_valid", "placement_passed")}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def compare(replay_run, source_run=None, output_dir=None, tool_check=False):
    replay_run = safe_path(replay_run)
    replay = read_run(replay_run)
    declared_source = replay["summary"].get("source_run")
    if source_run is None:
        require(declared_source, "Replay summary has no source_run; provide --source-run for an explicit tool check.")
        source_run = resolve_source(replay_run / "summary.json", declared_source)
    source_run = safe_path(source_run)
    require(tool_check or declared_source, "A normal replay comparison requires summary.source_run.")
    if declared_source:
        require(source_run == safe_path(resolve_source(replay_run / "summary.json", declared_source)), "Explicit source differs from replay provenance.")
    require(tool_check or source_run != replay_run, "Self comparison requires --tool-check and is not a replay.")
    source = read_run(source_run)
    if not tool_check:
        require(not any(r["summary"].get("tool_check") or
                        str(r["summary"].get("kind", "")).startswith("synthetic")
                        for r in (source, replay)), "Synthetic/tool fixtures require --tool-check.")
    # Older summaries may omit an upstream-summary digest. If present, verify it.
    for field in ("source_summary_sha256", "source_summary_json_sha256"):
        if replay["summary"].get(field):
            require(str(replay["summary"][field]).lower() == source["hashes"]["summary.json"], "Source summary hash differs from replay provenance.")
    require(abs(source["dt_s"]-replay["dt_s"]) < 1e-12, "Different timesteps cannot be aligned as the same simulation steps.")
    sr, rr = source["rows"], replay["rows"]
    n = min(len(sr), len(rr))
    dt = source["dt_s"]
    full_coverage = bool(sr and len(rr) >= len(sr) and not replay["recording_issues"])
    metrics = {}
    for prefix in ("actual", "cube"):
        components = {a: [1000*(rr[i][f"{prefix}_{a}_m"]-sr[i][f"{prefix}_{a}_m"]) for i in range(n)] for a in "xyz"}
        metrics[f"{prefix}_trajectory_delta_mm"] = {
            "definition": "Replay actual minus demonstration actual at identical simulation steps; not tracking error.",
            "axes": {a: stats(v) for a, v in components.items()},
            "norm": stats([math.sqrt(sum(components[a][i]**2 for a in "xyz")) for i in range(n)])}
    for field in CONTINUOUS:
        unit_scale = 1000 if field.endswith("_m") else 1
        metrics[field] = dict(
            display_unit="mm" if unit_scale == 1000 else ("rad" if field.endswith("_rad") else "N" if field.endswith("_n") else "s"),
            source_overlap=stats([r[field]*unit_scale for r in sr[:n]]),
            replay_overlap=stats([r[field]*unit_scale for r in rr[:n]]),
            replay_minus_source=stats([(rr[i][field]-sr[i][field])*unit_scale for i in range(n)]))
    metrics["tracking_error_definition"] = "position_error_m is each run's actual tool minus its current FK reference norm; it is independent of demonstration/replay trajectory differences."
    metrics["contact_and_flag_disagreements"] = {field: dict(
        different_samples=sum(sr[i][field] != rr[i][field] for i in range(n)),
        source_positive_samples=sum(r[field] > 0 for r in sr[:n]),
        replay_positive_samples=sum(r[field] > 0 for r in rr[:n])) for field in DISCRETE}
    metrics["reference_max_abs_difference"] = {p: max((abs(rr[i][f"{p}_{j}"]-sr[i][f"{p}_{j}"])
        for i in range(n) if not sr[i]["terminal_state"] and not rr[i]["terminal_state"]
        for j in range(9)), default=None) for p in ("q_ref", "qd_ref")}
    metrics["reference_difference_semantics"] = "Only shared nonterminal pre-step actions; terminal qd_ref may be absent and is never replaced by zero."
    se, re = event_times(sr), event_times(rr)
    events = {k: {"source_first_sim_s": se[k], "replay_first_sim_s": re[k],
                  "difference_s": re[k]-se[k] if se[k] is not None and re[k] is not None else None} for k in se}
    output_dir = safe_path(output_dir, exists=False) if output_dir else replay_run / ("comparison_"+datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")+"_"+uuid.uuid4().hex[:6])
    require(not output_dir.exists(), "Output directory already exists; records must not be overwritten.")
    output_dir.mkdir(parents=True)
    summary = dict(kind="saved_replay_comparison", schema_version=1, tool_check=bool(tool_check),
        new_physics_steps=0, source=run_label(source), replay=run_label(replay),
        alignment=dict(method="integer simulation step; no time shifting, interpolation or time warping",
            dt_s=dt, common_samples=n, first_step=0 if n else None, last_step=n-1 if n else None,
            common_simulation_window_s=[0, sr[n-1]["sim_s"]] if n else None,
            source_full_timeline_covered=full_coverage, source_uncompared_samples=len(sr)-n,
            replay_uncompared_samples=len(rr)-n),
        metrics=metrics, first_observed_events=events,
        task_assessment="Saved outcome flags are quoted; this comparator does not change success criteria or certify a replay.",
        limitations=["This 12-panel comparison covers position, fingers, contacts and task timing; full orientation/velocity trajectory comparisons are outside this view, even when newly recorded.",
                     "Statistics apply only to overlapping recorded steps; stopped runs are not extrapolated.",
                     "Event times are first saved observations and are quantized by the recording timestep."])
    if tool_check:
        summary["limitations"].insert(0, "TOOL CHECK ONLY: this comparison provides no evidence of a new dynamic replay.")
    plot_rows = []
    # Both runs share the demonstration's world origin; never center each run independently.
    origin = {p: [sr[0][f"{p}_{a}_m"] for a in "xyz"] for p in ("actual", "cube")} if sr else {}
    for i in range(n):
        row = dict(sim_s=sr[i]["sim_s"], step_id=i)
        for label, r in (("source", sr[i]), ("replay", rr[i])):
            for p in ("actual", "cube"):
                for axis, offset in zip("xyz", origin[p]):
                    row[f"{label}_{p}_{axis}_mm"] = 1000*(r[f"{p}_{axis}_m"]-offset)
            for field in CONTINUOUS:
                display_field = field[:-2]+"_mm" if field.endswith("_m") else field
                row[label+"_"+display_field] = r[field]*(1000 if field.endswith("_m") else 1)
        for p in ("actual", "cube"):
            row[p+"_trajectory_delta_mm"] = math.sqrt(sum((1000*(rr[i][f"{p}_{a}_m"]-sr[i][f"{p}_{a}_m"]))**2 for a in "xyz"))
        plot_rows.append(row)
    keep = {0, n-1} if n else set()
    for i in range(1, n):
        if any(sr[i][k] != sr[i-1][k] or rr[i][k] != rr[i-1][k] for k in DISCRETE):
            keep.update((i-1, i))
    if plot_rows:
        for k in plot_rows[0]:
            if k not in ("step_id", "sim_s"):
                keep.update((min(range(n), key=lambda i: plot_rows[i][k]), max(range(n), key=lambda i: plot_rows[i][k])))
    # Preserve discrete transitions even if they exceed the desired budget.
    stride = max(1, math.ceil(n / max(1, MAX_PLOT-len(keep))))
    keep.update(range(0, n, stride))
    selected = sorted(keep)
    payload = dict(kind="replay_comparison_plot_payload", tool_check=bool(tool_check),
        source_status=source["summary"].get("status"), replay_status=replay["summary"].get("status"),
        replay_end_reason=replay["summary"].get("end_reason"), full_coverage=full_coverage,
        source_duration_s=source["summary"].get("simulation_s"), overlap_samples=n,
        plotted_samples=len(selected), stride=stride, retained_indices=selected,
        common_window_s=summary["alignment"]["common_simulation_window_s"], origin_world_m=origin,
        comparison_json=str(output_dir/"comparison.json"), rows=[plot_rows[i] for i in selected],
        output_png=str(output_dir/"comparison.png"), output_source_json=str(output_dir/"comparison.source.json"))
    for record in (source, replay):
        for name, digest in record["hashes"].items():
            require(sha(record["path"]/name) == digest, "Input changed during comparison.")
    write_json(output_dir/"comparison.json", summary)
    write_json(output_dir/"comparison.plotdata.json", payload)
    lines = ["# 原示范与动力学回放：已保存数据对比", "",
        "状态：工具校验，未产生新回放。" if tool_check else "状态：只读统计；任务标签引用原摘要，不重新判定成功。", "",
        f"原示范：`{source_run}`", f"回放：`{replay_run}`", "",
        f"原示范 {source['summary'].get('status')} / {source['summary'].get('end_reason')}；回放 {replay['summary'].get('status')} / {replay['summary'].get('end_reason')}。",
        f"步长 {dt:g} s；共同样本 {n}；共同区间 " + (f"0–{sr[n-1]['sim_s']:.6f} s。" if n else "无。"),
        "覆盖原示范完整时间线。" if full_coverage else "**未覆盖完整示范；提前结束或缺失部分不补点、不延伸，也不作为完整回放通过。**", "",
        "## 两种误差分开报告", "",
        "轨迹差 = 同一仿真步的回放实际位置 − 原示范实际位置。跟踪误差 = 各自实际工具位置与各自当前参考的距离。以下统计均取重叠区间全部原始样本。", "",
        "| 指标 | 峰值 / 最大绝对值 | RMSE |", "|---|---:|---:|"]
    for p, label in (("actual", "工具三维轨迹差 / mm"), ("cube", "方块三维轨迹差 / mm")):
        st = metrics[p+"_trajectory_delta_mm"]["norm"]
        if st:
            lines.append(f"| {label} | {st['max_abs']:.9g} | {st['rmse']:.9g} |")
    for label, key in (("原示范位置跟踪误差 / mm", "source_overlap"), ("回放位置跟踪误差 / mm", "replay_overlap")):
        st = metrics["position_error_m"][key]
        if st:
            lines.append(f"| {label} | {st['max_abs']:.9g} | {st['rmse']:.9g} |")
    lines += ["", "## 首次观测事件", "", "| 事件 | 示范仿真秒 | 回放仿真秒 | 回放−示范 / s |", "|---|---:|---:|---:|"]
    def fmt(v):
        return "未观测" if v is None else f"{v:.6f}"
    for key, event in events.items():
        lines.append(f"| {key} | {fmt(event['source_first_sim_s'])} | {fmt(event['replay_first_sim_s'])} | {fmt(event['difference_s'])} |")
    lines += ["", "夹爪位置、接触力/计数、落盘连续时长、参考差异及输入 SHA-256 见 comparison.json。计数不一致样本数不等同于独立碰撞事件数。",
              "曲线使用实际仿真时间，无时间伸缩或平滑；两次轨迹采用同一个示范初始位置作为零点。绘图可抽样，统计不抽样。",
              "本12面板视图未比较完整朝向/速度轨迹；新记录中的这些字段仍保存在原始CSV，旧记录缺失的字段不补造。朝向跟踪误差不等于两次运行的朝向轨迹差。绘图与统计不推进物理，不证明新初态泛化。", ""]
    if source["recording_issues"] or replay["recording_issues"]:
        lines += ["记录问题：", *["- "+s for s in source["recording_issues"]+replay["recording_issues"]], ""]
    (output_dir/"对比简报.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(kind="replay_comparison_receipt", protocol=1, comparison_dir=str(output_dir),
                comparison_json=str(output_dir/"comparison.json"), report=str(output_dir/"对比简报.md"),
                plotdata_json=str(output_dir/"comparison.plotdata.json"),
                plotdata_sha256=sha(output_dir/"comparison.plotdata.json"), tool_check=bool(tool_check))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--output-dir")
    parser.add_argument("--tool-check", action="store_true")
    args = parser.parse_args()
    guard()
    receipt = compare(args.replay_run, args.source_run, args.output_dir, args.tool_check)
    # ASCII JSON preserves Chinese paths across Windows PowerShell native stdout.
    print(json.dumps(receipt, ensure_ascii=True))


if __name__ == "__main__":
    main()
