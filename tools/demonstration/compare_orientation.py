"""Compare saved world orientations. No MuJoCo import or physics execution.

wxyz unit quaternions describe local frames in world. The rotation vector is
log(R(t) R_demo(0)^T), expressed in world axes, not Euler angles. The same
demonstration baseline is used for both runs. Statistics never subsample.
"""
import argparse
import csv
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import compare_replay as saved

NORM_TOLERANCE = 1e-3
PREFIXES = ("actual", "cube")


def unit_quaternion(values):
    saved.require(len(values) == 4, "Quaternion requires wxyz components.")
    q = tuple(saved.number(v, "quaternion") for v in values)
    norm = math.hypot(*q)
    saved.require(norm > 0 and abs(norm - 1) <= NORM_TOLERANCE,
                  "Invalid quaternion norm; missing/corrupt orientations are not replaced.")
    return tuple(v / norm for v in q), abs(norm - 1)


def conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def canonical(q):
    # Unique sign except the unavoidable rotation-vector branch at pi.
    # At exactly w=0, the first nonzero vector component resolves q versus -q.
    first = next((v for v in q if v != 0), 1.)
    return tuple(-v for v in q) if first < 0 else q


def relative_rotation(q, baseline):
    delta, _ = unit_quaternion(product(q, conjugate(baseline)))
    delta = canonical(delta)
    sine = math.hypot(*delta[1:])
    if sine == 0:
        return delta, (0., 0., 0.)
    angle = 2 * math.atan2(sine, delta[0])
    return delta, tuple(math.degrees(angle * v / sine) for v in delta[1:])


def angular_distance_deg(a, b):
    # Chord formula is sign-invariant and better conditioned near zero than acos(dot).
    chord = min(math.dist(a, b), math.dist(a, tuple(-v for v in b)))
    return math.degrees(4 * math.asin(min(1., chord / 2)))


def read_orientations(record):
    path = record["path"] / "manual.csv"
    result = {p: [] for p in PREFIXES}
    norm_error = {p: 0. for p in PREFIXES}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(f"{p}_quat_{a}" for p in PREFIXES for a in "wxyz")
        saved.require(reader.fieldnames and all(f in reader.fieldnames for f in fields),
                      "Full tool/cube quaternion fields are absent (legacy data); no zero filling is permitted.")
        count = 0
        for index, raw in enumerate(reader):
            saved.require(index < len(record["rows"]), "CSV changed while reading orientations.")
            saved.require(saved.number(raw["sim_s"], "sim_s") == record["rows"][index]["sim_s"],
                          "Orientation and validated state timelines differ.")
            for p in PREFIXES:
                q, error = unit_quaternion([raw[f"{p}_quat_{a}"] for a in "wxyz"])
                result[p].append(q)
                norm_error[p] = max(norm_error[p], error)
            count += 1
    saved.require(count == len(record["rows"]) and count > 0, "Orientation samples missing.")
    saved.require(saved.sha(path) == record["hashes"]["manual.csv"], "CSV changed during orientation read.")
    return result, norm_error


def compare(replay_run, source_run=None, output_dir=None, tool_check=False):
    replay_run = saved.safe_path(replay_run)
    replay = saved.read_run(replay_run)
    declared = replay["summary"].get("source_run")
    saved.require(source_run or declared, "Replay has no declared source.")
    source_run = saved.safe_path(source_run or saved.resolve_source(replay_run / "summary.json", declared))
    saved.require(tool_check or declared, "Normal comparison requires replay provenance.")
    if declared:
        saved.require(source_run == saved.safe_path(saved.resolve_source(replay_run / "summary.json", declared)), "Explicit source differs from replay provenance.")
    saved.require(tool_check or source_run != replay_run, "Self comparison requires --tool-check.")
    source = saved.read_run(source_run)
    if not tool_check:
        saved.require(not any(r["summary"].get("tool_check") or
            str(r["summary"].get("kind", "")).startswith("synthetic") for r in (source, replay)),
            "Synthetic fixtures require --tool-check.")
    for field in ("source_summary_sha256", "source_summary_json_sha256"):
        if replay["summary"].get(field):
            saved.require(str(replay["summary"][field]).lower() == source["hashes"]["summary.json"],
                          "Source summary hash differs from replay provenance.")
    saved.require(abs(source["dt_s"] - replay["dt_s"]) < 1e-12, "Different timestep grids.")
    sq, sn = read_orientations(source)
    rq, rn = read_orientations(replay)
    n = min(len(source["rows"]), len(replay["rows"]))
    full = bool(n == len(source["rows"]) and not source["recording_issues"] and not replay["recording_issues"])
    origins = {p: sq[p][0] for p in PREFIXES}
    rows = [{"step_id": i, "sim_s": source["rows"][i]["sim_s"]} for i in range(n)]
    metrics, breaks = {}, {}
    for p in PREFIXES:
        metrics[p + "_orientation_delta_deg"] = saved.stats([angular_distance_deg(sq[p][i], rq[p][i]) for i in range(n)])
        for label, quats in (("source", sq), ("replay", rq)):
            prior = None
            branch_indices, rotation_magnitudes = [], []
            for i in range(n):
                relative, vector = relative_rotation(quats[p][i], origins[p])
                branch = prior is not None and sum(a*b for a, b in zip(prior, relative)) < 0
                if branch:
                    branch_indices.append(i)
                prior = relative
                rows[i][f"{label}_{p}_break"] = branch
                for a, v in zip("xyz", vector):
                    rows[i][f"{label}_{p}_rotvec_{a}_deg"] = v
                rotation_magnitudes.append(math.hypot(*vector))
            breaks[f"{label}_{p}"] = branch_indices
            metrics[f"{label}_{p}_rotation_from_demo_initial_deg"] = saved.stats(rotation_magnitudes)
        for i in range(n):
            rows[i][p + "_orientation_delta_deg"] = angular_distance_deg(sq[p][i], rq[p][i])
    for label, record in (("source", source), ("replay", replay)):
        values = [math.degrees(record["rows"][i]["orientation_error_rad"]) for i in range(n)]
        metrics[label + "_tracking_orientation_deg"] = saved.stats(values)
        for row, v in zip(rows, values):
            row[label + "_tracking_orientation_deg"] = v
    # Retain endpoints, each plotted field's extrema, and every branch neighbour.
    keep = {0, n - 1}
    for field in rows[0]:
        if field.endswith("_deg"):
            keep.add(min(range(n), key=lambda i: rows[i][field]))
            keep.add(max(range(n), key=lambda i: rows[i][field]))
    for indices in breaks.values():
        for i in indices:
            keep.update(j for j in (i-1, i, i+1) if 0 <= j < n)
    stride = max(1, math.ceil(n / saved.MAX_PLOT))
    keep.update(range(0, n, stride))
    selected = sorted(keep)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = saved.safe_path(output_dir or replay_run / ("orientation_" + stamp + "_" + uuid.uuid4().hex[:6]), exists=False)
    saved.require(not output_dir.exists(), "Output directory already exists; original outputs are never replaced.")
    for record in (source, replay):
        for name, digest in record["hashes"].items():
            saved.require(saved.sha(record["path"] / name) == digest, "Input changed during comparison.")
    output_dir.mkdir(parents=True)
    script_paths = (Path(__file__), Path(saved.__file__))
    summary = dict(kind="saved_orientation_comparison", schema_version=1, tool_check=bool(tool_check),
        new_physics_steps=0, source=saved.run_label(source), replay=saved.run_label(replay),
        implementation=[dict(path=str(p), sha256=saved.sha(p)) for p in script_paths],
        alignment=dict(common_samples=n, dt_s=source["dt_s"], common_window_s=[0, rows[-1]["sim_s"]],
                       source_full_timeline_covered=full, interpolation=False, statistics_use_all_samples=True),
        quaternion_semantics="wxyz; local tool site/cube body orientation in world; q and -q equivalent",
        rotation_vector_semantics="principal log(R(t) R_demo(0)^T), world-axis components in degrees, NOT Euler angles",
        common_demo_initial_wxyz=origins,
        quaternion_norm_check=dict(accepted_tolerance=NORM_TOLERANCE, source_max_error=sn, replay_max_error=rn,
                                   normalization="accepted near-unit quaternions normalized for calculations only"),
        branch_or_large_jump_indices=breaks, metrics=metrics,
        limitations=["Only overlapping saved states; absent legacy orientations cause rejection, never zero filling.",
            "Rotation-vector principal branch ends at 180 degrees; detected branch/large jumps are plotted with breaks.",
            "Angular trajectory difference compares actual replay with actual demonstration, not with controller reference.",
            "Tracking orientation error is quoted from each saved CSV, not reconstructed from missing reference quaternions.",
            "No physics, task re-scoring, learned policy or unseen-state generalization is produced by this comparison."])
    payload = dict(kind="orientation_comparison_plot_payload", tool_check=bool(tool_check),
        source_status=source["summary"].get("status"), replay_status=replay["summary"].get("status"),
        replay_end_reason=replay["summary"].get("end_reason"), full_coverage=full,
        source_duration_s=source["summary"]["simulation_s"], overlap_samples=n,
        plotted_samples=len(selected), stride=stride, retained_indices=selected,
        rows=[rows[i] for i in selected], output_png=str(output_dir / "orientation.png"),
        output_source_json=str(output_dir / "orientation.source.json"))
    saved.write_json(output_dir / "orientation.json", summary)
    saved.write_json(output_dir / "orientation.plotdata.json", payload)
    lines = ["# 示范与动力学回放：完整朝向对比", "",
        "状态：工具校验，未产生新回放。" if tool_check else "状态：只读保存数据；任务标签沿用原摘要。", "",
        f"示范：`{source_run}`", f"回放：`{replay_run}`", "",
        f"共同状态 {n}；区间 0–{rows[-1]['sim_s']:.6f} 仿真秒；完整覆盖：{full}。", "",
        "前六图是相对同一个示范初始朝向的旋转向量，XYZ 为世界轴，单位为度，不是欧拉角。向量方向表示旋转轴，长度表示最短旋转角。",
        "底行前两图是同一步两次实际朝向的最短夹角，q 与 -q 视为相同朝向。底行第三图是各自实际工具相对当前控制参考的跟踪角误差，直接引用保存字段并转为度；不是回放偏差。", "",
        "| 指标（度） | 峰值 | RMSE |", "|---|---:|---:|"]
    labels = (("actual_orientation_delta_deg", "工具两次朝向差"), ("cube_orientation_delta_deg", "方块两次朝向差"),
              ("source_tracking_orientation_deg", "示范工具跟踪角误差"), ("replay_tracking_orientation_deg", "回放工具跟踪角误差"),
              ("source_actual_rotation_from_demo_initial_deg", "示范工具相对初始旋转"),
              ("source_cube_rotation_from_demo_initial_deg", "示范方块相对初始旋转"))
    for key, label in labels:
        st = metrics[key]
        lines.append(f"| {label} | {st['max_abs']:.12g} | {st['rmse']:.12g} |")
    lines += ["", "统计使用全部共同样本；显示保留首尾、极值及分支邻点，不平滑、不插值、不伸缩时间。",
        "旋转向量采用最短角分支，跨 ±180° 的检测跳变断线显示；统计的最短夹角不受四元数符号翻转影响。",
        "输入 CSV/summary、算法版本哈希、归一化误差和分支位置见 orientation.json；旧记录缺失朝向不会被补成零。",
        "图形不会产生新仿真、修改原任务成功状态，或证明未见初态泛化。", ""]
    (output_dir / "朝向对比简报.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(kind="orientation_comparison_receipt", protocol=1, comparison_dir=str(output_dir),
        comparison_json=str(output_dir / "orientation.json"), report=str(output_dir / "朝向对比简报.md"),
        plotdata_json=str(output_dir / "orientation.plotdata.json"),
        plotdata_sha256=saved.sha(output_dir / "orientation.plotdata.json"), tool_check=bool(tool_check))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--output-dir")
    parser.add_argument("--tool-check", action="store_true")
    args = parser.parse_args()
    saved.guard()
    print(json.dumps(compare(args.replay_run, args.source_run, args.output_dir, args.tool_check), ensure_ascii=True))


if __name__ == "__main__":
    main()
