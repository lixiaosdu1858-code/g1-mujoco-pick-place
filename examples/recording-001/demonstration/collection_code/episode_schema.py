"""Validate and export one saved v0.2 episode, using standard-library file I/O.

The format is scoped to the frozen fixed-base G1 package, not arbitrary models.
Validation never simulates, repairs source data, or promotes a failed task.
"""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path


FORMAT = "g1.demonstration.v0.2.0"
REFERENCE_COLUMNS = ["step_id", "sim_s", "command_id", "phase"] + [
    f"{prefix}_{j}" for prefix in ("q_ref", "qd_ref") for j in range(9)]
SOURCE_NAMES = ("summary.json", "manual.csv", "initialization.json", "actions.jsonl",
                "events.jsonl", "jog_references.jsonl", "manual_initialization.json")
SEMANTIC_KEYS = ("state_action_timing", "joint_order", "tool_orientation", "cube_orientation",
                 "tool_velocity", "cube_velocity", "full_state", "input_events", "actions",
                 "planned_vs_executed", "gripper_position")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Source path escaped run: " + str(relative))
    require(not path.is_relative_to(root / "episode"), "Derived episode is not a source")
    return path


def number(value, label):
    result = float(value)
    require(math.isfinite(result), "Nonfinite " + label)
    return result


def integer(value, label):
    result = int(value)
    require(not isinstance(value, bool) and float(value) == result, "Noninteger " + label)
    return result


def vector(value, size, label):
    require(isinstance(value, list) and len(value) == size, "Wrong vector size: " + label)
    return [number(v, label) for v in value]


def columns(row, prefix, suffixes):
    return [number(row[f"{prefix}_{j}"], prefix) for j in suffixes]


def close(left, right, label, tolerance=1e-12):
    require(len(left) == len(right) and all(abs(a-b) <= tolerance for a, b in zip(left, right)),
            "Mismatch: " + label)


def json_lines(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_inventory(run, summary):
    """Explicit raw inventory: never recurse over plots or previous comparisons."""
    paths = set(SOURCE_NAMES)
    for item in summary.get("snapshots", []) + summary.get("collection_code", []):
        paths.add(item["path"])
    paths.update(p.relative_to(run).as_posix() for p in (run / "command_references").glob("*.json"))
    if (run / "field-semantics.json").is_file():
        paths.add("field-semantics.json")
    return {name: digest(local(run, name)) for name in sorted(paths)}


def check_initialization(init, frozen, summary, config):
    require(init["format"] == FORMAT, "Wrong initialization format")
    require(init["scope"] == "same_fixed_package_initial_state", "Unsupported initialization scope")
    dims = init["model_dimensions"]
    require(all(dims[k] == v for k, v in dict(nq=16, nv=15, nu=9, na=0, nmocap=0,
                neq=0, nuserdata=0, nplugin=0).items()), "Unsupported model dimensions")
    # The frozen MuJoCo 3.3.7 model has mjSTATE_INTEGRATION=8191, size 305.
    # These constants are intentionally scoped; changing the model needs review.
    require(init["state_spec"] == 8191 and init["integration_state_size"] == 305,
            "Unsupported integration state specification/size")
    state = vector(init["integration_state"], 305, "integration_state")
    require(number(init["time_s"], "initial time") == 0.0, "Initial time must be zero")
    values = {}
    for key, size in (("qpos", 16), ("qvel", 15), ("qacc_warmstart", 15), ("ctrl", 9)):
        values[key] = vector(init[key], size, key)
        close(values[key], vector(frozen["initial_" + key], size, "frozen " + key), key + " vs frozen")
        if "initial_" + key in summary:
            close(values[key], vector(summary["initial_" + key], size, "summary " + key), key + " vs summary")
    prefix = [0.0] + values["qpos"] + values["qvel"] + values["qacc_warmstart"] + values["ctrl"]
    close(state[:len(prefix)], prefix, "integration state prefix")
    require(all(v == 0.0 for v in state[len(prefix):]), "Nonzero initial applied/user state")
    software = init["software"]
    require(all(isinstance(software[k], dict) and software[k]
                for k in ("session", "planner", "task_observer", "input_policy")), "Incomplete software state")
    session = software["session"]
    require(all(key in session for key in ("phase", "q_hold", "goal", "plan", "start_step", "settle_step",
                "ready_since", "command_id", "completed_ids", "pending_id", "command_records", "generation",
                "closed_requested", "pending_gripper_closed", "notice", "jog_reference_completed_ids", "last_plan_kind")),
            "Incomplete session state")
    require(session["phase"] == "waiting_start" and session["command_id"] == 0 and
            session["plan"] is None and session["command_records"] == [], "Software state not fresh")
    require(session["closed_requested"] is False and session["pending_gripper_closed"] is None,
            "Unexpected initial gripper intent")
    require(session["start_step"] == session["settle_step"] == session["generation"] == 0 and
            session["ready_since"] is None and session["pending_id"] is None and
            session["completed_ids"] == session["jog_reference_completed_ids"] == [] and
            session["last_plan_kind"] is None, "Session counters are not initial")
    addresses = config["joint_qpos_addresses"]
    require(addresses == list(range(9)), "Unsupported joint address mapping")
    close(vector(session["q_hold"], 9, "q_hold"), [values["qpos"][j] for j in addresses], "initial held reference")
    planner = software["planner"]
    for key in ("origin_xyz", "target_xyz"):
        close(vector(planner[key], 3, key), vector(session["goal"], 3, "initial goal"), key + " vs initial goal")
    for key in ("initial_q9", "last_q9"):
        close(vector(planner[key], 9, key), session["q_hold"], key + " vs initial hold")
    orientation = vector(planner["orientation_wxyz"], 4, "planner orientation")
    require(abs(sum(v*v for v in orientation) - 1.) < 1e-8, "Nonunit planner orientation")
    require(planner["mode"] == "empty" and planner["payload"] is None and
            planner["allow_finger_cube"] is False and planner["grasp_confirmed"] is False, "Planner is not fresh")
    observer = software["task_observer"]
    require(observer["dt"] == summary["dt_s"] and observer["last_step"] is None and observer["attempt_id"] == 0,
            "Observer clock/counter is not initial")
    for key in ("manual_close_requested", "baseline_active", "grasp_hold_passed", "transfer_completed", "descent_requested",
                "descent_observed", "release_started", "retraction_requested", "retraction_observed", "placement_passed",
                "grasp_observed", "lift_observed", "release_had_verified_transport"):
        require(observer[key] is False, "Initial observer already has task flag: " + key)
    for key in ("capture_requested_step", "above_tool_z_m", "release_tool_z_m", "grasp_origin_xyz_m", "capture_origin_xyz_m"):
        require(observer[key] is None, "Initial observer already has capture: " + key)
    for key in ("stop_reasons", "attempt_history", "intent_history"):
        require(observer[key] == [], "Initial observer history is not empty")
    require(observer["milestones"] == {}, "Initial observer milestones are not empty")
    for key in ("contact_timer", "hold_timer", "above_timer", "placement_timer", "loss_timer"):
        timer = observer[key]
        require(timer["dt"] == summary["dt_s"] and timer["since"] is None and timer["maximum_s"] == 0,
                "Initial timer not empty: " + key)
    for key in ("tray_rim_above_floor_m", "above_rim_gap_m", "minimum_retract_rise_m", "minimum_carried_xy_displacement_m"):
        require(number(observer[key], key) > 0, "Missing positive observer configuration: " + key)
    policy = software["input_policy"]
    require(policy["armed"] is False and policy["need_release"] is True and policy["pending_keys"] == []
            and policy["saturation_s"] == 0 and policy["grip_baseline"] is None, "Input policy not fresh")
    return values


def read_plans(run, summary, dt):
    records = summary["command_records"]
    require(isinstance(records, list), "Missing command records")
    commands = {}
    for position, record in enumerate(records, 1):
        cid = integer(record["command_id"], "command id")
        require(cid == position, "Command ids must be consecutive")
        commands[cid] = dict(record)
    plans = json_lines(run / "jog_references.jsonl")
    for path in sorted((run / "command_references").glob("*.json")):
        plan = read_json(path)
        plans.append(dict(plan, command_id=int(path.stem)))
    require(len(plans) == len(commands), "Command plan count mismatch")
    mapping = {}
    previous_end = 0
    for plan in sorted(plans, key=lambda p: int(p["command_id"])):
        cid = integer(plan["command_id"], "plan command id")
        require(cid in commands and cid not in mapping and plan["accepted"] is True, "Invalid/duplicate accepted plan")
        command = commands[cid]
        require(plan["command"] == command["command"], "Command name differs from plan")
        start = integer(plan["start_step"], "plan start")
        require(start >= previous_end, "Overlapping or reversed accepted plans")
        require(len(plan["rows"]) >= 2, "Plan needs both endpoints")
        for offset, row in enumerate(plan["rows"]):
            require(integer(row["step"], "plan relative step") == offset and
                    abs(number(row["sim_s"], "plan time") - offset * dt) < 1e-8, "Plan time grid mismatch")
            vector(row["q_ref9"], 9, "planned reference")
            vector(row["dq_ref9"], 9, "planned reference velocity")
        end = start + len(plan["rows"]) - 1
        mapping[cid] = dict(command, start_step=start, end_step=end, rows=plan["rows"])
        previous_end = end
    return mapping


def check_actions(events, plans, total, dt):
    previous_step = 0
    starts = 0
    accepted_commands = {}
    pending = None
    counts = {}
    for serial, event in enumerate(events, 1):
        kind = event["kind"]
        require(kind in ("start", "request_jog", "gripper_intent", "gripper_commit"), "Unknown action event kind")
        require(integer(event["event_id"], "event id") == serial, "Action event ids not consecutive")
        step = integer(event["request_step"], "request step")
        require(previous_step <= step <= total, "Action events outside saved timeline or reversed")
        previous_step = step
        require(abs(number(event["request_sim_s"], "request time") - step * dt) < 1e-8, "Action event clock mismatch")
        require(type(event["accepted"]) is bool and isinstance(event["reason"], str) and event["reason"],
                "Missing action acceptance/reason")
        context = event["request_context"]
        require(isinstance(context, dict) and isinstance(context["phase"], str) and context["phase"],
                "Missing pre-request software context")
        vector(context["q_hold"], 9, "pre-request q_hold")
        vector(context["goal"], 3, "pre-request goal")
        require(type(context["closed_requested"]) is bool and
                (context["pending_gripper_closed"] is None or type(context["pending_gripper_closed"]) is bool) and
                context["planner_mode"] in ("empty", "closing", "held", "released") and
                integer(context["generation"], "pre-request generation") >= 0,
                "Invalid pre-request state fields")
        if "wall_s" in event:
            require(number(event["wall_s"], "wall time") >= 0.0, "Negative event wall time")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "start":
            if event["accepted"]:
                require(step == 0, "Accepted start not at initial physical state")
                starts += 1
            continue
        if kind == "gripper_intent":
            if event["accepted"]:
                require(type(event["desired_closed"]) is bool, "Gripper intent lacks explicit state")
                pending = event["desired_closed"]
            continue
        if kind == "gripper_commit":
            require(type(event["desired_closed"]) is bool and pending == event["desired_closed"],
                    "Gripper commit has no matching latest intent")
            pending = None
        if not event["accepted"]:
            continue
        cid = integer(event["command_id"], "accepted event command id")
        require(cid in plans and cid not in accepted_commands, "Accepted action has missing/duplicate command")
        plan = plans[cid]
        require(integer(event["effective_start_step"], "effective start") == step == plan["start_step"] and
                integer(event["effective_end_step_exclusive"], "effective end") == plan["end_step"],
                "Accepted action interval differs from reference plan")
        if kind == "request_jog":
            require(event["command"] in ("X+", "X-", "Y+", "Y-", "Z+", "Z-") and
                    event["command"] == plan["command"], "Jog direction differs from plan")
        else:
            require(plan["command"] == "GRIPPER_TARGET", "Gripper event refers to non-gripper plan")
            target = number(event["finger_target_m"], "finger target")
            require(target == plan["finger_target_m"] and target == (-.0065 if event["desired_closed"] else .0245),
                    "Gripper target differs from explicit requested state")
            close(plan["rows"][-1]["q_ref9"][7:], [target, target], "planned final finger target")
        accepted_commands[cid] = event["event_id"]
    require(starts <= 1 and (total == 0 or starts == 1), "Missing/repeated accepted start")
    require(set(accepted_commands) == set(plans), "Plan missing accepted action event")
    return dict(counts=counts, accepted_command_count=len(accepted_commands),
                pending_gripper_intent_at_end=pending,
                planned_segments_truncated_by_stop=[cid for cid, plan in plans.items() if plan["end_step"] > total])


def validate(run, index):
    summary = read_json(run / "summary.json")
    index["historical_result_from_summary"] = {key: summary.get(key) for key in (
        "status", "end_reason", "manual_task_completed", "grasp_hold_passed", "transfer_completed",
        "release_started", "placement_passed", "tracking_thresholds_passed")}
    index["task_completed"] = summary.get("manual_task_completed") is True
    require(summary["status"] in ("completed", "partial", "stopped", "error", "interrupted"), "Run has not ended")
    sources = source_inventory(run, summary)
    index["source_files_sha256"] = sources
    require(sources["manual.csv"] == summary["manual_csv_sha256"], "CSV hash differs from summary")
    require(sources["jog_references.jsonl"] == summary["jog_references_sha256"], "Jog hash differs from summary")
    require(sources["initialization.json"] == summary["initialization_sha256"], "Initialization hash differs from summary")
    require(sources["actions.jsonl"] == summary["actions_sha256"], "Action event hash differs from summary")
    require(all(name in sources for name in ("configs/config.json", "configs/frozen_context.json", "configs/scene.xml",
                "configs/runtime_integrity.json", "assets/manifest.json", "assets/vfs_map.json",
                "runtime/manual_pickplace.py", "collection_code/collect_episode.py", "collection_code/episode_state.py",
                "collection_code/episode_schema.py")), "Required runtime/configuration/collection provenance missing")
    for key in ("snapshots", "collection_code"):
        require(isinstance(summary[key], list) and summary[key], "Missing source snapshot list: " + key)
        require(len({item["path"] for item in summary[key]}) == len(summary[key]), "Duplicate source snapshot path")
        for item in summary[key]:
            require(sources[item["path"]] == item["sha256"], "Snapshot hash mismatch: " + item["path"])
    config = read_json(run / "configs/config.json")
    frozen = read_json(run / "configs/frozen_context.json")
    init = read_json(run / "initialization.json")
    semantic = summary.get("field_semantics") or read_json(run / "field-semantics.json")
    require(all(isinstance(semantic[key], str) and semantic[key] for key in SEMANTIC_KEYS), "Incomplete field semantics")
    index["field_semantics"] = semantic
    dt = number(summary["dt_s"], "dt")
    require(dt > 0 and dt == frozen["effective_solver_options"]["timestep"] ==
            summary["frozen_physics_options"]["timestep"], "Physics timestep mismatch")
    total = integer(summary["steps"], "steps")
    require(total >= 0, "Negative steps")
    index.update(dt_s=dt, action_count=total, state_count=total + 1, joint_order=config["joint_order"])
    require(len(config["joint_order"]) == 9, "Unsupported joint order")
    for key in ("kp", "kd"):
        close(vector(summary[key], 9, key), vector(frozen[key], 9, "frozen " + key), key)
    require(summary["software_caps"] == frozen["cap"], "Control caps mismatch")
    initial = check_initialization(init, frozen, summary, config)
    plans = read_plans(run, summary, dt)
    index["action_event_check"] = check_actions(json_lines(run / "actions.jsonl"), plans, total, dt)
    with (run / "manual.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV columns")
        rows = list(reader)
    require(len(rows) == total + 1 == summary["data_saved_samples"], "State/action count mismatch")
    previous_id = 0
    max_grid_error = 0.0
    for step, row in enumerate(rows):
        require(integer(row["step_id"], "state step") == step, "Missing/repeated state step")
        elapsed = number(row["sim_s"], "state time")
        grid_error = abs(elapsed - step * dt)
        max_grid_error = max(max_grid_error, grid_error)
        require(grid_error < 1e-7, "State time grid mismatch")
        require(integer(row["terminal_state"], "terminal flag") == int(step == total), "Invalid terminal flag")
        qpos, qvel = columns(row, "qpos", range(16)), columns(row, "qvel", range(15))
        close(columns(row, "q", range(9)), qpos[:9], "joint q vs qpos")
        close(columns(row, "qd", range(9)), qvel[:9], "joint qd vs qvel")
        for prefix in ("actual", "cube"):
            columns(row, prefix, (axis + "_m" for axis in "xyz"))
            quat = columns(row, prefix + "_quat", "wxyz")
            require(abs(sum(v*v for v in quat) - 1.0) < 1e-8, "Nonunit quaternion: " + prefix)
        close(columns(row, "cube_quat", "wxyz"), qpos[12:16], "cube orientation vs qpos")
        close(columns(row, "cube", (axis + "_m" for axis in "xyz")), qpos[9:12], "cube position vs qpos")
        close(columns(row, "cube_linear_velocity_world", "xyz"), qvel[9:12], "cube linear velocity")
        close(columns(row, "cube_angular_velocity_local", "xyz"), qvel[12:15], "cube angular velocity")
        columns(row, "actual_linear_velocity_world", "xyz")
        columns(row, "actual_angular_velocity_world", "xyz")
        qref = columns(row, "q_ref", range(9))
        close(qref[7:], [number(row["finger_ref_m"], "finger reference")] * 2, "finger reference")
        if step == 0:
            close(qpos, initial["qpos"], "first qpos vs initial")
            close(qvel, initial["qvel"], "first qvel vs initial")
            close(qref, init["software"]["session"]["q_hold"], "first reference vs initial")
        cid = integer(row["command_id"], "state command id")
        require(cid >= previous_id, "Command ids went backwards")
        if cid != previous_id:
            require(cid == previous_id + 1 and cid in plans and plans[cid]["start_step"] == step,
                    "Command transition not bound to plan start")
        previous_id = cid
        if cid:
            require(cid in plans, "Unknown command id in state")
            plan = plans[cid]
            offset = min(step - plan["start_step"], len(plan["rows"]) - 1)
            require(offset >= 0, "Command used before start")
            expected = plan["rows"][offset]
            close(qref, expected["q_ref9"], "executed reference vs plan")
        else:
            expected = {"q_ref9": init["software"]["session"]["q_hold"], "dq_ref9": [0.0] * 9}
            close(qref, expected["q_ref9"], "initial hold reference")
        if step < total:
            close(columns(row, "qd_ref", range(9)), expected["dq_ref9"], "reference velocity vs plan")
        else:
            require(all(row[f"qd_ref_{j}"] == "" for j in range(9)), "Terminal has future reference velocity")
            require(all(row[f"tau_{j}"] == "" for j in range(9)), "Terminal has future control torque")
            require(abs(elapsed - summary["simulation_s"]) < 1e-8, "Terminal time differs from summary")
    require(set(range(1, previous_id + 1)) == set(plans), "Accepted plan absent from saved states")
    for relative, before in sources.items():
        require(digest(local(run, relative)) == before, "Source changed during validation: " + relative)
    index.update(max_clock_grid_error_s=max_grid_error, snapshot_count=len(summary["snapshots"]),
                 initialization_sha256=sources["initialization.json"], actions_sha256=sources["actions.jsonl"],
                 terminal=dict(step_id=total, sim_s=number(rows[-1]["sim_s"], "terminal time"), has_next_action=False),
                 checks=["source_hashes", "frozen_initial_integration_and_software", "finite_complete_state_columns",
                         "consecutive_state_action_grid", "reference_matches_accepted_plan",
                         "explicit_intent_commit_intervals", "terminal_has_no_next_action", "sources_unchanged"])
    return rows


def finalize_episode(run):
    """Export a fresh episode subdirectory; return index even on data failure.

    Caller must finish writing summary/raw files first and leave them unchanged.
    A second call refuses to overwrite an existing episode archive.
    """
    run = Path(run).resolve(strict=True)
    output = run / "episode"
    output.mkdir(exist_ok=False)
    index = dict(format=FORMAT, result="EPISODE_CHECK_FAILED", source_run=str(run), episode_id=run.name,
                 created_at_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 complete_v02_episode=False, task_completed=False, dynamic_replay_executed=False,
                 replay_passed=None, validator_sha256=digest(Path(__file__)),
                 semantics="N executed pre-step references pair with N+1 states; planned segments may extend beyond an early stop",
                 scope="same_fixed_package_initial_state; file completeness is not task success")
    try:
        rows = validate(run, index)
        reference_file = output / "references.csv"
        with reference_file.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REFERENCE_COLUMNS)
            writer.writeheader()
            writer.writerows({name: row[name] for name in REFERENCE_COLUMNS} for row in rows[:-1])
        index.update(result="REFERENCE_TIMELINE_CHECK_PASSED", complete_v02_episode=True,
                     references_sha256=digest(reference_file))
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, OSError, csv.Error) as exc:
        index["validation_error"] = f"{type(exc).__name__}: {exc}"
    (output / "episode-index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    outcome = index.get("historical_result_from_summary", {})
    lines = ["# v0.2 新采集回合检查", "", f"回合：`{run.name}`。",
             f"完整格式：**{index['complete_v02_episode']}**；原任务成功：**{index['task_completed']}**。",
             f"原结果：`{outcome.get('status', 'unknown')}` / `{outcome.get('end_reason', 'unknown')}`。", "",
             "格式通过只表示已保存状态、参考、初态和请求/生效事件完整且相互一致；失败及提前结束也原样保留。",
             "本检查只读原件，不运行物理、不判定新的抓放成功、不填造缺失状态。动力学回放尚未执行。", "",
             f"状态数：{index.get('state_count', 'unknown')}；实际动作区间数：{index.get('action_count', 'unknown')}。",
             "终态没有下一步动作。已接受但未走完的计划保持原长度，只有终态之前的参考是实际执行记录。"]
    if index.get("validation_error"):
        lines.extend(["", "检查失败：`" + index["validation_error"] + "`。禁止使用不完整导出执行回放。"])
    else:
        lines.extend(["", "产物：references.csv、episode-index.json；原始记录和来源哈希留存。",
                      "逐步四元数使用 wxyz；工具速度使用世界坐标轴，方块角速度使用方块局部坐标轴。"])
    (output / "检查报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return index
