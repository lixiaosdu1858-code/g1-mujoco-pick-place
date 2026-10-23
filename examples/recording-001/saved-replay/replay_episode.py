"""One user-started dynamic replay of a verified demonstration timeline.

Inspect mode never steps physics. Runtime uses frozen v0.1 control/observers;
only initialization sets actual positions. Recorded states are comparisons.
"""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import time
import traceback

from inspect_episode import digest, read_json, require

PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / "outputs/releases/g1-manual-pick-place-v0.1.0"
COMPLETE_FORMAT = "g1.demonstration.v0.2.0"
HELP = ("DYNAMIC REPLAY | FIXED BASE | SAVED JOINT REFERENCES\n"
        "7: start once   8: camera   9: stop/save\n"
        "Motion and gripper follow your recorded demonstration.\n"
        "Live PD feedback and MuJoCo physics produce the new motion.\n"
        "At the end physics pauses; inspect the scene, then press 9.")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def local(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Manifest path escaped root: " + relative)
    return path


def verify_input(index_path):
    index = read_json(index_path)
    require(index["result"] == "REFERENCE_TIMELINE_CHECK_PASSED", "Episode inspection did not pass")
    source = Path(index["source_run"]).resolve()
    require(source.is_relative_to(PROJECT / "outputs"), "Source run outside project")
    for relative, expected in index["source_files_sha256"].items():
        require(digest(local(source, relative)) == expected, "Source changed: " + relative)
    reference = index_path.parent / "references.csv"
    require(digest(reference) == index["references_sha256"], "Exported references changed")
    summary = read_json(source / "summary.json")
    require(summary["status"] == "completed" and summary["manual_task_completed"], "First replay requires the accepted demonstration")
    if index.get("complete_v02_episode") is True:
        require("initialization.json" in index["source_files_sha256"], "Complete episode initialization is not hash-bound")
        require(read_json(source / "initialization.json").get("format") == COMPLETE_FORMAT,
                "Unsupported complete episode initialization format")
        collection_code = summary.get("collection_code")
        require(isinstance(collection_code, list) and bool(collection_code), "Missing collection-code provenance")
        seen = set()
        state_helpers = []
        for item in collection_code:
            relative, expected = item["path"], item["sha256"]
            require(relative not in seen, "Duplicate collection-code provenance: " + relative)
            seen.add(relative)
            require(index["source_files_sha256"].get(relative) == expected,
                    "Collection code is not bound by episode index: " + relative)
            require(digest(local(source, relative)) == expected, "Collection code changed: " + relative)
            if Path(relative).name == "episode_state.py":
                state_helpers.append(expected)
        require(len(state_helpers) == 1, "Expected one recorded episode-state helper")
        require(digest(Path(__file__).with_name("episode_state.py")) == state_helpers[0],
                "Recorded state schema differs from the replay helper")
    # Imports come from the complete local package, whose runtime/configuration
    # must match the original run snapshots before any module is executed.
    for item in summary["snapshots"]:
        if item["path"] == "manual_initialization.json":
            continue
        require(digest(local(PACKAGE, item["path"])) == item["sha256"], "Package differs from recorded snapshot: " + item["path"])
    for item in read_json(PACKAGE / "assets/manifest.json")["files"]:
        require(digest(local(PACKAGE, item["package_path"])) == item["sha256"], "Mesh changed: " + item["package_path"])
    with reference.open(encoding="utf-8-sig", newline="") as stream:
        refs = list(csv.DictReader(stream))
    with (source / "manual.csv").open(encoding="utf-8-sig", newline="") as stream:
        states = list(csv.DictReader(stream))
    n = int(index["action_count"])
    require(len(refs) == n and len(states) == n + 1 and summary["steps"] == n, "Reference/state count mismatch")
    commands = {int(c["command_id"]): dict(c) for c in summary["command_records"]}
    with (source / "jog_references.jsonl").open(encoding="utf-8-sig") as stream:
        plans = [json.loads(line) for line in stream if line.strip()]
    for path in sorted((source / "command_references").glob("*.json")):
        plans.append(dict(read_json(path), command_id=int(path.stem)))
    for plan in plans:
        command = commands[int(plan["command_id"])]
        command.update(start_step=int(plan["start_step"]), end_step=int(plan["start_step"]) + len(plan["rows"]) - 1,
                       target_xyz=plan["target_xyz"])
    grippers = []
    previous = 0
    dt = float(index["dt_s"])
    for k, (ref, state) in enumerate(zip(refs, states)):
        require(int(ref["step_id"]) == k and abs(float(ref["sim_s"]) - k * dt) < 1e-8, "Invalid step grid")
        for prefix in ("q_ref", "qd_ref"):
            for j in range(9):
                require(ref[f"{prefix}_{j}"] == state[f"{prefix}_{j}"], "Export differs from original reference")
        cid = int(ref["command_id"])
        require(cid == int(state["command_id"]), "Command id mismatch")
        if cid != previous:
            require(cid in commands and commands[cid]["start_step"] == k, "Command transition not bound to a plan")
            require(commands[cid]["end_step"] <= n, "Plan beyond terminal state")
            if commands[cid]["command"] == "GRIPPER_TARGET":
                grippers.append({"step": k, "command": "close" if commands[cid]["finger_target_m"] < 0 else "open"})
            previous = cid
    require(grippers == [{"step": e["step"], "command": e["command"]} for e in summary["manual_gripper_intents"]],
            "Gripper commit timing differs from original intent log")
    require(len(plans) == len(commands), "Missing command plans")
    return index, source, summary, refs, states, commands


def serialized(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serialized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialized(v) for v in value]
    if hasattr(value, "__dict__"):
        return serialized(vars(value))
    return value


def make_session(manual, ctx, refs, states, commands, np):
    class ReplaySession(manual.ManualSession):
        def __init__(self):
            super().__init__(ctx, None, np)
            self.phase = "settling"
            self.notice = "RECORDED REFERENCES | 9: stop/save"

        def reference(self, step):
            require(0 <= step <= len(refs), "Reference index outside recorded sequence")
            row = states[step] if step == len(refs) else refs[step]
            return (np.array([float(row[f"q_ref_{j}"]) for j in range(9)]),
                    np.zeros(9) if step == len(refs) else np.array([float(row[f"qd_ref_{j}"]) for j in range(9)]))

        def advance_replay(self, step, previous_row):
            cid = int(refs[step]["command_id"])
            event = None
            if cid != self.command_id:
                command = commands[cid]
                self.command_id = cid
                self.pending_id = cid
                self.goal = np.asarray(command["target_xyz"], dtype=float)
                self.phase = "moving"
                self.ready_since = None
                self.command_records.append(command)
                if command["command"] == "GRIPPER_TARGET":
                    self.closed_requested = command["finger_target_m"] == manual.base.CLOSED
                    self.checks.notify_gripper(self.closed_requested, step, previous_row)
                    ctx["grip_baseline"] = None
                    event = "CLOSE" if self.closed_requested else "OPEN"
                elif command["command"] == "Z+" and self.checks.release_started:
                    self.checks.notify_retraction(step)
                elif command["command"] == "Z-" and self.checks.transfer_completed:
                    self.checks.notify_descent(step)
            if cid and step >= commands[cid]["end_step"] and self.phase == "moving":
                self.phase = "settling"
                self.settle_step = step
                self.ready_since = None
            return event

    return ReplaySession()


def extended_sample(manual, ctx, session, step, wall_s, mujoco, np):
    from episode_state import enrich_state
    row, problems, contacts = manual.sample(ctx, session, step, wall_s, mujoco, np)
    enrich_state(row, ctx, step, mujoco, np)
    return row, problems, contacts


def verify_complete_initialization(source, manual, ctx, mujoco, np):
    """The first complete format supports only the unchanged fixed initial state.

    Compare a fresh model and fresh software, never restore arbitrary recorded
    values or transfer the old success labels into the new task observer.
    """
    from episode_state import capture_initialization
    recorded = read_json(source / "initialization.json")
    planner = manual.TaskPlanner(ctx, mujoco, np)
    canonical_session = manual.ManualSession(ctx, planner, np)
    canonical = capture_initialization(manual, ctx, canonical_session, planner, mujoco, np)
    for key in ("format", "scope", "capture_point", "state_spec", "integration_state_size",
                "model_dimensions", "empty_fields", "time_s"):
        require(recorded.get(key) == canonical[key], "Complete initialization differs: " + key)
    differences = {}
    for key in ("integration_state", "qpos", "qvel", "ctrl", "qacc_warmstart"):
        expected = np.asarray(recorded[key], dtype=float)
        actual = np.asarray(canonical[key], dtype=float)
        require(expected.shape == actual.shape and np.isfinite(expected).all(),
                "Invalid complete initialization array: " + key)
        differences[key] = float(np.max(np.abs(expected - actual))) if actual.size else 0.0
        require(np.array_equal(expected, actual), "Complete initialization differs: " + key)
    require(recorded.get("software") == canonical["software"],
            "Complete software initialization differs from fresh manual session/planner/observer/input policy")
    return dict(status="verified_exactly", initialization_sha256=digest(source / "initialization.json"),
                state_spec=canonical["state_spec"], integration_state_size=len(canonical["integration_state"]),
                maximum_absolute_differences=differences, software_exact_match=True,
                physical_state_source="fresh_frozen_package_setup_verified_against_recorded_full_state",
                software_transition="manual waiting_start verified; replay enters settling when started; no old task labels restored")


def prepare(args, report, out):
    index, source, summary, refs, states, commands = verify_input(args.episode_index)
    report.update(source_run=str(source), episode_index=str(args.episode_index),
                  episode_index_sha256=digest(args.episode_index), source_summary_sha256=digest(source / "summary.json"),
                  reference_sha256=index["references_sha256"], planned_steps=len(refs),
                  dt_s=index["dt_s"], model_timestep_s=index["dt_s"],
                  source_files_sha256=index["source_files_sha256"],
                  source_episode_format=index.get("format"),
                  source_complete_v02_episode=index.get("complete_v02_episode") is True)
    sys.path.insert(0, str(PACKAGE / "runtime"))
    import mujoco
    import numpy as np
    manual = importlib.import_module("manual_pickplace")
    manual.guard()
    if args.inspect:
        def blocked_step(*a, **kw):
            raise RuntimeError("mj_step is prohibited in static inspection")
        mujoco.mj_step = blocked_step
    ctx = manual.setup(mujoco, np)
    report.update(model_loaded=True, mujoco_version=mujoco.__version__, numpy_version=np.__version__,
                  python_version=sys.version.split()[0], executable=sys.executable,
                  thresholds=manual.THRESHOLDS, placement_thresholds=manual.task.PLACEMENT_THRESHOLDS,
                  frozen_physics_options=manual.frozen.option_snapshot(ctx["model"], np))
    require(report["frozen_physics_options"] == summary["frozen_physics_options"], "Physics options differ")
    require(ctx["model"].opt.timestep == index["dt_s"] == manual.DT, "Timestep differs")
    m, d = ctx["model"], ctx["data"]
    require(m.na == 0 and m.nmocap == 0 and m.neq == 0 and m.nuserdata == 0 and m.nplugin == 0,
            "Unrecorded activation/mocap/equality/user/plugin state")
    require(np.array_equal(d.qpos, summary["initial_qpos"]) and np.array_equal(d.qvel, summary["initial_qvel"]), "Initial state differs")
    if index.get("complete_v02_episode") is True:
        report["complete_initialization_verification"] = verify_complete_initialization(source, manual, ctx, mujoco, np)
        report["source_collection_code"] = summary["collection_code"]
    session = make_session(manual, ctx, refs, states, commands, np)
    state_kind = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    initial_vector = np.empty(mujoco.mj_stateSize(m, state_kind))
    mujoco.mj_getState(m, d, initial_vector, state_kind)
    initialization = dict(kind="fresh_package_initialization_before_first_sample", time_s=float(d.time),
        model_dimensions={key: int(getattr(m, key)) for key in ("nq", "nv", "nu", "na", "nmocap", "neq", "nuserdata", "nplugin")},
        state_spec=state_kind, integration_state=initial_vector.tolist(),
        qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist(), qacc_warmstart=d.qacc_warmstart.tolist(),
        empty_fields={"act": "no actuator dynamics", "mocap": "no mocap bodies", "eq_active": "no equality constraints",
                      "userdata": "zero model dimension", "plugin_state": "no plugins"},
        software=dict(phase=session.phase, command_id=0, q_hold=session.q_hold.tolist(),
                      goal=session.goal.tolist(), closed_requested=False, grip_baseline=None,
                      task_observer=serialized(session.checks), saturation_s=0.0),
        limitations=("Recorded complete integration/software state verified exactly against fresh frozen setup; replay uses a reference scheduler instead of keyboard planning."
                     if index.get("complete_v02_episode") is True else
                     "Old recordings did not serialize a complete integration vector; fresh initialization is reconstructed from the frozen package and logged here."))
    require(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied) and d.time == 0, "Unexpected initial forces/time")
    save(out / "initialization.json", initialization)
    row, problems, _ = extended_sample(manual, ctx, session, 0, 0.0, mujoco, np)
    require(not problems, "Initial safety checks failed: " + str(problems))
    differences = {}
    for prefix in ("q_ref", "qd_ref", "q", "qd", "tau"):
        differences[prefix] = max(abs(row[f"{prefix}_{j}"] - float(states[0][f"{prefix}_{j}"])) for j in range(9))
    for prefix in ("actual", "cube", "ref"):
        differences[prefix] = max(abs(row[f"{prefix}_{axis}_m"] - float(states[0][f"{prefix}_{axis}_m"])) for axis in "xyz")
    require(max(differences.values()) <= 1e-9, "Initial sample differs: " + str(differences))
    require(d.time == 0 and np.array_equal(d.qpos, summary["initial_qpos"]) and np.array_equal(d.qvel, summary["initial_qvel"]), "Static preparation advanced state")
    # Timetable validation without using measured states as simulation input.
    check_session = make_session(manual, ctx, refs, states, commands, np)
    emitted = []
    for k in range(len(refs)):
        event = check_session.advance_replay(k, row)
        if event:
            emitted.append(dict(step=k, command=event.lower()))
    require(emitted == [{"step": e["step"], "command": e["command"]} for e in summary["manual_gripper_intents"]], "Replay gripper event scheduler differs")
    require(ctx["grip_baseline"] is None, "Static scheduling changed grip baseline")
    report["static_preparation"] = dict(status="passed_static_only", physics_steps=0,
        initial_sample_max_differences=differences, command_count=len(commands), gripper_events=emitted,
        sequence_states=len(states), sequence_actions=len(refs), integration_state_size=len(initial_vector),
        note="No dynamics, rendering or replay success verified")
    if "complete_initialization_verification" in report:
        report["static_preparation"]["complete_initialization_verification"] = report["complete_initialization_verification"]
    save(out / "static-preparation.json", report["static_preparation"])
    return manual, ctx, session, refs, states, row, mujoco, np


def run(prepared, report, out):
    manual, ctx, session, refs, states, initial, mujoco, np = prepared
    import mujoco.viewer
    m, d = ctx["model"], ctx["data"]
    commands = queue.SimpleQueue()
    started = time.perf_counter()
    physics_start = physics_end = None
    step = 0
    last = initial
    terminated = False
    finalized = False
    last_stage = None
    camera_index = 0
    saturation_s = 0.0
    options = report["frozen_physics_options"]
    metrics = dict(max_position_error_m=0.0, max_orientation_error_rad=0.0, max_joint_error_rad=0.0,
        max_torque_ratio=0.0, max_finger_force_ratio=0.0, max_unexpected_contact_count=0,
        max_held_relative_slip_m=0.0, max_held_relative_rotation_rad=0.0)
    events_path = out / "events.jsonl"

    def event(value):
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=True) + "\n")

    def write(row, terminal=False):
        nonlocal last
        if terminal:
            row = manual.base.motion.terminal_row(row)
            for key in row:
                if key.startswith("qd_ref_"):
                    row[key] = ""
        writer.writerow(row)
        stream.flush()
        last = row
        report["data_saved_samples"] += 1
        for metric, field in (("max_position_error_m", "position_error_m"), ("max_orientation_error_rad", "orientation_error_rad"),
                ("max_joint_error_rad", "max_joint_error_rad"), ("max_torque_ratio", "max_torque_ratio"),
                ("max_finger_force_ratio", "max_finger_force_ratio"), ("max_unexpected_contact_count", "unexpected_contact_count"),
                ("max_held_relative_slip_m", "relative_slip_m"), ("max_held_relative_rotation_rad", "relative_rotation_rad")):
            if row[field] != "":
                metrics[metric] = max(metrics[metric], row[field])

    def finish(reason, terminal=None, problems=None):
        nonlocal finalized, physics_end
        if finalized:
            return
        physics_end = time.perf_counter()
        if terminal is None:
            terminal, current_problems, contacts = extended_sample(manual, ctx, session, step, physics_end - started, mujoco, np)
            current_problems += manual.observe_task(ctx, session, terminal, step, mujoco, np)
            problems = list(problems or []) + current_problems
        write(terminal, True)
        tracking = bool(step > 0 and metrics["max_position_error_m"] <= manual.THRESHOLDS["position_warning_m"] and
                        metrics["max_orientation_error_rad"] <= manual.THRESHOLDS["orientation_warning_rad"])
        sequence = step == len(refs)
        passed = bool(reason == "reference_sequence_end" and sequence and tracking and not problems and
                      not session.checks.stop_reasons and session.checks.placement_passed and
                      terminal["placement_valid"] and terminal["placement_continuous_s"] >= 2.0 - 1e-10 and
                      session.phase == "ready")
        report.update(status="completed" if passed else ("stopped" if problems else "partial"), end_reason=reason,
            replay_completed=passed, sequence_completed=sequence, task_completed=passed,
            dynamic_replay_executed=step > 0,
            tracking_thresholds_passed=tracking, metrics=metrics, steps=step, simulation_s=float(d.time),
            physics_loop_wall_s=None if physics_start is None else physics_end - physics_start,
            final_phase=session.phase, stop_reasons=list(problems or []), **session.checks.summary(),
            final_qpos=d.qpos.tolist(), final_qvel=d.qvel.tolist(), final_qacc_warmstart=d.qacc_warmstart.tolist(),
            manual_csv_sha256=digest(out / "manual.csv"))
        report["physical_result"] = dict(status=report["status"], end_reason=reason,
            steps=step, sequence_completed=sequence, replay_completed=passed,
            task_completed=passed, tracking_thresholds_passed=tracking)
        save(out / "summary.json", report)
        finalized = True
        print("REPLAY RESULT:", report["status"], "|", reason, "| placement:", session.checks.placement_passed, flush=True)
        print("Physics paused. Press 9 to close and save; 8 still changes camera.", flush=True)

    # Callback only queues keys. All model changes stay on this thread under
    # the viewer lock; text is set once, live labels use the locked user scene.
    def callback(key):
        commands.put(int(key))

    first = dict(initial)
    first.update(session.checks.observe(first, 0)["row_fields"])
    with (out / "manual.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        try:
            with mujoco.viewer.launch_passive(m, d, key_callback=callback, show_left_ui=False, show_right_ui=False) as viewer:
                report["viewer_started"] = True
                with viewer.lock():
                    manual.set_camera(viewer.cam, ctx["cfg"], manual.frozen.CAMERAS[camera_index])
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, HELP, ""))
                print("Click scene, press 7 once. Recorded motion and gripper replay automatically. 8 camera; 9 stop/save.", flush=True)
                while viewer.is_running() and not terminated:
                    tick = time.perf_counter()
                    while not commands.empty():
                        key = commands.get()
                        event(dict(key=key, sim_s=float(d.time), wall_s=tick - started))
                        if key == ord("9"):
                            report["ui_end_reason"] = "user_9"
                            with viewer.lock():
                                finish("user_9")
                            terminated = True
                            break
                        if key == ord("8"):
                            camera_index = (camera_index + 1) % len(manual.frozen.CAMERAS)
                            with viewer.lock():
                                manual.set_camera(viewer.cam, ctx["cfg"], manual.frozen.CAMERAS[camera_index])
                        if key == ord("7") and physics_start is None and not finalized:
                            physics_start = time.perf_counter()
                            report["physics_started"] = True
                            print("REPLAY START |", len(refs), "steps |", len(refs) * manual.DT, "simulation seconds", flush=True)
                    if terminated:
                        break
                    with viewer.lock():
                        manual.frozen.assert_options_unchanged(m, options, np)
                        if physics_start is not None and not finalized:
                            for _ in range(4):
                                # Notify accepted gripper intent at the original
                                # commit step, before that step's observation.
                                old_id = session.command_id
                                if int(refs[step]["command_id"]) != old_id:
                                    current, problems, _ = extended_sample(manual, ctx, session, step, time.perf_counter() - started, mujoco, np)
                                    if problems:
                                        finish("pre_command_safety_stop", current, problems)
                                        break
                                    prior = current
                                else:
                                    prior = last
                                grip_event = session.advance_replay(step, prior)
                                if grip_event:
                                    print("RECORDED GRIPPER:", grip_event, f"| sim {d.time:.3f}s", flush=True)
                                    event(dict(kind="gripper_committed", command=grip_event, step=step, sim_s=float(d.time)))
                                row, problems, contacts = extended_sample(manual, ctx, session, step, time.perf_counter() - started, mujoco, np)
                                problems += manual.observe_task(ctx, session, row, step, mujoco, np)
                                saturation_s = saturation_s + manual.DT if row["max_torque_ratio"] > 1.0 else 0.0
                                if saturation_s > .25:
                                    problems.append("sustained_arm_saturation")
                                if session.phase == "settling" and (step - session.settle_step) * manual.DT > manual.THRESHOLDS["settling_timeout_s"]:
                                    problems.append("settling_timeout")
                                if problems:
                                    report["stop_diagnostics"] = dict(problems=problems, contacts=contacts)
                                    finish("safety_stop", row, problems)
                                    break
                                write(row)
                                mujoco.mj_step(m, d)
                                step += 1
                                report["steps"] = step
                                if step == len(refs):
                                    # Resolve the last segment's planned end, if
                                    # it coincides with the terminal observation.
                                    if session.phase == "moving":
                                        session.phase = "settling"
                                        session.settle_step = step
                                    finish("reference_sequence_end")
                                    break
                            if not finalized and time.perf_counter() - physics_start > manual.THRESHOLDS["maximum_physics_wall_s"]:
                                finish("wall_time_limit", problems=["wall_time_limit"])
                        session.notice = ("FINISHED - 9: close/save" if finalized else
                                          "7: START REPLAY" if physics_start is None else "REPLAYING - 9: stop/save")
                        manual.update_scene(viewer, ctx, session, last, mujoco, np)
                        before = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), float(d.time))
                    if session.checks.task_stage != last_stage:
                        print("TASK:", session.checks.task_stage, f"| sim {d.time:.3f}s", flush=True)
                        last_stage = session.checks.task_stage
                    viewer.sync()
                    with viewer.lock():
                        manual.frozen.assert_options_unchanged(m, options, np)
                        require(np.array_equal(d.qpos, before[0]) and np.array_equal(d.qvel, before[1]) and
                                np.array_equal(d.ctrl, before[2]) and d.time == before[3] and
                                not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied), "Viewer changed physical state/forces")
                    time.sleep(max(0.0, (.016 if physics_start is not None and not finalized else .025) - (time.perf_counter() - tick)))
                if not finalized:
                    finish("window_closed")
                report.setdefault("ui_end_reason", "window_closed")
        except BaseException:
            if not finalized:
                try:
                    finish("exception", problems=["exception"])
                except Exception:
                    report["terminal_record_error"] = traceback.format_exc()
            raise
    return 0 if report["status"] in ("completed", "partial") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    args.episode_index = args.episode_index.resolve(strict=True)
    out = args.output.resolve()
    require(out.parent == PROJECT / "outputs" and out.name.startswith("replay_") and not out.exists(), "Use a fresh flat outputs/replay_* directory")
    out.mkdir()
    report = dict(kind="g1_v02_dynamic_reference_replay", status="preparing", end_reason=None,
                  steps=0, simulation_s=0.0, data_saved_samples=0, model_loaded=False, viewer_started=False,
                  physics_started=False, dynamic_replay_executed=False, replay_completed=False,
                  sequence_completed=False, manual_task_completed=False, learner_verified=False,
                  replay_script_sha256=digest(Path(__file__)), scope="Local fixed-scene same-initial-state replay; no learned policy")
    shutil.copy2(__file__, out / "replay_episode.py")
    support = out / "replay_code"
    support.mkdir()
    report["replay_support_code"] = []
    for name in ("inspect_episode.py", "episode_state.py"):
        source = Path(__file__).with_name(name)
        destination = support / name
        shutil.copy2(source, destination)
        report["replay_support_code"].append(dict(path=destination.relative_to(out).as_posix(), sha256=digest(destination)))
    started = time.perf_counter()
    code = 1
    try:
        prepared = prepare(args, report, out)
        if args.inspect:
            report.update(status="prepared_static_only", end_reason="inspect_only")
            print("REPLAY PREPARATION: PASS; physics steps: 0; viewer: NOT OPENED", flush=True)
            code = 0
        else:
            code = run(prepared, report, out)
    except KeyboardInterrupt:
        report.update(status="interrupted", end_reason="KeyboardInterrupt")
        if not report.get("physical_result", {}).get("replay_completed", False):
            report["replay_completed"] = False
        code = 130
    except Exception:
        report.update(status="error", end_reason="exception", error=traceback.format_exc())
        if report.get("physical_result", {}).get("replay_completed", False):
            report["ui_error_after_physical_completion"] = report["error"]
        else:
            report["replay_completed"] = False
        (out / "error.txt").write_text(report["error"], encoding="utf-8")
        print(report["error"], file=sys.stderr)
    finally:
        report["dynamic_replay_executed"] = report["steps"] > 0
        report["wall_s"] = time.perf_counter() - started
        if (out / "manual.csv").exists():
            report["manual_csv_sha256"] = digest(out / "manual.csv")
        save(out / "summary.json", report)
        print("REPLAY RUN:", out, flush=True)
        print("SUMMARY:", out / "summary.json", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
