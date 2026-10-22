"""Build a two-joint G1 geometry lesson; static FK/Jacobian checks only.

This deliberately performs no physics integration, controller rollout, viewer
playback, or user learning experiment. The existing asset files are read-only.
"""

from copy import deepcopy
import json
import os
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

from lesson_common import ROOT, guard, save_json, sha, stamp
from build_elbow_scene import (ASSET_ROOT, REVISION, SITE, OPEN_FINGER,
                               merged_source, fix_pelvis, make_elbow_lesson,
                               require, write_xml)

JOINTS = ["right_shoulder_pitch_joint", "right_elbow_joint"]
ACTUATORS = ["right_shoulder_pitch", "right_elbow"]
HOME = [0.0, 0.8]
OUTPUT = ROOT / "models" / "g1_shoulder_elbow"


def make_shoulder_elbow_lesson(fixed):
    """Restore shoulder pitch in the exact existing elbow derivation."""
    root, locks = make_elbow_lesson(fixed)
    root.set("model", "g1_fixed_body_shoulder_elbow_kinematics")
    shoulder_body = require(root, ".//body[@name='right_shoulder_pitch_link']")
    shoulder_body.insert(1, deepcopy(require(fixed, f".//joint[@name='{JOINTS[0]}']")))
    # Restore original source motors, without implying dynamics were tested.
    actuators = require(root, "actuator")
    for child in list(actuators):
        actuators.remove(child)
    for name in ACTUATORS:
        actuators.append(deepcopy(require(fixed, f"actuator/motor[@name='{name}']")))
    sensors = root.find("sensor")
    if sensors is not None:
        source_sensors = fixed.find("sensor")
        for sensor in source_sensors:
            if sensor.get("joint") == JOINTS[0]:
                sensors.append(deepcopy(sensor))
    contact = require(root, "contact")
    # The elbow's welded parent is now a moving shoulder body, so its normal
    # parent-child filter works again. The shoulder's welded parent is world;
    # restore only that original adjacent pair instead.
    for exclude in list(contact.findall("exclude")):
        if {exclude.get("body1"), exclude.get("body2")} == {
                "right_shoulder_yaw_link", "right_elbow_link"}:
            contact.remove(exclude)
    ET.SubElement(contact, "exclude", {"body1": "torso_link",
                                      "body2": "right_shoulder_pitch_link"})
    return root, locks


def checked_assets():
    manifest_path = ASSET_ROOT / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("revision") != REVISION or manifest.get("status") != "completed_asset_fetch_only":
        raise ValueError("Pinned asset manifest is not complete or has changed revision.")
    records = {item["path"]: item for item in manifest["files"]}
    relative_paths = ["assets/scene_33dof.xml", "assets/g1_29dof_with_grippers.xml"]
    relative_paths += manifest["dependency_closure"]["meshes"]
    mesh_files, vfs = [], {}
    for relative in relative_paths:
        path = ASSET_ROOT / relative
        if not path.resolve().is_relative_to(ASSET_ROOT.resolve()) or sha(path) != records[relative]["sha256"]:
            raise ValueError("Asset changed: " + relative)
        if relative != "assets/scene_33dof.xml":
            vfs[relative.removeprefix("assets/")] = path.read_bytes()
        if relative.lower().endswith(".stl"):
            mesh_files.append({"path": "repos/g1_grasp_assets/" + relative,
                               "sha256": records[relative]["sha256"]})
    if len(mesh_files) != 39:
        raise ValueError("Expected the pinned 39 meshes.")
    return manifest_path, mesh_files, vfs


def main():
    guard()
    audit = ROOT / "outputs" / ("shoulder_model_audit_" + stamp())
    audit.mkdir(parents=True, exist_ok=False)
    os.chdir(audit)
    report = {"status": "preparing_static_model", "physics_steps": 0,
              "scope": "Asset/model and local kinematics preparation only; no dynamics or user experiment.",
              "script_sha256": sha(__file__),
              "derivation_dependency_sha256": sha(ROOT / "scripts/grasp/build_elbow_scene.py"),
              "python_executable": sys.executable, "temp": os.environ["TEMP"],
              "audit_directory": str(audit)}
    try:
        import mujoco
        import numpy as np
        if mujoco.__version__ != "3.3.7":
            raise RuntimeError("Use the registered MuJoCo 3.3.7 environment.")
        report["mujoco_version"] = mujoco.__version__
        manifest_path, meshes, vfs = checked_assets()
        fixed = merged_source()
        fix_pelvis(fixed)
        scene, locks = make_shoulder_elbow_lesson(fixed)
        scene_path = audit / "shoulder_elbow.xml"
        write_xml(scene_path, scene)
        model = mujoco.MjModel.from_xml_string(scene_path.read_text(encoding="utf-8"), assets=vfs)
        if (model.nq, model.nv, model.nu) != (2, 2, 2):
            raise ValueError("The teaching model must contain exactly two generalized coordinates and motors.")
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE)
        if joint_ids != [0, 1] or site < 0:
            raise ValueError("Unexpected joint ordering or missing hand site.")
        if list(model.jnt_qposadr) != [0, 1] or list(model.jnt_dofadr) != [0, 1]:
            raise ValueError("Unexpected coordinate addresses.")
        ranges = model.jnt_range[joint_ids].copy()
        report["joint_metadata"] = [
            {"name": name, "joint_id": int(j), "qpos_address": int(model.jnt_qposadr[j]),
             "qvel_address": int(model.jnt_dofadr[j]), "axis_body": model.jnt_axis[j].tolist(),
             "range_rad": model.jnt_range[j].tolist(), "limited": bool(model.jnt_limited[j]),
             "actuator_name": ACTUATORS[k], "actuator_gear": model.actuator_gear[k].tolist(),
             "actuator_ctrlrange": model.actuator_ctrlrange[k].tolist()}
            for k, (name, j) in enumerate(zip(JOINTS, joint_ids))]
        report["model_dimensions"] = {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)}

        def evaluate(q):
            q = np.asarray(q, dtype=float)
            if not np.all(np.isfinite(q)) or np.any(q < ranges[:, 0]) or np.any(q > ranges[:, 1]):
                raise ValueError("Static configuration is outside source joint ranges.")
            data = mujoco.MjData(model)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, site)
            contacts = []
            for contact in data.contact[:data.ncon]:
                geoms = [int(contact.geom1), int(contact.geom2)]
                contacts.append({"geom_ids": geoms,
                                 "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                    int(model.geom_bodyid[g])) for g in geoms],
                                 "distance_m": float(contact.dist)})
            return {"q_rad": q.tolist(), "hand_position_world_m": data.site_xpos[site].tolist(),
                    "jacobian_xyz_m_per_rad": jac.tolist(),
                    "joint_axes_world": data.xaxis[joint_ids].tolist(),
                    "joint_anchors_world_m": data.xanchor[joint_ids].tolist(),
                    "contacts": contacts, "ncon": int(data.ncon),
                    "warnings": [int(w.number) for w in data.warning], "simulation_time_s": float(data.time)}

        home = evaluate(HOME)
        jac = np.asarray(home["jacobian_xyz_m_per_rad"])
        jxz = jac[[0, 2], :]
        eps = 1e-6
        numerical = np.column_stack([
            (np.asarray(evaluate(np.asarray(HOME) + np.eye(2)[k] * eps)["hand_position_world_m"]) -
             np.asarray(evaluate(np.asarray(HOME) - np.eye(2)[k] * eps)["hand_position_world_m"])) / (2 * eps)
            for k in range(2)])
        report["home"] = home
        report["jacobian_validation"] = {
            "xz_m_per_rad": jxz.tolist(), "xz_rank": int(np.linalg.matrix_rank(jxz)),
            "xz_condition_number_2": float(np.linalg.cond(jxz)),
            "xz_determinant": float(np.linalg.det(jxz)),
            "centered_difference_step_rad": eps,
            "centered_difference_xyz_m_per_rad": numerical.tolist(),
            "max_absolute_difference_m_per_rad": float(np.max(np.abs(numerical - jac)))}
        report["small_configuration_probes"] = [evaluate(np.asarray(HOME) + np.eye(2)[k] * sign * .03)
                                                 for k in range(2) for sign in (-1, 1)]
        elbow_path = ROOT / "models/g1_fixed_base/elbow_lesson.xml"
        elbow_model = mujoco.MjModel.from_xml_string(elbow_path.read_text(encoding="utf-8"), assets=vfs)
        elbow_data = mujoco.MjData(elbow_model)
        elbow_data.qpos[:] = .8
        mujoco.mj_forward(elbow_model, elbow_data)
        elbow_site = mujoco.mj_name2id(elbow_model, mujoco.mjtObj.mjOBJ_SITE, SITE)
        difference = np.asarray(home["hand_position_world_m"]) - elbow_data.site_xpos[elbow_site]
        report["previous_elbow_home_comparison"] = {
            "scene": str(elbow_path), "scene_sha256": sha(elbow_path),
            "previous_world_position_m": elbow_data.site_xpos[elbow_site].tolist(),
            "delta_world_m": difference.tolist(), "max_absolute_difference_m": float(np.max(np.abs(difference)))}

        # Three independent static poses bound a proposed 2 cm local exercise.
        # This is a preparation check, not the user's sampled FK/IK experiment.
        candidates = []
        for dx in (0.0, 0.01, 0.02):
            target = np.asarray(home["hand_position_world_m"])[[0, 2]] + [dx, 0.0]
            q = np.asarray(HOME).copy()
            for iteration in range(20):
                point = evaluate(q)
                error = target - np.asarray(point["hand_position_world_m"])[[0, 2]]
                if np.max(np.abs(error)) < 1e-10:
                    break
                local_j = np.asarray(point["jacobian_xyz_m_per_rad"])[[0, 2], :]
                dq = np.linalg.solve(local_j, error)
                q += dq * min(1.0, .05 / max(np.max(np.abs(dq)), 1e-15))
            point = evaluate(q)
            point.update({"requested_dx_m": dx, "requested_dz_m": 0.0,
                          "xz_error_m": (np.asarray(point["hand_position_world_m"])[[0, 2]] - target).tolist(),
                          "unconstrained_delta_y_m": point["hand_position_world_m"][1] - home["hand_position_world_m"][1],
                          "iterations": iteration,
                          "xz_condition_number_2": float(np.linalg.cond(np.asarray(point["jacobian_xyz_m_per_rad"])[[0, 2], :]))})
            candidates.append(point)
        report["candidate_static_poses"] = candidates
        report["candidate_scope"] = "Only X and Z positions constrained; Y and gripper orientation remain unconstrained. Three checked poses do not prove collision-free motion between them."
        original = mujoco.MjModel.from_xml_string((ASSET_ROOT / "assets/scene_33dof.xml").read_text(encoding="utf-8"), assets=vfs)
        pair = {"torso_link", "right_shoulder_pitch_link"}
        original_pair_contacts = []
        for point in [home] + report["small_configuration_probes"] + candidates:
            data = mujoco.MjData(original)
            for name, qvalue in zip(JOINTS, point["q_rad"]):
                j = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[int(original.jnt_qposadr[j])] = qvalue
            for finger in ("left_dex1_finger_joint_1", "left_dex1_finger_joint_2", "right_dex1_finger_joint_1", "right_dex1_finger_joint_2"):
                j = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, finger)
                data.qpos[int(original.jnt_qposadr[j])] = OPEN_FINGER
            mujoco.mj_forward(original, data)
            for contact in data.contact[:data.ncon]:
                bodies = {mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_BODY,
                          int(original.geom_bodyid[int(g)])) for g in (contact.geom1, contact.geom2)}
                if bodies == pair:
                    original_pair_contacts.append({"q_rad": point["q_rad"], "distance_m": float(contact.dist)})
        report["adjacent_collision_preservation"] = {"excluded_pair": sorted(pair),
            "reason": "Restore this original parent-child filtering pair after welding the torso to world.",
            "original_contacts_for_excluded_pair_at_checked_poses": original_pair_contacts}
        points = [home] + report["small_configuration_probes"] + candidates
        checks = {
            "home_xz_jacobian_rank_two": np.linalg.matrix_rank(jxz) == 2,
            "home_xz_condition_below_100": np.linalg.cond(jxz) < 100,
            "jacobian_matches_centered_difference": np.max(np.abs(numerical - jac)) < 1e-7,
            "home_matches_previous_elbow_model": np.max(np.abs(difference)) < 1e-10,
            "all_static_times_zero": all(p["simulation_time_s"] == 0 for p in points),
            "no_static_warnings": all(not any(p["warnings"]) for p in points),
            "no_contacts_at_checked_poses": all(p["ncon"] == 0 for p in points),
            "candidate_xz_error_below_1e_minus_8_m": all(max(abs(v) for v in p["xz_error_m"]) < 1e-8 for p in candidates),
            "source_does_not_contact_excluded_adjacent_pair": not original_pair_contacts}
        report["checks"] = {name: bool(value) for name, value in checks.items()}
        passed = all(checks.values())
        report.update({"status": "static_preparation_passed" if passed else "static_preparation_needs_review",
                       "static_checks_passed": bool(passed), "scene_sha256": sha(scene_path),
                       "asset_manifest_sha256": sha(manifest_path), "locked_fingers": locks})
        config = {"joints": JOINTS, "actuators": ACTUATORS, "home_rad": HOME, "site": SITE,
                  "scene_sha256": sha(scene_path), "asset_manifest_sha256": sha(manifest_path),
                  "mesh_files": meshes, "static_checks_passed": bool(passed),
                  "static_audit": str(audit / "static_audit.json"),
                  "scope": "Kinematics only; physics integration, joint torque control and grasping have not been tested.",
                  "proposed_world_dx_m": 0.02, "joint_ranges_rad": ranges.tolist(),
                  "unconstrained": ["world Y", "hand orientation"], "source_revision": REVISION}
        save_json(audit / "config.json", config)
        save_json(audit / "static_audit.json", report)
        if passed:
            if OUTPUT.exists():
                raise FileExistsError("Refusing to overwrite the existing two-joint model directory.")
            OUTPUT.mkdir(parents=True, exist_ok=False)
            for name in ("shoulder_elbow.xml", "config.json"):
                shutil.copy2(audit / name, OUTPUT / name)
        print("STATIC_AUDIT:", audit / "static_audit.json")
        print("STATUS:", report["status"])
        print("HOME_J_XZ:", jxz.tolist())
        print("HOME_CONDITION:", float(np.linalg.cond(jxz)))
        print("No mj_step, control experiment or viewer playback was executed.")
        return 0 if passed else 2
    except Exception as error:
        report.update({"status": "failed_static_preparation", "error": str(error), "traceback": traceback.format_exc()})
        save_json(audit / "static_audit.json", report)
        print("STATIC_AUDIT:", audit / "static_audit.json", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
