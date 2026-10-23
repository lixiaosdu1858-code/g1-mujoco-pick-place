"""Prepare derived G1 models and static FK audits; never step dynamics."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

from lesson_common import ROOT, guard, save_json, sha, stamp


REVISION = "905111699ef28bdeb1c69c6fe8ada8405e963a48"
JOINT = "right_elbow_joint"
ACTUATOR = "right_elbow"
SITE = "right_grasp_site"
HOME = 0.8
TORQUE_CAP = 6.0
OPEN_FINGER = 0.0245
ASSET_ROOT = ROOT / "repos" / "g1_grasp_assets"
OUTPUT = ROOT / "models" / "g1_fixed_base"


def vector(text, count):
    result = [float(value) for value in text.split()]
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError(f"Expected {count} finite values: {text!r}")
    return result


def require(root, query):
    result = root.find(query)
    if result is None:
        raise ValueError(f"Missing expected XML element: {query}")
    return result


def merged_source():
    robot = ET.parse(ASSET_ROOT / "assets/g1_29dof_with_grippers.xml").getroot()
    scene = ET.parse(ASSET_ROOT / "assets/scene_33dof.xml").getroot()
    includes = list(scene.iter("include"))
    if len(includes) != 1 or includes[0].get("file") != "g1_29dof_with_grippers.xml":
        raise ValueError("Unexpected source scene includes.")
    # Assets are read by Python and supplied through MuJoCo's VFS, avoiding
    # the Windows C file API's failure on Chinese absolute paths.
    require(robot, "compiler").set("meshdir", ".")
    option = require(robot, "option")
    option.set("timestep", "0.004")
    option.set("integrator", "implicitfast")
    option.set("gravity", "0 0 -9.81")
    for child in scene:
        if child.tag == "include":
            continue
        if child.tag in {"asset", "worldbody", "default"}:
            destination = require(robot, child.tag)
            destination.extend(deepcopy(list(child)))
        elif robot.find(child.tag) is None:
            robot.append(deepcopy(child))
        else:
            raise ValueError(f"Unexpected duplicate source section: {child.tag}")
    world = require(robot, "worldbody")
    marker = world.find("site[@name='com_marker']")
    if marker is not None:
        world.remove(marker)
    if robot.find("keyframe") is not None:
        raise ValueError("Source has keyframes; their qpos layout needs explicit migration.")
    return robot


def fix_pelvis(root):
    pelvis = require(root, "worldbody/body[@name='pelvis']")
    floating = require(pelvis, "joint[@name='floating_base_joint']")
    if floating.get("type") != "free":
        raise ValueError("Expected the original pelvis free joint.")
    pelvis.remove(floating)
    root.set("model", "g1_fixed_base_33dof")


def make_elbow_lesson(fixed):
    root = deepcopy(fixed)
    root.set("model", "g1_fixed_body_right_elbow_lesson")
    removed = {"floating_base_joint"}
    locked_fingers = []
    world = require(root, "worldbody")
    # Only joints in worldbody are removed; default-class joint properties stay.
    for body in world.iter("body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name == JOINT:
                continue
            if joint.get("type", "hinge") == "slide":
                if not name or "dex1_finger_joint_" not in name:
                    raise ValueError(f"Unexpected sliding joint: {name}")
                if any(key in body.attrib for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
                    raise ValueError(f"Finger body orientation requires explicit handling: {name}")
                if float(joint.get("ref", "0")) != 0.0:
                    raise ValueError(f"Unexpected finger reference displacement: {name}")
                axis = vector(joint.get("axis", "0 0 1"), 3)
                if not math.isclose(sum(value * value for value in axis), 1.0, abs_tol=1e-9):
                    raise ValueError(f"Non-unit finger axis: {name}")
                original = vector(body.get("pos", "0 0 0"), 3)
                shifted = [position + direction * OPEN_FINGER for position, direction in zip(original, axis)]
                body.set("pos", " ".join(f"{value:.12g}" for value in shifted))
                locked_fingers.append({"joint": name, "locked_displacement_m": OPEN_FINGER,
                                       "original_body_pos": original, "derived_body_pos": shifted})
            elif joint.get("type", "hinge") != "hinge":
                raise ValueError(f"Unexpected remaining joint type: {name}")
            removed.add(name)
            body.remove(joint)
    if len(locked_fingers) != 4:
        raise ValueError("Expected exactly four locked open finger joints.")
    actuators = require(root, "actuator")
    for motor in list(actuators):
        if motor.get("joint") != JOINT:
            actuators.remove(motor)
    if len(actuators) != 1 or actuators[0].get("name") != ACTUATOR:
        raise ValueError("Expected exactly one right-elbow motor.")
    # Keep the source motor transmission, with the lesson's smaller force cap.
    actuators[0].set("ctrlrange", "-6 6")
    actuators[0].set("ctrllimited", "true")
    actuators[0].set("forcerange", "-6 6")
    actuators[0].set("forcelimited", "true")
    sensors = root.find("sensor")
    if sensors is not None:
        for sensor in list(sensors):
            if sensor.get("joint") in removed or (sensor.get("objtype") == "joint" and sensor.get("objname") in removed):
                sensors.remove(sensor)
            elif "actuator" in sensor.attrib and sensor.get("actuator") != ACTUATOR:
                sensors.remove(sensor)
    grasp = require(root, f".//site[@name='{SITE}']")
    grasp.attrib.update({"type": "sphere", "size": "0.012", "rgba": "0.1 0.85 0.25 1", "group": "0"})
    # Locking the shoulder to world removes the engine's automatic adjacent-
    # body collision filter for this joint. Preserve that one upstream pair;
    # keep all other collision geometry and contact rules intact.
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    ET.SubElement(contact, 'exclude', {'body1': 'right_shoulder_yaw_link',
                                     'body2': 'right_elbow_link'})
    origin = [0.45, -0.45, 0.8]
    for index, (axis, color) in enumerate(zip("xyz", ("1 0.15 0.15 1", "0.15 1 0.15 1", "0.15 0.35 1 1"))):
        end = origin.copy()
        end[index] += 0.15
        ET.SubElement(world, "site", {"name": f"world_axis_{axis}", "type": "capsule", "size": "0.004",
                      "fromto": " ".join(str(value) for value in origin + end), "rgba": color, "group": "0"})
    ET.SubElement(world, "site", {"name": "fixed_pelvis_visual_support", "type": "capsule", "size": "0.025",
                  "fromto": "-0.12 0 0.08 -0.12 0 0.76", "rgba": "0.45 0.5 0.55 0.7", "group": "0"})
    return root, locked_fingers


def write_xml(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def model_size(model):
    return {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)}


def static_audit(mujoco, model):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT)
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE)
    if min(joint, actuator, site) < 0:
        raise ValueError("Lesson joint, actuator or site is missing.")
    if int(model.actuator_trnid[actuator, 0]) != joint:
        raise ValueError("Right-elbow actuator transmission does not reference the elbow joint.")
    active_body = int(model.jnt_bodyid[joint])

    def moving_body(body):
        while body:
            if body == active_body:
                return True
            body = int(model.body_parentid[body])
        return False

    points = []
    warnings = []
    for angle in (0.6, 0.8, 1.0):
        data = mujoco.MjData(model)
        for finger in ('left_dex1_finger_joint_1', 'left_dex1_finger_joint_2',
                       'right_dex1_finger_joint_1', 'right_dex1_finger_joint_2'):
            finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger)
            if finger_id >= 0:
                data.qpos[int(model.jnt_qposadr[finger_id])] = OPEN_FINGER
        data.qpos[int(model.jnt_qposadr[joint])] = angle
        mujoco.mj_forward(model, data)
        gravity_torque = float(data.qfrc_bias[int(model.jnt_dofadr[joint])])
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = [int(model.geom_bodyid[geom]) for geom in geoms]
            moving = any(moving_body(body) for body in bodies)
            contacts.append({
                "geom_ids": geoms,
                "geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) for geom in geoms],
                "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) for body in bodies],
                "distance_m": float(contact.dist), "position_world_m": contact.pos.tolist(),
                "normal_world": contact.frame[:3].tolist(), "contact_dimension": int(contact.dim),
                "touches_active_arm": moving,
            })
            if moving and float(contact.dist) <= 0.0:
                warnings.append(f"Active-arm collision at q={angle}: bodies={contacts[-1]['body_names']}, distance={float(contact.dist):.8g} m")
        if abs(gravity_torque) > TORQUE_CAP:
            warnings.append(f"Static gravity torque at q={angle} is {gravity_torque:.8g} Nm, above the {TORQUE_CAP} Nm cap.")
        if bool(model.jnt_limited[joint]) and not model.jnt_range[joint, 0] <= angle <= model.jnt_range[joint, 1]:
            warnings.append(f"Requested angle {angle} is outside the joint limits.")
        points.append({"q_rad": angle, "grasp_position_world_m": data.site_xpos[site].tolist(),
                       "qfrc_bias_Nm_at_zero_velocity": gravity_torque, "contacts": contacts,
                       "simulation_time_s": float(data.time)})
    return {
        "joint_id": joint, "joint_limited": bool(model.jnt_limited[joint]),
        "joint_range_rad": model.jnt_range[joint].tolist(),
        "joint_qpos_address": int(model.jnt_qposadr[joint]), "joint_qvel_address": int(model.jnt_dofadr[joint]),
        "actuator_id": actuator, "transmission_type": int(model.actuator_trntype[actuator]),
        "transmission_gear": model.actuator_gear[actuator].tolist(),
        "actuator_ctrlrange": model.actuator_ctrlrange[actuator].tolist(),
        "actuator_forcerange": model.actuator_forcerange[actuator].tolist(),
        "joint_actuatorfrcrange": model.jnt_actfrcrange[joint].tolist(),
        "points": points, "warnings": warnings,
    }


def main():
    guard()
    if ROOT.resolve().drive.casefold() != "e:":
        raise RuntimeError("Project root must resolve to E:.")
    audit = ROOT / "outputs" / ("audit_elbow_scene_" + stamp())
    audit.mkdir(parents=True, exist_ok=False)
    os.chdir(audit)
    report = {"status": "preparing_static_models", "audit_directory": str(audit), "physics_steps": 0,
              "scope": "MJCF construction, compilation and mj_forward only; no control or grasp success is claimed."}
    try:
        # Import only after cwd has moved to the audited E: directory, so an
        # engine diagnostic file cannot be emitted to another working directory.
        import mujoco

        report["mujoco_version"] = mujoco.__version__
        if mujoco.__version__ != "3.3.7":
            raise RuntimeError("This lesson was prepared for the registered MuJoCo 3.3.7 environment.")
        manifest_path = ASSET_ROOT / "asset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("revision") != REVISION or manifest.get("status") != "completed_asset_fetch_only":
            raise ValueError("The pinned asset fetch has not completed successfully.")
        report["asset_manifest_sha256"] = sha(manifest_path)
        records = {item["path"]: item for item in manifest["files"]}
        mesh_files = []
        for relative in ["assets/scene_33dof.xml", "assets/g1_29dof_with_grippers.xml"] + manifest["dependency_closure"]["meshes"]:
            path = ASSET_ROOT / relative
            if not path.resolve().is_relative_to(ASSET_ROOT.resolve()) or sha(path) != records[relative]["sha256"]:
                raise ValueError(f"Source file differs from its asset manifest: {relative}")
            if relative.lower().endswith(".stl"):
                mesh_files.append({"path": "repos/g1_grasp_assets/" + relative, "sha256": records[relative]["sha256"]})
        if len(mesh_files) != 39:
            raise ValueError("Expected 39 original mesh files.")
        vfs = {relative.removeprefix("assets/"): (ASSET_ROOT / relative).read_bytes()
               for relative in ["assets/g1_29dof_with_grippers.xml"] + manifest["dependency_closure"]["meshes"]}
        fixed = merged_source()
        fix_pelvis(fixed)
        elbow, finger_locks = make_elbow_lesson(fixed)
        fixed_path, elbow_path = audit / "g1_fixed_base.xml", audit / "elbow_lesson.xml"
        write_xml(fixed_path, fixed)
        write_xml(elbow_path, elbow)
        sources = (("original", ASSET_ROOT / "assets/scene_33dof.xml", (40, 39, 33)),
                   ("fixed_base", fixed_path, (33, 33, 33)),
                   ("elbow_lesson", elbow_path, (1, 1, 1)))
        report["models"] = {}
        lesson_model = None
        original_model = None
        for name, path, expected in sources:
            model = mujoco.MjModel.from_xml_string(path.read_text(encoding="utf-8"), assets=vfs)
            report["models"][name] = dict(model_size(model), path=str(path), sha256=sha(path))
            if (model.nq, model.nv, model.nu) != expected:
                raise ValueError(f"Unexpected model dimensions for {name}: {model_size(model)}")
            if name == "elbow_lesson":
                lesson_model = model
            if name == "original":
                original_model = model
        report["locked_fingers"] = finger_locks
        report["asset_loading"] = "Python Unicode paths plus MuJoCo in-memory VFS; no assets copied outside E:"
        report["forward_kinematics"] = static_audit(mujoco, lesson_model)
        report['original_reference_fk'] = static_audit(mujoco, original_model)
        upstream_pair = {'right_shoulder_yaw_link', 'right_elbow_link'}
        upstream_pair_contacts = [c for p in report['original_reference_fk']['points']
                                  for c in p['contacts'] if set(c['body_names']) == upstream_pair]
        if upstream_pair_contacts:
            raise RuntimeError('Original model also contacts the excluded adjacent pair; re-evaluate derivation.')
        report['adjacent_collision_preservation'] = {
            'excluded_pair': sorted(upstream_pair),
            'original_contacts_for_pair_at_three_poses': len(upstream_pair_contacts),
            'reason': 'After locking shoulders to world, restore the original automatic parent-child filter for this adjacent pair only.'}
        passed = not report["forward_kinematics"]["warnings"]
        report["static_checks_passed"] = passed
        config = {
            "joint": JOINT, "actuator": ACTUATOR, "home_rad": HOME,
            "kp_Nm_per_rad": 40, "kd_Nm_s_per_rad": 3, "torque_cap_Nm": TORQUE_CAP,
            "dt_s": 0.004, "max_wall_s": 300, "max_sim_s": 180,
            "key_targets_rad": {"7": 0.6, "8": 0.8, "9": 1.0},
            "scene_sha256": sha(elbow_path), "asset_manifest_sha256": sha(manifest_path),
            "mesh_files": mesh_files, "static_checks_passed": passed,
            "static_audit": str(audit / "static_audit.json"),
            "scope": "One active right elbow; other joints fixed, grippers held open; no dynamics tested by this builder.",
        }
        save_json(audit / "elbow_config.json", config)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name in ("g1_fixed_base.xml", "elbow_lesson.xml", "elbow_config.json"):
            destination = OUTPUT / name
            if destination.exists():
                previous = audit / "previous_outputs"
                previous.mkdir(exist_ok=True)
                shutil.copy2(destination, previous / name)
            shutil.copy2(audit / name, destination)
        report["output_directory"] = str(OUTPUT)
        report["status"] = "static_preparation_passed" if passed else "static_preparation_needs_review"
        save_json(audit / "static_audit.json", report)
        print("STATIC_AUDIT:", audit / "static_audit.json")
        print("ELBOW_CONFIG:", OUTPUT / "elbow_config.json")
        for warning in report["forward_kinematics"]["warnings"]:
            print("REVIEW_REQUIRED:", warning)
        print("No dynamics, rendering, window, controller or grasp validation was performed.")
        return 0 if passed else 2
    except Exception as error:
        report.update({"status": "failed_static_preparation", "error": str(error), "traceback": traceback.format_exc()})
        save_json(audit / "static_audit.json", report)
        print("STATIC_AUDIT:", audit / "static_audit.json", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
