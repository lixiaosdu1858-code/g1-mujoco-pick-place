"""Contact measurement helpers and bounded, zero-step grasp geometry audit.

Static closure intersects prescribed rigid configurations deliberately. Such
contact forces are NOT physical grasp evidence. Only a user's integrated run
may use contact_metrics to assess real closure and retention.
"""
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile

from lesson_common import ROOT, guard, save_json, sha, stamp

FINGER_GEOM_NAMES = ('right_dex1_finger_1_collision', 'right_dex1_finger_2_collision')
OPEN_Q_M = .0245
CLOSE_Q_M = -.0065


def contact_context(model, mujoco, np):
    """Resolve frozen scene names; no state mutation and no integration."""
    def gid(name):
        value = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if value < 0:
            raise ValueError('Missing contact geometry: '+name)
        return value
    fingers = [gid(name) for name in FINGER_GEOM_NAMES]
    meshes = []
    for g in fingers:
        if model.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError('Expected existing collision mesh, not a new pad.')
        mid = int(model.geom_dataid[g])
        first, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        meshes.append((g, model.mesh_vert[first:first+count].astype(float).copy()))
    cube = gid('task_cube_geom')
    if model.geom_type[cube] != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError('Cube lowest point assumes the frozen box geometry.')
    return dict(finger_geom_ids=fingers, finger_meshes=meshes, cube_geom_id=cube,
                cube_body_id=int(model.geom_bodyid[cube]), table_geom_id=gid('task_table_top'),
                tray_geom_ids=[gid('tray_floor')]+[gid('tray_wall_'+str(i)) for i in range(4)])


def cube_bottom_z(model, data, context, np):
    """Exact box minimum world Z, including current rotation and geom offset."""
    gid = context['cube_geom_id']
    rotation = data.geom_xmat[gid].reshape(3, 3)
    return float(data.geom_xpos[gid, 2]-np.abs(rotation[2]) @ model.geom_size[gid])


def contact_metrics(model, data, context, mujoco, np):
    """Measure CURRENT solved contacts, without calling forward or step.

    Caller must refresh dynamics/constraints (e.g. mj_forward with current
    q/qdot/ctrl) at the intended sample instant. Scalar normal force sums are
    separate from motor forces. Contact normals/wrenches use MuJoCo's contact
    frame. A geometric contact can exist while its solved normal force is zero.
    """
    cube = context['cube_geom_id']
    fingers = context['finger_geom_ids']
    counts = [0, 0]
    forces = [0., 0.]
    table_count, table_normal = 0, 0.
    contacts, unexpected = [], []
    for ci in range(data.ncon):
        c = data.contact[ci]
        pair = {int(c.geom1), int(c.geom2)}
        finger = next((i for i, g in enumerate(fingers) if pair == {cube, g}), None)
        on_table = pair == {cube, context['table_geom_id']}
        if finger is not None or on_table:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, wrench)
            normal = max(0., float(wrench[0]))
            row = dict(index=ci, finger_index=finger, cube_table=on_table,
                       distance_m=float(c.dist), normal_force_n=normal,
                       position_world_m=c.pos.tolist(), normal_axis_world=c.frame[:3].tolist(),
                       efc_address=int(c.efc_address), contact_friction=c.friction.tolist())
            contacts.append(row)
            if finger is not None:
                counts[finger] += 1
                forces[finger] += normal
            else:
                table_count += 1
                table_normal += normal
        elif c.dist <= 0:
            unexpected.append(dict(geoms=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                                          for g in (int(c.geom1), int(c.geom2))], distance_m=float(c.dist)))
    return dict(finger_contact_counts=counts, finger_normal_forces_n=forces,
                cube_table_contact_count=table_count, cube_table_normal_force_n=table_normal,
                unexpected_contacts=unexpected, contacts=contacts)


def box_vertices(model, data, gid, np):
    local = np.asarray(list(itertools.product((-1., 1.), repeat=3))) * model.geom_size[gid]
    return local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]


def separated_box_lower_bound(vertices, model, data, box_gid, np):
    """Conservative distance lower bound from projection onto box's axes.

    Positive proves separation of entire vertex convex hull and box. Zero is
    inconclusive, never a no-collision certificate.
    """
    local = (vertices-data.geom_xpos[box_gid]) @ data.geom_xmat[box_gid].reshape(3, 3)
    axis_gaps = np.maximum(-model.geom_size[box_gid]-local.max(axis=0),
                          local.min(axis=0)-model.geom_size[box_gid])
    return float(np.linalg.norm(np.maximum(axis_gaps, 0.)))


def _audit(out, mujoco, np):
    from reachable_motion import setup, CAP, KP
    ctx = setup(mujoco, np)
    m = ctx['model']
    # Every audit operation gets independent data; runtime context stays at A.
    d = mujoco.MjData(m)
    cc = contact_context(m, mujoco, np)
    cfg = ctx['cfg']
    table_z = cfg['table']['top_z_m']
    B = np.asarray(ctx['points']['B']['q_arm_rad'])
    def state(finger_q, q_arm=B):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = ctx['base_qpos']
        d.qpos[ctx['arm_adr']] = q_arm
        d.qpos[ctx['finger_adr']] = finger_q
        mujoco.mj_forward(m, d)
    state(OPEN_Q_M)
    site_R0 = d.site_xmat[ctx['site']].reshape(3, 3).copy()
    p0 = d.site_xpos[ctx['site']].copy()
    cube_g = cc['cube_geom_id']
    cube_rel = site_R0.T @ (d.geom_xpos[cube_g]-p0)
    cube_rel_R = site_R0.T @ d.geom_xmat[cube_g].reshape(3, 3)
    initial_cube_qpos = d.qpos[ctx['cube_adr']:ctx['cube_adr']+7].copy()
    closure = []
    for qf in np.linspace(OPEN_Q_M, CLOSE_Q_M, 125):
        state(float(qf))
        cm = contact_metrics(m, d, cc, mujoco, np)
        closure.append(dict(finger_q_m=float(qf), finger_contact_counts=cm['finger_contact_counts'],
                            contact_distances_m=[r['distance_m'] for r in cm['contacts'] if r['finger_index'] is not None],
                            min_finger_table_gap_m=min(float((v @ d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
                                                       for g,v in cc['finger_meshes'])-table_z,
                            unexpected_contacts=cm['unexpected_contacts']))
    contact_q = []
    # Binary locate onset for each finger; prescribed pose only, not execution.
    for i in range(2):
        clear, penetrating = OPEN_Q_M, CLOSE_Q_M
        for _ in range(45):
            mid = (clear+penetrating)/2
            state(mid)
            cm = contact_metrics(m, d, cc, mujoco, np)
            present = any(r['finger_index']==i and r['distance_m']<=0 for r in cm['contacts'])
            if present: penetrating = mid
            else: clear = mid
        contact_q.append((clear+penetrating)/2)
    state(min(contact_q)-1e-6)
    patches = []
    cm = contact_metrics(m, d, cc, mujoco, np)
    for r in cm['contacts']:
        if r['finger_index'] is None: continue
        patches.append(dict(finger_index=r['finger_index'],
                            position_in_cube_m=((np.asarray(r['position_world_m'])-d.geom_xpos[cube_g]) @ d.geom_xmat[cube_g].reshape(3,3)).tolist(),
                            normal_axis_world=r['normal_axis_world'],
                            distance_m=r['distance_m'], effective_friction=r['contact_friction']))
    # Reuse EXACT saved B_to_A arm configurations; no re-solving or new route.
    lift_rows = [r for r in ctx['rows'] if 8.-1e-10 <= r['sim_s'] <= 12.+1e-10]
    route = []
    # Check closure-reference geometry (overlap expected at fingers), and
    # tangent geometry. Cube transform is a HYPOTHESIS, not a dynamic constraint.
    for qf_kind, qf in [('tangent_estimate', min(contact_q)), ('closed_reference', CLOSE_Q_M)]:
        for r in lift_rows:
            state(qf, np.asarray(r['q_ref']))
            R = d.site_xmat[ctx['site']].reshape(3,3).copy()
            p = d.site_xpos[ctx['site']].copy()
            cube_pos = p+R@cube_rel
            cube_R = R@cube_rel_R
            cube_quat = np.empty(4)
            mujoco.mju_mat2Quat(cube_quat, cube_R.ravel())
            a = ctx['cube_adr']
            d.qpos[a:a+3] = cube_pos
            d.qpos[a+3:a+7] = cube_quat
            mujoco.mj_forward(m,d)
            verts = box_vertices(m,d,cube_g,np)
            finger_verts = [v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g,v in cc['finger_meshes']]
            unexpected = contact_metrics(m,d,cc,mujoco,np)['unexpected_contacts']
            cube_gap = cube_bottom_z(m,d,cc,np)-table_z
            finger_gap = min(float(v[:,2].min()) for v in finger_verts)-table_z
            tray_gap = min(separated_box_lower_bound(v,m,d,g,np) for v in [verts]+finger_verts for g in cc['tray_geom_ids'])
            route.append(dict(source_sim_s=r['sim_s'], finger_geometry=qf_kind,
                              hypothetical_cube_bottom_table_gap_m=cube_gap,
                              min_finger_table_gap_m=finger_gap,
                              min_cube_or_finger_tray_separation_lower_bound_m=tray_gap,
                              unexpected_contacts=unexpected, hand_position_world_m=p.tolist()))
    jids = [int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)) for n in cfg['finger_joint_order']]
    result = dict(kind='grasp_contact_geometry_preparation', physics_steps=0, simulation_s=0.,
                  geometry_only=True, grasp_validated=False, no_viewer=True,
                  executable=sys.executable,temp=tempfile.gettempdir(),
                  source_scene=str(ctx['folder']/'scene.xml'),source_scene_sha256=sha(ctx['folder']/'scene.xml'),
                  source_config_sha256=sha(ctx['folder']/'config.json'),
                  source_reference=str(ctx['reference_file']),source_reference_sha256=sha(ctx['reference_file']),
                  code_sha256=sha(Path(__file__)),
                  finger_axes=[m.jnt_axis[j].tolist() for j in jids],
                  finger_ranges_m=[m.jnt_range[j].tolist() for j in jids],
                  open_q_m=OPEN_Q_M,close_reference_q_m=CLOSE_Q_M,
                  software_finger_force_caps_n=CAP[-2:],finger_kp_n_m=KP[-2:],
                  finger_geom_ids=cc['finger_geom_ids'], cube_geom_id=cube_g,
                  table_geom_id=cc['table_geom_id'],tray_geom_ids=cc['tray_geom_ids'],
                  finger_friction=[m.geom_friction[g].tolist() for g in cc['finger_geom_ids']],
                  cube_friction=m.geom_friction[cube_g].tolist(),
                  cube_mass_kg=float(m.body_mass[cc['cube_body_id']]),
                  zero_velocity_contact_onset_q_m=contact_q,
                  geometric_overclosure_after_onset_m=[q-CLOSE_Q_M for q in contact_q],
                  estimated_contact_spring_forces_n=[KP[-1]*(q-CLOSE_Q_M) for q in contact_q],
                  contact_patch_at_onset=patches, closure_scan=closure,
                  hypothetical_carried_envelope=route,
                  route_samples_per_finger_configuration=len(lift_rows),
                  lift_endpoint_delta_world_m=(np.asarray(ctx['points']['A']['position_world_m'])-p0).tolist(),
                  minimum_hypothetical_cube_table_gap_m=min(r['hypothetical_cube_bottom_table_gap_m'] for r in route),
                  minimum_finger_table_gap_m=min(r['min_finger_table_gap_m'] for r in route),
                  minimum_cube_finger_tray_bound_m=min(r['min_cube_or_finger_tray_separation_lower_bound_m'] for r in route),
                  unexpected_route_contact_samples=sum(bool(r['unexpected_contacts']) for r in route),
                  integrated_state_unchanged=bool(np.array_equal(ctx['data'].qpos,ctx['base_qpos']) and ctx['data'].time==0),
                  limitations=['No physical close, lift, slip, force-control or grasp executed.',
                               'Static prescribed penetration does not measure physical gripping force.',
                               'Carried cube pose is a geometry-only rigid-transform hypothesis in separate MjData.',
                               'No weld, equality, altered friction or mass is written into the frozen scene.',
                               'Discrete static checks do not certify continuous dynamics.'])
    result['static_geometry_prepared'] = bool(all(not r['unexpected_contacts'] for r in closure)
        and result['unexpected_route_contact_samples']==0 and result['minimum_hypothetical_cube_table_gap_m']>=-1e-9
        and result['minimum_finger_table_gap_m']>.001 and result['minimum_cube_finger_tray_bound_m']>.001
        and all(m.jnt_range[j,0]<CLOSE_Q_M<OPEN_Q_M<=m.jnt_range[j,1] for j in jids))
    result['suggested_learning_gates'] = dict(
        close_reference_m=CLOSE_Q_M, lift_route='Existing source B_to_A 8..12 s, retimed only if desired.',
        expected_endpoint_cube_raise_m=float(np.asarray(ctx['points']['A']['position_world_m'])[2]-p0[2]),
        both_finger_normal_force_min_n=.30,
        force_gate_reason='Per-finger normal >=0.30 N implies 0.60 N nominal friction capacity at observed mu=1 versus 0.34335 N weight; actual lift/retention must still pass.',
        fully_off_table_gap_min_m=.020,
        hold_relative_position_drift_max_m=.005,
        hold_relative_orientation_drift_max_rad=.10,
        continuous_final_hold_s=2.,
        gate_limitation='Proposed pedagogical acceptance thresholds, not hardware specifications or guaranteed success.')
    save_json(out/'receipt.json', result)
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    print(json.dumps({k:result[k] for k in ('static_geometry_prepared','zero_velocity_contact_onset_q_m','estimated_contact_spring_forces_n','minimum_finger_table_gap_m','minimum_cube_finger_tray_bound_m','minimum_hypothetical_cube_table_gap_m','unexpected_route_contact_samples','lift_endpoint_delta_world_m')},indent=2))
    print('GEOMETRY RECEIPT:', out/'receipt.json')
    return result


def main():
    guard()
    import numpy as np
    import mujoco
    # Enforce zero integration even if a future dependency accidentally steps.
    def forbidden_step(*args, **kwargs):
        raise AssertionError('Static geometry audit must not call mj_step')
    mujoco.mj_step = forbidden_step
    out=ROOT/'outputs'/('grasp_contact_geometry_preparation_'+stamp())
    out.mkdir(parents=True,exist_ok=False)
    result=_audit(out,mujoco,np)
    return 0 if result['static_geometry_prepared'] else 2


if __name__=='__main__':
    raise SystemExit(main())
