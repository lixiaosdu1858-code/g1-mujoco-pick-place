"""Frozen carry/release geometry for the existing successful noslip=3 grasp.

Preparation only uses independent MjData objects and mj_forward. Virtual cube
attachment is a geometric envelope assumption, never a dynamics operation.
Runtime loads the frozen arm references and must preserve the real free cube.
"""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

from lesson_common import ROOT, guard, save_json, sha, stamp

SUCCESS_RUN = ROOT/'outputs/solver_grasp_user_20261020_155236_233_repeat_n3'
SUCCESS_SHA = '73ace9b44044367f747f3dd70639e6ae20bd7e9265783fc8db9c98fe6e9e3e5b'
FROZEN_RELATIVE = 'outputs/carry_route_preparation_20261020_162150_917767/path.json'
FROZEN_SHA = 'fd36e1647d135f097b0597e9c68fcac93768cceffee5e65a16f7e5595119ae45'
DT = .004
MAX_FK = 25000
MAX_WALL_S = 90.


def _source(ctx):
    if sha(SUCCESS_RUN/'summary.json') != SUCCESS_SHA:
        raise ValueError('Frozen successful grasp summary changed')
    summary = json.loads((SUCCESS_RUN/'summary.json').read_text(encoding='utf-8-sig'))
    if (summary['status'] != 'completed' or not summary['grasp_hold_passed']
            or summary['noslip_iterations'] != 3):
        raise ValueError('Carry preparation requires the independently repeated successful grasp')
    if sha(SUCCESS_RUN/'grasp.csv') != summary['grasp_csv_sha256']:
        raise ValueError('Successful grasp CSV changed')
    for item in summary['snapshots']:
        if sha(SUCCESS_RUN/item['path']) != item['sha256']:
            raise ValueError('Successful grasp snapshot changed: '+item['path'])
    if (sha(ctx['folder']/'scene.xml') != summary['frozen_inputs']['scene']
            or sha(ctx['reference_file']) != summary['frozen_inputs']['reference']):
        raise ValueError('Carry model or original reference differs from successful grasp')
    if int(ctx['model'].opt.noslip_iterations) != 3:
        raise ValueError('Expected frozen successful noslip=3 setting')
    return summary


def _tray(ctx, mujoco, np, data):
    m = ctx['model']
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'task_tray')
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'tray_floor')
    walls = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'tray_wall_'+str(i)) for i in range(4)]
    position, rotation = data.xpos[body].copy(), data.xmat[body].reshape(3,3).copy()
    if not np.allclose(rotation[:,2], [0,0,1], atol=1e-12, rtol=0):
        raise ValueError('This frozen task requires a horizontal tray')
    return dict(body_id=body, floor_geom_id=floor, wall_geom_ids=walls,
                allowed_cube_tray_geom_ids=[floor]+walls,
                position_world_m=position.tolist(), rotation_world=rotation.tolist(),
                inner_half_size_xy_m=ctx['cfg']['tray']['interior_half_size_xy_m'],
                floor_top_z_m=ctx['cfg']['tray']['floor_top_z_m'],
                wall_top_z_m=ctx['cfg']['tray']['wall_top_z_m'])


def _shape_vertices(m, data, gid, np, mujoco, cache):
    if gid not in cache:
        kind = int(m.geom_type[gid])
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            cache[gid] = np.asarray(list(itertools.product((-1.,1.), repeat=3)))*m.geom_size[gid]
        elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(m.geom_dataid[gid])
            first, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            cache[gid] = m.mesh_vert[first:first+count].astype(float).copy()
        else:
            raise ValueError('Unexpected geometry type in explicit carry clearance check')
    return cache[gid] @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid]


def _separation(vertices, rotation, box_position, box_rotation, box_size, np):
    """Positive projection separation is a conservative distance lower bound.

    Mesh convex collision hulls lie inside all compiled-vertex projections.
    A zero bound proves nothing. Arbitrary-axis bounds are maximized, not RSS.
    """
    directions = [box_rotation[:,i] for i in range(3)] + [rotation[:,i] for i in range(3)]
    directions += [np.cross(rotation[:,i], box_rotation[:,j]) for i in range(3) for j in range(3)]
    best = 0.
    for axis in directions:
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            continue
        axis = axis/norm
        projected = vertices @ axis
        center = float(box_position @ axis)
        extent = float(np.abs(box_rotation.T @ axis) @ box_size)
        best = max(best, center-extent-float(projected.max()), float(projected.min())-(center+extent))
    return best


def _cases(ctx, summary, mujoco, np):
    m = ctx['model']
    data = mujoco.MjData(m)
    data.qpos[:] = summary['final_qpos']
    data.qvel[:] = 0.
    mujoco.mj_forward(m, data)
    R = data.site_xmat[ctx['site']].reshape(3,3)
    p = data.site_xpos[ctx['site']]
    cg = ctx['cube_geom']
    final_p = R.T @ (data.geom_xpos[cg]-p)
    final_R = R.T @ data.geom_xmat[cg].reshape(3,3)
    fingers = data.qpos[ctx['qa'][7:]].copy()
    baseline = summary['grip_baseline']
    return [dict(id='successful_grip_baseline', position_in_tool_m=baseline['position_in_tool_m'],
                 rotation_in_tool=baseline['rotation_in_tool'], finger_q_m=fingers.tolist()),
            dict(id='successful_final_hold', position_in_tool_m=final_p.tolist(),
                 rotation_in_tool=final_R.tolist(), finger_q_m=fingers.tolist())]


def _segments(ctx, np):
    result = {}
    for name, indices, direction in [('transport',range(3000,4251),1.),
                                     ('descend',range(4250,5501),1.),
                                     ('retract',range(5500,4249,-1),-1.)]:
        rows=[]
        for step, old in enumerate(indices):
            source=ctx['rows'][old]
            rows.append(dict(step=step,sim_s=step*DT,phase=name,source_reference_step=old,
                q_ref=ctx['qref'][old].tolist(),dq_ref=(direction*ctx['qdref'][old]).tolist(),
                position_world_m=source['position_world_m'],orientation_wxyz=source['orientation_wxyz']))
        result[name] = dict(duration_s=(len(rows)-1)*DT,rows=rows)
    if not np.allclose(result['transport']['rows'][0]['q_ref'],ctx['qref'][3000],rtol=0,atol=1e-12):
        raise ValueError('Carry start differs from final successful lift reference')
    return result


def prepare(ctx, mujoco, np):
    summary = _source(ctx)
    actual = ctx['data']
    before = (actual.qpos.copy(),actual.qvel.copy(),actual.time)
    m, data = ctx['model'], mujoco.MjData(ctx['model'])
    data.qpos[:] = ctx['base_qpos']
    mujoco.mj_forward(m,data)
    tray = _tray(ctx,mujoco,np,data)
    cases = _cases(ctx,summary,mujoco,np)
    segments = _segments(ctx,np)
    cache = {}
    cube = ctx['cube_geom']
    finger_ids = ctx['finger_geoms']
    obstacles = [ctx['table_geom'],tray['floor_geom_id']]+tray['wall_geom_ids']
    obstacle_boxes = {g:(data.geom_xpos[g].copy(),data.geom_xmat[g].reshape(3,3).copy(),m.geom_size[g].copy()) for g in obstacles}
    start, count = time.perf_counter(),0
    samples=[]; failures=[]

    def fk():
        nonlocal count
        if count>=MAX_FK or time.perf_counter()-start>MAX_WALL_S:
            raise RuntimeError('Bounded static carry preparation exceeded budget')
        mujoco.mj_forward(m,data);count+=1

    def evaluate(q, finger, case, label, step, fixed_cube=None, allow_floor=False, allow_finger_contact=True):
        mujoco.mj_resetData(m,data)
        data.qpos[:]=ctx['base_qpos']
        data.qpos[ctx['qa'][:7]]=q
        data.qpos[ctx['qa'][7:]]=finger
        fk()
        if fixed_cube is None:
            R=data.site_xmat[ctx['site']].reshape(3,3)
            cube_p=data.site_xpos[ctx['site']]+R@np.asarray(case['position_in_tool_m'])
            cube_R=R@np.asarray(case['rotation_in_tool'])
            cube_quat=np.empty(4);mujoco.mju_mat2Quat(cube_quat,cube_R.ravel())
            cube_pose=np.r_[cube_p,cube_quat]
        else:
            cube_pose=np.asarray(fixed_cube)
        data.qpos[ctx['cube_adr']:ctx['cube_adr']+7]=cube_pose
        fk()
        unintended=[]; permitted=[]
        for contact in data.contact[:data.ncon]:
            pair={int(contact.geom1),int(contact.geom2)}
            if contact.dist>0:continue
            expected_grip=allow_finger_contact and cube in pair and bool(pair.intersection(finger_ids))
            expected_floor=allow_floor and pair=={cube,tray['floor_geom_id']}
            row=dict(geoms=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in pair],distance_m=float(contact.dist))
            (permitted if expected_grip or expected_floor else unintended).append(row)
        lower_bounds=[]
        for gid in [cube]+finger_ids:
            vertices=_shape_vertices(m,data,gid,np,mujoco,cache)
            R=data.geom_xmat[gid].reshape(3,3)
            for obstacle in obstacles:
                if gid==cube and obstacle==tray['floor_geom_id'] and allow_floor:continue
                bound=_separation(vertices,R,*obstacle_boxes[obstacle],np)
                # Positive MuJoCo distances are also admissible evidence;
                # zero results never replace independent separation checks.
                distance=float(mujoco.mj_geomDistance(m,data,gid,obstacle,1.,None))
                if distance>0:bound=max(bound,distance)
                lower_bounds.append(dict(moving_geom=gid,obstacle_geom=obstacle,clearance_lower_bound_m=bound))
        cube_vertices=_shape_vertices(m,data,cube,np,mujoco,cache)
        local=(cube_vertices-np.asarray(tray['position_world_m']))@np.asarray(tray['rotation_world'])
        inner=np.asarray(tray['inner_half_size_xy_m'])
        xy_margin=float(np.min(inner-np.max(np.abs(local[:,:2]),axis=0)))
        floor_gap=float(cube_vertices[:,2].min()-tray['floor_top_z_m'])
        tightest=min(lower_bounds,key=lambda r:r['clearance_lower_bound_m'])
        margin=float(np.min(np.minimum(np.asarray(q)-ctx['limits'][:,0],ctx['limits'][:,1]-np.asarray(q))))
        row=dict(phase=label,step=step,case=case['id'],q_ref=np.asarray(q).tolist(),
            finger_q_m=np.asarray(finger).tolist(),cube_pose_world=cube_pose.tolist(),
            cube_bottom_world_m=float(cube_vertices[:,2].min()),cube_floor_clearance_m=floor_gap,
            cube_inner_xy_margin_m=xy_margin,tightest=tightest,
            min_joint_limit_margin_rad=margin,unintended_contacts=unintended,permitted_contacts=permitted)
        row['static_passed']=bool(not unintended and margin>.02 and tightest['clearance_lower_bound_m']>.0005
            and not any(w.number for w in data.warning) and data.time==0.)
        samples.append(row)
        if not row['static_passed']:failures.append(row)
        return row

    release_states=[]
    for case in cases:
        for name in ('transport','descend'):
            for row in segments[name]['rows']:
                evaluate(row['q_ref'],case['finger_q_m'],case,name,row['step'])
                if failures:break
            if failures:break
        if failures:break
        qC=segments['descend']['rows'][-1]['q_ref']
        release=evaluate(qC,case['finger_q_m'],case,'release_geometry',0)
        if release['cube_floor_clearance_m']<.003 or release['cube_inner_xy_margin_m']<.01:
            failures.append(dict(reason='release_drop_or_inner_margin_gate',**release))
            break
        resting=np.asarray(release['cube_pose_world']).copy()
        resting[2]-=release['cube_floor_clearance_m']
        release_states.append(dict(case=case['id'],release_cube_pose_world=release['cube_pose_world'],
            resting_cube_pose_world=resting.tolist(),drop_gap_m=release['cube_floor_clearance_m'],
            inner_margin_m=release['cube_inner_xy_margin_m']))
        # Cartesian product covers all sampled finger opening and cube-drop
        # offsets; this is a geometric envelope, not a falling-object rollout.
        for opening in np.linspace(0.,1.,21):
            finger=(1-opening)*np.asarray(case['finger_q_m'])+opening*.0245
            for falling in np.linspace(0.,1.,11):
                pose=(1-falling)*np.asarray(release['cube_pose_world'])+falling*resting
                evaluate(qC,finger,case,'opening_drop_envelope',len(samples),fixed_cube=pose,
                         allow_floor=bool(falling==1.),allow_finger_contact=True)
                if failures:break
            if failures:break
        if failures:break
        for row in segments['retract']['rows']:
            evaluate(row['q_ref'],[.0245,.0245],case,'retract',row['step'],fixed_cube=resting,
                     allow_floor=True,allow_finger_contact=False)
            if failures:break
        if failures:break
    unchanged=np.array_equal(before[0],actual.qpos) and np.array_equal(before[1],actual.qvel) and before[2]==actual.time
    if not unchanged:raise AssertionError('Static preparation changed integrated context data')
    return dict(kind='grasp_carry_release_static_reference',status='static_route_prepared' if not failures else 'blocked_static_route',
        dt_s=DT,segments=segments,tray=tray,grip_geometry_cases=cases,release_geometry=release_states,
        source=dict(success_directory=str(SUCCESS_RUN),success_summary_sha256=SUCCESS_SHA,
                    scene_sha256=sha(ctx['folder']/'scene.xml'),config_sha256=sha(ctx['folder']/'config.json'),
                    original_reference_sha256=sha(ctx['reference_file']),noslip_iterations=3),
        audit=dict(physics_steps=0,simulation_s=0.,fk_evaluations=count,wall_s=time.perf_counter()-start,
            actual_context_unchanged=bool(unchanged),sample_count=len(samples),failures=failures,
            minimum_clearance_lower_bound_m=min(r['tightest']['clearance_lower_bound_m'] for r in samples),
            minimum_joint_margin_rad=min(r['min_joint_limit_margin_rad'] for r in samples),
            checks_by_phase={phase:dict(sample_count=sum(r['phase']==phase for r in samples),
                min_clearance_lower_bound_m=min(r['tightest']['clearance_lower_bound_m'] for r in samples if r['phase']==phase))
                for phase in dict.fromkeys(r['phase'] for r in samples)},
            dynamic_carry_verified=False,physical_release_verified=False,
            limitations=['Two measured rigid hand-object relations are static envelopes, not predictions of slip or contact stability.',
                'Discrete checks do not certify continuous collision absence.',
                'Opening/drop envelope includes intended cube/finger and cube/floor contacts; no contact force or success is inferred.',
                'Original pre-lift reference and integrated state are not modified.']),static_samples=samples)


def load_route(ctx,mujoco,np):
    if not FROZEN_RELATIVE or not FROZEN_SHA:
        raise RuntimeError('No accepted carry reference has been frozen yet')
    path=(ROOT/FROZEN_RELATIVE).resolve()
    if not path.is_relative_to((ROOT/'outputs').resolve()) or sha(path)!=FROZEN_SHA:
        raise ValueError('Frozen carry route missing or changed')
    route=json.loads(path.read_text(encoding='utf-8-sig'))
    source=route['source']
    if (route['status']!='static_route_prepared' or source['scene_sha256']!=sha(ctx['folder']/'scene.xml')
        or source['config_sha256']!=sha(ctx['folder']/'config.json')
        or source['original_reference_sha256']!=sha(ctx['reference_file'])
        or int(ctx['model'].opt.noslip_iterations)!=3 or route['dt_s']!=float(ctx['model'].opt.timestep)):
        raise ValueError('Carry reference and current model/configuration disagree')
    for name,seg in route['segments'].items():
        rows=seg['rows']
        if len(rows)!=round(seg['duration_s']/DT)+1:raise ValueError('Truncated carry segment')
        for i,row in enumerate(rows):
            q=np.asarray(row['q_ref']);dq=np.asarray(row['dq_ref'])
            if q.shape!=(7,) or dq.shape!=(7,) or not np.isfinite(np.r_[q,dq]).all():
                raise ValueError('Malformed carry joint reference')
            if row['step']!=i or abs(row['sim_s']-i*DT)>1e-10:raise ValueError('Carry sample grid mismatch')
            if np.any(q<=ctx['limits'][:,0]+.02) or np.any(q>=ctx['limits'][:,1]-.02):
                raise ValueError('Carry reference outside joint margin')
    if not np.allclose(route['segments']['transport']['rows'][0]['q_ref'],ctx['qref'][3000],rtol=0,atol=1e-12):
        raise ValueError('Carry start and successful lift endpoint mismatch')
    for first,second in [('transport','descend'),('descend','retract')]:
        if not np.array_equal(route['segments'][first]['rows'][-1]['q_ref'],route['segments'][second]['rows'][0]['q_ref']):
            raise ValueError('Carry segment endpoints are not continuous')
    for seg in route['segments'].values():
        if (np.max(np.abs(seg['rows'][0]['dq_ref']))>1e-12
                or np.max(np.abs(seg['rows'][-1]['dq_ref']))>1e-12):
            raise ValueError('Each user-triggered segment must begin/end at zero reference velocity')
    route.update(path=path,sha256=FROZEN_SHA)
    return route


def main():
    guard()
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if not args.prepare:raise ValueError('Only explicit static preparation is exposed by this script')
    out=ROOT/'outputs'/('carry_route_preparation_'+stamp());out.mkdir(exist_ok=False)
    import numpy as np
    import mujoco
    import grasp_solver_compare
    try:
        ctx=grasp_solver_compare.setup(3,mujoco,np)
        report=prepare(ctx,mujoco,np)
        samples=report.pop('static_samples')
        save_json(out/'static_samples.json',samples)
        report['audit']['static_samples_sha256']=sha(out/'static_samples.json')
        report['audit']['script_sha256']=sha(__file__)
        save_json(out/'path.json',report)
        shutil.copy2(__file__,out/'carry_route.py')
        save_json(out/'receipt.json',dict(status=report['status'],path_sha256=sha(out/'path.json'),
             source=report['source'],audit=report['audit'],executable=sys.executable,temp=tempfile.gettempdir()))
        print(json.dumps(dict(output=str(out),path_sha256=sha(out/'path.json'),status=report['status'],
             audit=report['audit'],release_geometry=report['release_geometry']),indent=2))
        return 0 if report['status']=='static_route_prepared' else 2
    except BaseException as exc:
        save_json(out/'preparation_error.json',dict(status='error',error=str(exc),physics_steps=0))
        raise


if __name__=='__main__':raise SystemExit(main())
