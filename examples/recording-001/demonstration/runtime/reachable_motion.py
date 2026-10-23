"""User-operated, torque-driven empty-hand motion between FK-defined poses.

Only initialization sets the integrated configuration. Reference and gravity
calculations use separate MjData objects. --inspect never calls mj_step.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import time
import traceback

from lesson_common import ROOT, guard, save_json, sha, stamp
from reachable_pose_preview import read_frozen_source, model_context, apply_and_check
from pickplace_layout_preview import set_camera

INDEX = Path(__file__).with_name('reachable_motion_source.json')
KP = [120., 100., 80., 60., 40., 30., 30., 600., 600.]
KD = [8., 7., 5., 4., 2.5, 2., 2., 8., 8.]
CAP = [12., 12., 10., 8., 6., 3., 3., 4., 4.]
THRESHOLDS = dict(position_warning_m=.003, orientation_warning_rad=.05,
                  max_joint_error_rad=.15, max_joint_speed_rad_s=2.,
                  arm_limit_margin_rad=.02, min_finger_table_gap_m=.0005,
                  max_finger_error_m=.003, max_cube_xy_displacement_m=.003,
                  max_cube_z_displacement_m=.005, saturation_duration_s=.25)


def load_reference(np):
    index = json.loads(INDEX.read_text(encoding='utf-8-sig'))
    source = (ROOT / index['path']).resolve()
    if not source.is_relative_to((ROOT/'outputs').resolve()) or sha(source) != index['sha256']:
        raise ValueError('Frozen motion reference missing or changed.')
    plan = json.loads(source.read_text(encoding='utf-8-sig'))
    rows = plan['reference_samples']
    if plan.get('status') != 'discrete_static_reference_prepared' or not all(r.get('static_sample_passed') for r in rows):
        raise ValueError('The reference has not passed its discrete static checks.')
    q = np.asarray([r['q_ref'] for r in rows], dtype=float)
    qd = np.asarray([r['dq_ref'] for r in rows], dtype=float)
    t = np.asarray([r['sim_s'] for r in rows], dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or qd.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError('Malformed motion reference.')
    if len(rows) < 2 or len(rows) > 10001 or not np.allclose(t, np.arange(len(t))*.004, atol=1e-10, rtol=0):
        raise ValueError('Expected a bounded 0.004-second reference grid including terminal state.')
    return source, plan, rows, q, qd


def setup(mujoco, np):
    folder, cfg, audit, points = read_frozen_source()
    source, plan, rows, q, qd = load_reference(np)
    index = json.loads(INDEX.read_text(encoding='utf-8-sig'))
    if index['scene_sha256'] != sha(folder/'scene.xml') or index['config_sha256'] != sha(folder/'config.json'):
        raise ValueError('Motion reference and model are not the same frozen layout.')
    if plan['source_scene_sha256'] != cfg['scene_sha256'] or plan['source_config_sha256'] != sha(folder/'config.json'):
        raise ValueError('Reference-file source metadata differs from the index.')
    if np.any(np.asarray(plan['statistics']['max_abs_gravity_bias_nm']) > .7*np.asarray(CAP[:7])):
        raise ValueError('Full-path static gravity headroom is insufficient.')
    ctx = model_context(folder, cfg, mujoco, np)
    m, d = ctx['model'], ctx['data']
    if mujoco.__version__ != '3.3.7' or abs(m.opt.timestep-.004)>1e-12 or int(m.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError('Expected MuJoCo 3.3.7, 0.004 s and implicitfast.')
    if not np.array_equal(m.opt.gravity, [0., 0., -9.81]):
        raise ValueError('Unexpected gravity.')
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg['joint_order']]
    aids = []
    for jid in jids:
        matches = np.where((m.actuator_trnid[:, 0] == jid) & (m.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT)))[0]
        if len(matches) != 1:
            raise ValueError('Each controlled joint requires exactly one direct motor.')
        aids.append(int(matches[0]))
    for i, aid in enumerate(aids):
        if (int(m.actuator_gaintype[aid]) != 0 or int(m.actuator_biastype[aid]) != 0 or int(m.actuator_dyntype[aid]) != 0
                or not np.array_equal(m.actuator_gear[aid], [1,0,0,0,0,0]) or m.actuator_gainprm[aid,0] != 1
                or not m.actuator_ctrllimited[aid] or m.actuator_ctrlrange[aid,0] > -CAP[i] or m.actuator_ctrlrange[aid,1] < CAP[i]):
            raise ValueError('Unexpected motor semantics or insufficient source range.')
        if m.actuator_forcelimited[aid] and (m.actuator_forcerange[aid,0]>-CAP[i] or m.actuator_forcerange[aid,1]<CAP[i]):
            raise ValueError('Software force cap exceeds actuator force range.')
        jid = jids[i]
        if m.jnt_actfrclimited[jid] and (m.jnt_actfrcrange[jid,0]>-CAP[i] or m.jnt_actfrcrange[jid,1]<CAP[i]):
            raise ValueError('Software cap exceeds joint actuator force range.')
    qa, va = m.jnt_qposadr[jids].copy(), m.jnt_dofadr[jids].copy()
    limits = m.jnt_range[jids[:7]].copy()
    if (not np.allclose(q[0], points['A']['q_arm_rad'], rtol=0, atol=1e-12)
            or not np.allclose(q[-1], points['C']['q_arm_rad'], rtol=0, atol=1e-12)
            or np.any(q <= limits[:,0]+.02) or np.any(q >= limits[:,1]-.02)):
        raise ValueError('Path endpoints or joint bounds changed.')
    apply_and_check(ctx, points['A'], mujoco, np)  # The only actual-state initialization.
    meshes=[]
    for name in ('right_dex1_finger_1_collision', 'right_dex1_finger_2_collision'):
        gid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        mid=int(m.geom_dataid[gid]); start=int(m.mesh_vertadr[mid]); count=int(m.mesh_vertnum[mid])
        meshes.append((gid,m.mesh_vert[start:start+count].astype(float).copy()))
    ctx.update(folder=folder,cfg=cfg,points=points,reference_file=source,plan=plan,rows=rows,qref=q,qdref=qd,
               qa=qa,va=va,aids=np.array(aids),limits=limits,gravity=mujoco.MjData(m),reference=mujoco.MjData(m),
               kp=np.array(KP),kd=np.array(KD),cap=np.array(CAP),finger_meshes=meshes,
               cube_initial=d.qpos[ctx['cube_adr']:ctx['cube_adr']+3].copy())
    return ctx


def control_terms(ctx, qref, qdref, mujoco, np):
    m,d,g=ctx['model'],ctx['data'],ctx['gravity']
    g.qpos[:]=d.qpos
    g.qvel[:]=0
    g.qfrc_applied[:]=0
    g.xfrc_applied[:]=0
    g.ctrl[:]=0
    mujoco.mj_forward(m,g)
    gravity=g.qfrc_bias[ctx['va']].copy()  # qvel=0: excludes velocity bias.
    pd=ctx['kp']*(qref-d.qpos[ctx['qa']])+ctx['kd']*(qdref-d.qvel[ctx['va']])
    raw=pd+gravity
    return pd,gravity,raw,np.clip(raw,-ctx['cap'],ctx['cap'])


def sample(ctx, step, wall_s, mujoco, np):
    m,d,r=ctx['model'],ctx['data'],ctx['reference']
    spec=ctx['rows'][step]
    qref=np.r_[ctx['qref'][step], [.0245,.0245]]
    qdref=np.r_[ctx['qdref'][step], [0.,0.]]
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        raise RuntimeError('Nonfinite actual state.')
    mujoco.mj_forward(m,d)
    r.qpos[:]=ctx['base_qpos']
    r.qpos[ctx['qa']]=qref
    r.qvel[:]=0
    mujoco.mj_forward(m,r)
    actual=d.site_xpos[ctx['site']].copy(); ref=r.site_xpos[ctx['site']].copy()
    qa=np.empty(4);qr=np.empty(4)
    mujoco.mju_mat2Quat(qa,d.site_xmat[ctx['site']]);mujoco.mju_mat2Quat(qr,r.site_xmat[ctx['site']])
    angle=2*np.arccos(np.clip(abs(float(qa@qr)),0.,1.))
    finger_gap=min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in ctx['finger_meshes'])-ctx['cfg']['table']['top_z_m']
    unexpected=[]
    for contact in d.contact[:d.ncon]:
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (contact.geom1,contact.geom2)]
        if set(names)!={'task_cube_geom','task_table_top'} and contact.dist<=0:
            unexpected.append(dict(geoms=names,distance_m=float(contact.dist)))
    q=d.qpos[ctx['qa']].copy(); qd=d.qvel[ctx['va']].copy()
    pd,g,raw,tau=control_terms(ctx,qref,qdref,mujoco,np)
    cube=d.qpos[ctx['cube_adr']:ctx['cube_adr']+3].copy()
    delta=cube-ctx['cube_initial']
    row=dict(sim_s=float(d.time),wall_s=wall_s,phase=spec['phase'],terminal_state=0,
             **dict(zip(('ref_x_m','ref_y_m','ref_z_m'),ref)),
             **dict(zip(('actual_x_m','actual_y_m','actual_z_m'),actual)),
             position_error_m=float(np.linalg.norm(actual-ref)),orientation_error_rad=float(angle),
             min_finger_table_gap_m=float(finger_gap),unexpected_contact_count=len(unexpected),
             max_joint_error_rad=float(np.max(np.abs(qref[:7]-q[:7]))),
             max_torque_ratio=float(np.max(np.abs(raw[:7])/ctx['cap'][:7])),
             max_finger_force_ratio=float(np.max(np.abs(raw[7:])/ctx['cap'][7:])),
             cube_displacement_m=float(np.linalg.norm(delta)),cube_x_m=float(cube[0]),cube_y_m=float(cube[1]),cube_z_m=float(cube[2]))
    for i in range(9):
        for key,vector in (('q_ref',qref),('q',q),('qd_ref',qdref),('qd',qd),('pd',pd),('gravity',g),('tau_raw',raw),('tau',tau)):
            row[f'{key}_{i}']=float(vector[i])
    problems=[]
    if abs(d.time-step*m.opt.timestep)>1e-8: problems.append('unexpected_simulation_time')
    if np.any(np.abs(qref[:7]-q[:7])>THRESHOLDS['max_joint_error_rad']): problems.append('joint_tracking_stop')
    if np.any(np.abs(qd[:7])>THRESHOLDS['max_joint_speed_rad_s']): problems.append('joint_speed_stop')
    if np.any(q[:7]<=ctx['limits'][:,0]+.02) or np.any(q[:7]>=ctx['limits'][:,1]-.02): problems.append('joint_limit_stop')
    if np.max(np.abs(q[7:]-.0245))>THRESHOLDS['max_finger_error_m']: problems.append('open_finger_tracking_stop')
    if finger_gap<THRESHOLDS['min_finger_table_gap_m']: problems.append('finger_table_clearance_stop')
    if unexpected: problems.append('unexpected_contact')
    if np.linalg.norm(delta[:2])>THRESHOLDS['max_cube_xy_displacement_m'] or abs(delta[2])>THRESHOLDS['max_cube_z_displacement_m']: problems.append('cube_displacement_stop')
    if any(w.number for w in d.warning): problems.append('mujoco_warning')
    return row,tau,problems,unexpected


def terminal_row(row):
    row=dict(row,terminal_state=1)
    for key in list(row):
        if key.startswith(('tau_','pd_','gravity_')) or key in ('max_torque_ratio','max_finger_force_ratio'):
            row[key]=''
    return row


def inspect(ctx,mujoco,np):
    """Static regression only: no integration or viewer."""
    m,d=ctx['model'],ctx['data']
    row,tau,problems,_=sample(ctx,0,0.,mujoco,np)
    if problems: raise RuntimeError('Initial-state check: '+','.join(problems))
    # The zero-error initial controller must equal the independently recomputed gravity.
    if not np.allclose(tau,ctx['gravity'].qfrc_bias[ctx['va']],atol=1e-12,rtol=0):
        raise RuntimeError('Gravity compensation or initial motor cap mismatch.')
    maxg=np.zeros(9)
    ref=ctx['reference']
    for step in sorted(set(range(0,len(ctx['rows']),25))|{len(ctx['rows'])-1}):
        ref.qpos[:]=ctx['base_qpos'];ref.qpos[ctx['qa'][:7]]=ctx['qref'][step];ref.qvel[:]=0
        mujoco.mj_forward(m,ref)
        p=ref.site_xpos[ctx['site']]
        if np.linalg.norm(p-np.asarray(ctx['rows'][step]['position_world_m']))>1e-8:
            raise RuntimeError('Runtime reference FK differs from preparation.')
        maxg=np.maximum(maxg,np.abs(ref.qfrc_bias[ctx['va']]))
    if np.any(maxg>.7*ctx['cap']): raise RuntimeError('Insufficient static gravity headroom.')
    return dict(status='static_control_inspection_passed',physics_steps=0,model_loaded=True,
                planned_steps=len(ctx['rows'])-1,planned_simulation_s=ctx['rows'][-1]['sim_s'],
                initial_finger_table_gap_m=row['min_finger_table_gap_m'],max_abs_gravity_sampled=maxg.tolist(),
                max_gravity_to_cap_sampled=float(np.max(maxg/ctx['cap'])),
                controlled_joints=ctx['cfg']['joint_order'],actuator_ids=ctx['aids'].tolist(),
                gravity_sampling='Every 25th reference and terminal; full static route audit is separate.',
                limitation='No dynamic tracking, GUI or grasp was executed.')


def draw(viewer,ctx,mujoco,np):
    scn=viewer.user_scn;scn.ngeom=0
    for pos,color in ((ctx['data'].site_xpos[ctx['site']],[.1,.9,.8,1]),
                      (ctx['reference'].site_xpos[ctx['site']],[1,.55,.1,.8])):
        g=scn.geoms[scn.ngeom];scn.ngeom+=1
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.004,0,0]),pos,np.eye(3).ravel(),np.array(color,dtype=np.float32))


def run(ctx,report,out,mujoco,np):
    import mujoco.viewer
    m,d=ctx['model'],ctx['data'];dt=float(m.opt.timestep)
    commands=queue.SimpleQueue(); started=False; loop_start=None; step=0; last_phase=None
    first,tau,_,_=sample(ctx,0,0.,mujoco,np)
    start=time.perf_counter(); last_row=None; terminal_saved=False
    saturation=np.zeros(9); stats=dict(max_position_error_m=0.,max_orientation_error_rad=0.,
        max_joint_error_rad=0.,minimum_finger_table_gap_m=float('inf'),saturated_steps=0)
    def write_row(writer,row):
        nonlocal last_row
        writer.writerow(row);last_row=row;report['data_saved_samples']+=1
        for stat,col in (('max_position_error_m','position_error_m'),('max_orientation_error_rad','orientation_error_rad'),('max_joint_error_rad','max_joint_error_rad')):
            stats[stat]=max(stats[stat],row[col])
        stats['minimum_finger_table_gap_m']=min(stats['minimum_finger_table_gap_m'],row['min_finger_table_gap_m'])
    reason='window_closed';status='interrupted'
    with (out/'motion.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(first));writer.writeheader()
        try:
            with mujoco.viewer.launch_passive(m,d,key_callback=commands.put,show_left_ui=False,show_right_ui=False) as viewer:
                report['viewer_started']=True
                with viewer.lock():
                    set_camera(viewer.cam,ctx['cfg'],'oblique');viewer.opt.label=mujoco.mjtLabel.mjLABEL_NONE
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]=False
                print('Click scene, S starts 24 simulation seconds. Q ends. Camera 1/2/3.',flush=True)
                while viewer.is_running():
                    tick=time.perf_counter()
                    if tick-start>600 or (loop_start is not None and tick-loop_start>240):
                        reason='wall_time_limit';break
                    quit_requested=False
                    while not commands.empty():
                        key=commands.get()
                        if key not in (ord('S'),ord('Q'),ord('1'),ord('2'),ord('3')):continue
                        report['events'].append(dict(key=chr(key),wall_s=tick-start,sim_s=float(d.time)))
                        if key==ord('Q'):reason='user_Q';quit_requested=True;break
                        if key==ord('S') and not started:
                            started=True;loop_start=tick;report['physics_started']=True
                        if key in (ord('1'),ord('2'),ord('3')):
                            with viewer.lock():set_camera(viewer.cam,ctx['cfg'],{ord('1'):'oblique',ord('2'):'top',ord('3'):'side'}[key])
                    if quit_requested:break
                    stop=False
                    if started:
                        # Four physics/control updates per viewer refresh. Physics uses dt,
                        # never wall time, and the GUI remains responsive to S/Q.
                        for _ in range(4):
                            row,tau,problems,contacts=sample(ctx,step,time.perf_counter()-start,mujoco,np)
                            saturated=np.abs(np.array([row[f'tau_raw_{i}'] for i in range(9)]))>ctx['cap']
                            saturation=np.where(saturated,saturation+dt,0.)
                            if np.any(saturation>THRESHOLDS['saturation_duration_s']):problems.append('sustained_saturation')
                            if problems:
                                write_row(writer,terminal_row(row));terminal_saved=True
                                report['stop_diagnostics']=dict(problems=problems,contacts=contacts,row=row)
                                reason=','.join(problems);status='stopped';stop=True;break
                            if row['phase']!=last_phase:
                                print(f"PHASE: {row['phase']} | sim {d.time:.3f} s",flush=True);last_phase=row['phase']
                            if np.any(saturated):stats['saturated_steps']+=1
                            d.ctrl[ctx['aids']]=tau
                            # Confirm joint forces after writing this control, before integration.
                            mujoco.mj_forward(m,d)
                            if not np.allclose(d.qfrc_actuator[ctx['va']],tau,rtol=0,atol=1e-9):
                                raise RuntimeError('Actual actuator force differs from recorded unit-gear command.')
                            write_row(writer,row)
                            mujoco.mj_step(m,d);step+=1;report['steps']=step
                            if step==len(ctx['rows'])-1:
                                row,_,problems,contacts=sample(ctx,step,time.perf_counter()-start,mujoco,np)
                                write_row(writer,terminal_row(row));terminal_saved=True
                                status='completed' if not problems else 'stopped';reason='planned_steps_completed' if not problems else ','.join(problems)
                                if problems:report['stop_diagnostics']=dict(problems=problems,contacts=contacts,row=row)
                                stop=True;break
                        stream.flush()
                    with viewer.lock():draw(viewer,ctx,mujoco,np)
                    label=last_row['phase'] if last_row else 'waiting_at_A'
                    err=1000*last_row['position_error_m'] if last_row else 0.
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        'EMPTY-HAND MOTION | PD + gravity compensation\nS:start  Q:finish | 1:angle 2:top 3:side\n'
                        f'{label} | sim {d.time:.2f} / {ctx["rows"][-1]["sim_s"]:.1f} s | track {err:.2f} mm\n'
                        'Orange: reference | cyan: actual | cube remains free\nNo finger closing or object transport in this run.',''))
                    before=(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time))
                    viewer.sync()
                    if (not np.array_equal(d.qpos,before[0]) or not np.array_equal(d.qvel,before[1]) or not np.array_equal(d.ctrl,before[2])
                            or d.time!=before[3] or np.any(d.xfrc_applied) or np.any(d.qfrc_applied)
                            or abs(m.opt.timestep-dt)>1e-12 or not np.array_equal(m.opt.gravity,[0,0,-9.81])
                            or int(m.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)):
                        raise RuntimeError('Viewer changed state or physics. Use S/Q and camera controls only.')
                    if stop:
                        time.sleep(1.2);break
                    time.sleep(max(0.,(4*dt if started else .02)-(time.perf_counter()-tick)))
        finally:
            if not terminal_saved:
                try:
                    row,_,_,_=sample(ctx,step,time.perf_counter()-start,mujoco,np)
                    write_row(writer,terminal_row(row))
                except Exception:
                    report['terminal_record_error']=traceback.format_exc()
            report.update(simulation_s=float(d.time),steps=step,metrics=stats,
                          physics_loop_wall_s=None if loop_start is None else time.perf_counter()-loop_start,
                          final_qpos=d.qpos.tolist(),final_qvel=d.qvel.tolist())
    report.update(status=status,end_reason=reason)
    report['tracking_thresholds_passed']=(status=='completed' and stats['max_position_error_m']<=.003 and stats['max_orientation_error_rad']<=.05)
    return 0 if status in ('completed','interrupted') else 2


def main():
    guard()
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve() if args.output else ROOT/'outputs'/('reachable_motion_user_'+stamp())
    if out.parent!=(ROOT/'outputs').resolve() or not out.name.startswith('reachable_motion_user_'):
        raise ValueError('Use a fresh project outputs/reachable_motion_user_ directory.')
    if not args.inspect:out.mkdir(exist_ok=False)
    os.chdir(ROOT/'outputs' if args.inspect else out)
    report=dict(kind='reachable_empty_hand_dynamics',status='started',model_loaded=False,viewer_started=False,
                physics_started=False,steps=0,simulation_s=0.,data_saved_samples=0,events=[],
                learner_verified=False,grasp_validated=False,controller='PD_plus_zero_velocity_gravity',
                executable=sys.executable,temp=tempfile.gettempdir(),thresholds=THRESHOLDS,
                samples_semantics='Same-state pre-step references/positions and next-step commands; one final state has blank control fields. Simulation time, world coordinates. Motor columns 0..6: Nm; fingers 7..8: N.',
                reference_semantics='Quintic shoulder/elbow references, analytic wrist orientation coordination; qdot_ref from saved central differences with zero at holds/boundaries. FK path is not Cartesian straight.',
                kp=KP,kd=KD,software_caps=CAP,snapshots=[])
    start=time.perf_counter();code=0
    try:
        import numpy as np
        import mujoco
        ctx=setup(mujoco,np)
        inspection=inspect(ctx,mujoco,np)
        report.update(model_loaded=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,
                      expected_steps=len(ctx['rows'])-1,planned_simulation_s=ctx['rows'][-1]['sim_s'],dt_s=.004,
                      source_reference=str(ctx['reference_file']),static_inspection=inspection)
        if args.inspect:
            print(json.dumps(inspection,ensure_ascii=False,indent=2));return 0
        files=[(ctx['folder']/n,'source/'+n) for n in ('scene.xml','config.json','static_audit.json')]
        files.append((ctx['reference_file'],'source/path.json'))
        files += [(p,'code/'+p.name) for p in Path(__file__).parent.glob('*.py')]
        files += [(Path(__file__).with_name(n),'code/'+n) for n in ('reachable_preview_source.json','reachable_motion_source.json','Run-ReachableMotion.ps1','Show-ReachableMotion.ps1')]
        files.append((ROOT/'repos/g1_grasp_assets/asset_manifest.json','source/asset_manifest.json'))
        for source,relative in files:
            dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            report['snapshots'].append(dict(path=relative,sha256=sha(dest)))
        print('MOTION RUN:',out,flush=True)
        code=run(ctx,report,out,mujoco,np)
    except KeyboardInterrupt:
        report.update(status='interrupted',end_reason='KeyboardInterrupt');code=130
    except Exception:
        report.update(status='error',end_reason='exception',error=traceback.format_exc());code=1
        print(report['error'],file=sys.stderr)
        if not args.inspect:(out/'error.txt').write_text(report['error'],encoding='utf-8')
    finally:
        report['wall_s']=time.perf_counter()-start
        if not args.inspect:
            if (out/'motion.csv').exists():report['motion_csv_sha256']=sha(out/'motion.csv')
            save_json(out/'summary.json',report)
            print('SUMMARY:',out/'summary.json',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
