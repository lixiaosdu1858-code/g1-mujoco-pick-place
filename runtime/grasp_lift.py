"""User-triggered contact grasp and lift. No attachment or state playback.

S approaches B; C closes; L lifts after bilateral contact; Q stops.
Only setup initializes the integrated state. All subsequent motion uses motors.
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
import reachable_motion as motion
from pickplace_layout_preview import set_camera

DT = .004
OPEN = .0245
CLOSED = -.0065
THRESHOLDS = dict(minimum_lift_m=.02, maximum_relative_slip_m=.005,
    maximum_relative_rotation_rad=.1, minimum_finger_normal_n=.3, hold_s=2.,
    contact_stable_s=.25, max_hold_cube_speed_m_s=.02, max_hold_cube_angular_speed_rad_s=.2,
    position_warning_m=.003, orientation_warning_rad=.05,
    position_stop_m=.015, orientation_stop_rad=.12, max_joint_error_rad=.15,
    max_joint_speed_rad_s=2., max_finger_speed_m_s=.25, min_finger_table_gap_m=.0005,
    maximum_simulation_s=120., maximum_physics_wall_s=240., maximum_wait_phase_s=45.)


class Stages:
    """Pure command/phase timing; premature commands are rejected, never queued."""
    def __init__(self):
        self.phase='waiting_start'; self.start=0

    def enter(self, phase, step):
        self.phase=phase; self.start=step

    def age(self, step):
        return (step-self.start)*DT

    def advance(self, step):
        for old,new,seconds in [('approach','B_ready',6.),('closing','grip_settle',2.),
                                ('lifting','lift_hold',4.)]:
            if self.phase==old and self.age(step)>=seconds-1e-10:
                self.enter(new,step)

    def command(self, key, step, current_bilateral=False):
        expected={'S':'waiting_start','C':'B_ready','L':'grip_ready'}
        if key not in expected or self.phase!=expected[key]:
            return False,'wrong_phase'
        if key=='L' and not current_bilateral:
            return False,'bilateral_contact_not_ready'
        self.enter({'S':'approach','C':'closing','L':'lifting'}[key],step)
        return True,'accepted'


class Continuous:
    def __init__(self):
        self.since=None; self.maximum_s=0.

    def observe(self, valid, step):
        if not valid:
            self.since=None
            return 0.
        if self.since is None:self.since=step
        duration=(step-self.since)*DT
        self.maximum_s=max(self.maximum_s,duration)
        return duration


def references(ctx, stage, step, np):
    n=step-stage.start
    if stage.phase=='approach':idx=min(n,1500)
    elif stage.phase=='lifting':idx=2000+min(n,1000)
    elif stage.phase in ('lift_hold',):idx=3000
    elif stage.phase=='waiting_start':idx=0
    else:idx=1500
    moving=stage.phase in ('approach','lifting')
    finger,velocity=OPEN,0.
    if stage.phase=='closing':
        u=min(1.,max(0.,stage.age(step)/2.))
        finger=OPEN+(CLOSED-OPEN)*u**3*(10+u*(-15+6*u))
        velocity=(CLOSED-OPEN)*30*u*u*(1-u)**2/2.
    elif stage.phase in ('grip_settle','grip_ready','lifting','lift_hold'):
        finger=CLOSED
    return (np.r_[ctx['qref'][idx],[finger,finger]],
            np.r_[ctx['qdref'][idx] if moving else np.zeros(7),[velocity,velocity]])


def setup(mujoco,np):
    ctx=motion.setup(mujoco,np)
    index=json.loads(Path(__file__).with_name('grasp_lift_source.json').read_text(encoding='utf-8-sig'))
    prep_path=(ROOT/index['geometry_receipt']).resolve()
    if not prep_path.is_relative_to((ROOT/'outputs').resolve()) or sha(prep_path)!=index['geometry_receipt_sha256']:
        raise ValueError('Static contact geometry receipt missing or changed')
    prep=json.loads(prep_path.read_text(encoding='utf-8-sig'))
    if (not prep['static_geometry_prepared'] or prep['source_scene_sha256']!=sha(ctx['folder']/'scene.xml')
        or prep['source_config_sha256']!=sha(ctx['folder']/'config.json') or prep['source_reference_sha256']!=sha(ctx['reference_file'])
        or prep['close_reference_q_m']!=CLOSED or prep['open_q_m']!=OPEN):
        raise ValueError('Closure or route no longer matches static preparation')
    ctx['contact_geometry_receipt']=prep_path
    m=ctx['model']
    ctx['cube_geom']=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'task_cube_geom')
    ctx['table_geom']=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'task_table_top')
    ctx['finger_geoms']=[g for g,_ in ctx['finger_meshes']]
    finger_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ctx['cfg']['finger_joint_order']]
    ctx['finger_limits']=m.jnt_range[finger_ids].copy()
    if not (np.all(CLOSED>ctx['finger_limits'][:,0]) and np.all(OPEN<=ctx['finger_limits'][:,1]+1e-10)):
        raise ValueError('Finger reference outside frozen physical bounds.')
    if m.neq or m.nmocap:
        raise ValueError('This lesson must have no equality attachment or mocap body.')
    if not np.allclose(ctx['qref'][1500],ctx['qref'][2000],rtol=0,atol=1e-12) or not np.allclose(ctx['qref'][3000],ctx['qref'][0],rtol=0,atol=1e-12):
        raise ValueError('Frozen approach/lift endpoints disagree.')
    ctx['grip_baseline']=None
    return ctx


def cube_relative(ctx,np):
    d=ctx['data'];gid=ctx['cube_geom']
    rotation=d.site_xmat[ctx['site']].reshape(3,3)
    return (rotation.T@(d.geom_xpos[gid]-d.site_xpos[ctx['site']]),
            rotation.T@d.geom_xmat[gid].reshape(3,3))


def bilateral(row):
    return all(row[f'finger{i}_contact_count']>0 and row[f'finger{i}_normal_n']>=THRESHOLDS['minimum_finger_normal_n'] for i in (1,2))


def hold_valid(row):
    return (bilateral(row) and row['grip_reference_valid']==1
        and row['cube_bottom_clearance_m']>=THRESHOLDS['minimum_lift_m']
        and row['cube_table_contact_count']==0
        and row['relative_slip_m']<=THRESHOLDS['maximum_relative_slip_m']
        and row['relative_rotation_rad']<=THRESHOLDS['maximum_relative_rotation_rad']
        and row['cube_speed_m_s']<=THRESHOLDS['max_hold_cube_speed_m_s']
        and row['cube_angular_speed_rad_s']<=THRESHOLDS['max_hold_cube_angular_speed_rad_s']
        and row['position_error_m']<=THRESHOLDS['position_warning_m']
        and row['orientation_error_rad']<=THRESHOLDS['orientation_warning_rad']
        and row['unexpected_contact_count']==0)


def sample(ctx,stage,step,wall_s,mujoco,np):
    m,d,r=ctx['model'],ctx['data'],ctx['reference']
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        raise RuntimeError('Nonfinite actual state')
    qref,qdref=references(ctx,stage,step,np)
    pd,gravity,raw,tau=motion.control_terms(ctx,qref,qdref,mujoco,np)
    d.ctrl[ctx['aids']]=tau
    # Same-state solver forces with this row's command; no time integration.
    mujoco.mj_forward(m,d)
    if not np.allclose(d.qfrc_actuator[ctx['va']],tau,rtol=0,atol=1e-9):
        raise RuntimeError('Actuator force and recorded motor command differ')
    r.qpos[:]=ctx['base_qpos'];r.qpos[ctx['qa']]=qref;r.qvel[:]=0
    mujoco.mj_forward(m,r)
    q=d.qpos[ctx['qa']].copy();qd=d.qvel[ctx['va']].copy()
    actual=d.site_xpos[ctx['site']].copy();ref=r.site_xpos[ctx['site']].copy()
    quat=np.empty(4);qrefquat=np.empty(4)
    mujoco.mju_mat2Quat(quat,d.site_xmat[ctx['site']]);mujoco.mju_mat2Quat(qrefquat,r.site_xmat[ctx['site']])
    angle=float(2*np.arccos(np.clip(abs(float(quat@qrefquat)),0.,1.)))
    forces=[0.,0.];counts=[0,0];table_contacts=0;unexpected=[];contact_details=[]
    cube=ctx['cube_geom'];table=ctx['table_geom']
    for ci,c in enumerate(d.contact[:d.ncon]):
        pair={int(c.geom1),int(c.geom2)}
        if cube in pair and (table in pair or any(g in pair for g in ctx['finger_geoms'])):
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
            # efc_address excludes inactive gap contacts; positive normal indicates support.
            active=c.efc_address>=0 and wrench[0]>0
            if active:
                if table in pair:table_contacts+=1
                for i,g in enumerate(ctx['finger_geoms']):
                    if g in pair:counts[i]+=1;forces[i]+=float(wrench[0])
            contact_details.append(dict(pair=sorted(pair),dist_m=float(c.dist),normal_n=float(wrench[0]),active=bool(active)))
        elif c.dist<=0:
            unexpected.append(dict(geoms=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in sorted(pair)],distance_m=float(c.dist)))
    finger_gap=min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in ctx['finger_meshes'])-ctx['cfg']['table']['top_z_m']
    cube_position=d.geom_xpos[cube].copy();cube_rotation=d.geom_xmat[cube].reshape(3,3)
    bottom=float(cube_position[2]-np.abs(cube_rotation[2])@m.geom_size[cube])
    relpos,relrot=cube_relative(ctx,np)
    slip=rotation_error=''
    if ctx['grip_baseline'] is not None:
        oldpos,oldrot=ctx['grip_baseline']
        slip=float(np.linalg.norm(relpos-oldpos))
        rotation_error=float(np.arccos(np.clip((np.trace(oldrot.T@relrot)-1)/2.,-1.,1.)))
    cube_jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'task_cube_free')
    if cube_jid<0:raise ValueError('Missing free cube joint')
    cv=int(m.jnt_dofadr[cube_jid]);cube_speed=float(np.linalg.norm(d.qvel[cv:cv+3]));cube_spin=float(np.linalg.norm(d.qvel[cv+3:cv+6]))
    row=dict(sim_s=float(d.time),wall_s=wall_s,phase=stage.phase,terminal_state=0,
        position_error_m=float(np.linalg.norm(actual-ref)),orientation_error_rad=angle,
        finger_ref_m=float(qref[7]),finger1_q_m=float(q[7]),finger2_q_m=float(q[8]),
        finger1_normal_n=forces[0],finger2_normal_n=forces[1],finger1_contact_count=counts[0],finger2_contact_count=counts[1],
        cube_table_contact_count=table_contacts,cube_bottom_clearance_m=bottom-ctx['cfg']['table']['top_z_m'],
        cube_lift_from_start_m=float(cube_position[2]-ctx['cube_initial'][2]),
        relative_slip_m=slip,relative_rotation_rad=rotation_error,grip_reference_valid=int(ctx['grip_baseline'] is not None),
        unexpected_contact_count=len(unexpected),max_torque_ratio=float(np.max(np.abs(raw[:7])/ctx['cap'][:7])),
        max_finger_force_ratio=float(np.max(np.abs(raw[7:])/ctx['cap'][7:])),min_finger_table_gap_m=finger_gap,
        max_joint_error_rad=float(np.max(np.abs(q[:7]-qref[:7]))),cube_speed_m_s=cube_speed,cube_angular_speed_rad_s=cube_spin,
        **dict(zip(('cube_x_m','cube_y_m','cube_z_m'),map(float,cube_position))),
        **dict(zip(('actual_x_m','actual_y_m','actual_z_m'),map(float,actual))),
        **dict(zip(('ref_x_m','ref_y_m','ref_z_m'),map(float,ref))))
    for i in range(9):
        for key,vector in (('q_ref',qref),('q',q),('qd_ref',qdref),('qd',qd),('pd',pd),('gravity',gravity),('tau_raw',raw),('tau',tau)):
            row[f'{key}_{i}']=float(vector[i])
    problems=[]
    if abs(d.time-step*DT)>1e-8:problems.append('unexpected_time')
    if row['position_error_m']>THRESHOLDS['position_stop_m'] or angle>THRESHOLDS['orientation_stop_rad']:problems.append('arm_pose_tracking_stop')
    if row['max_joint_error_rad']>.15 or np.max(np.abs(qd[:7]))>2.:problems.append('arm_joint_tracking_stop')
    if np.any(q[:7]<=ctx['limits'][:,0]+.02) or np.any(q[:7]>=ctx['limits'][:,1]-.02):problems.append('arm_limit_stop')
    if np.any(q[7:]<ctx['finger_limits'][:,0]-.0005) or np.any(q[7:]>ctx['finger_limits'][:,1]+.0005) or np.max(np.abs(qd[7:]))>.25:problems.append('finger_limit_or_speed_stop')
    if finger_gap<THRESHOLDS['min_finger_table_gap_m']:problems.append('finger_table_clearance_stop')
    if unexpected:problems.append('unexpected_contact')
    if row['cube_bottom_clearance_m']<-.003 or np.linalg.norm(cube_position[:2]-ctx['cube_initial'][:2])>.15:problems.append('cube_left_test_workspace')
    if ctx['grip_baseline'] is not None and (slip>.015 or rotation_error>.25):problems.append('object_slip_stop')
    if any(w.number for w in d.warning):problems.append('mujoco_warning')
    return row,problems,dict(unexpected=unexpected,allowed=contact_details)


def inspect(ctx,mujoco,np):
    initial=(ctx['data'].qpos.copy(),ctx['data'].qvel.copy(),ctx['data'].time)
    stage=Stages();row,problems,_=sample(ctx,stage,0,0.,mujoco,np)
    if problems:raise ValueError(str(problems))
    if not np.array_equal(ctx['data'].qpos,initial[0]) or not np.array_equal(ctx['data'].qvel,initial[1]) or ctx['data'].time!=initial[2]:
        raise AssertionError('Inspection changed integrated state')
    return dict(status='static_grasp_input_check_passed',physics_steps=0,
        open_reference_m=OPEN,closed_reference_m=CLOSED,software_caps=motion.CAP,
        frozen_lift_reference_indices=[2000,3000],lift_reference_duration_s=4.,
        reference_hand_height_gain_m=float(ctx['rows'][3000]['position_world_m'][2]-ctx['rows'][2000]['position_world_m'][2]),
        initial_row=row,limitation='No physical closing, lift or GUI run. User result pending.')


def run(ctx,report,out,mujoco,np):
    import mujoco.viewer
    m,d=ctx['model'],ctx['data'];stage=Stages();commands=queue.SimpleQueue()
    stable=Continuous();hold=Continuous();step=0;start=time.perf_counter();loop_start=None
    last=None;terminal_saved=False;last_phase=None;status='interrupted';reason='window_closed'
    arm_saturation_s=0.;contact_loss_s=0.;hold_good_last=False
    first,_,_=sample(ctx,stage,0,0.,mujoco,np)
    metrics=dict(max_position_error_m=0.,max_orientation_error_rad=0.,max_relative_slip_m=0.,
        maximum_cube_bottom_clearance_m=0.,minimum_finger_table_gap_m=first['min_finger_table_gap_m'],
        max_torque_ratio=0.,max_finger_force_ratio=0.,max_unexpected_contact_count=0)
    def write(writer,row,terminal=False):
        nonlocal terminal_saved,last
        if terminal:row=motion.terminal_row(row);terminal_saved=True
        writer.writerow(row);last=row;report['data_saved_samples']+=1
        for key,col in [('max_position_error_m','position_error_m'),('max_orientation_error_rad','orientation_error_rad'),
                        ('max_relative_slip_m','relative_slip_m'),('maximum_cube_bottom_clearance_m','cube_bottom_clearance_m'),
                        ('max_torque_ratio','max_torque_ratio'),('max_finger_force_ratio','max_finger_force_ratio'),
                        ('max_unexpected_contact_count','unexpected_contact_count')]:
            if row[col]!='':metrics[key]=max(metrics[key],row[col])
        metrics['minimum_finger_table_gap_m']=min(metrics['minimum_finger_table_gap_m'],row['min_finger_table_gap_m'])
    with (out/'grasp.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(first));writer.writeheader()
        try:
            with mujoco.viewer.launch_passive(m,d,key_callback=commands.put,show_left_ui=False,show_right_ui=False) as viewer:
                report['viewer_started']=True
                with viewer.lock():set_camera(viewer.cam,ctx['cfg'],'oblique')
                print('Click scene: S approach B; C close at B_ready; L lift at grip_ready; Q end. Camera 1/2/3.',flush=True)
                stop=False
                while viewer.is_running():
                    tick=time.perf_counter()
                    if tick-start>600 or (loop_start is not None and tick-loop_start>THRESHOLDS['maximum_physics_wall_s']):reason='wall_time_limit';break
                    while not commands.empty():
                        key=commands.get()
                        if key in (ord('1'),ord('2'),ord('3')):
                            with viewer.lock():set_camera(viewer.cam,ctx['cfg'],{49:'oblique',50:'top',51:'side'}[key])
                        elif key==ord('Q'):
                            report['events'].append(dict(key='Q',accepted=True,reason='user_end',sim_s=float(d.time),wall_s=tick-start))
                            reason='user_Q';stop=True;break
                        elif key in (ord('S'),ord('C'),ord('L')):
                            current,_,_=sample(ctx,stage,step,tick-start,mujoco,np)
                            ready=bilateral(current) and ctx['grip_baseline'] is not None and stable.since is not None and (step-stable.since)*DT>=THRESHOLDS['contact_stable_s']
                            accepted,note=stage.command(chr(key),step,ready)
                            report['events'].append(dict(key=chr(key),accepted=accepted,reason=note,sim_s=float(d.time),wall_s=tick-start))
                            print('KEY:',chr(key),note,flush=True)
                            if accepted and key==ord('S'):loop_start=tick;report['physics_started']=True
                    if stop:break
                    if stage.phase!='waiting_start':
                        for _ in range(4):
                            stage.advance(step)
                            row,problems,contacts=sample(ctx,stage,step,time.perf_counter()-start,mujoco,np)
                            both=bilateral(row)
                            stable_s=stable.observe(both and row['cube_speed_m_s']<.01,step)
                            if stage.phase=='grip_settle' and stage.age(step)>=1. and stable_s>=THRESHOLDS['contact_stable_s']:
                                ctx['grip_baseline']=tuple(x.copy() for x in cube_relative(ctx,np))
                                report['grip_baseline']=dict(sim_s=float(d.time),position_in_tool_m=ctx['grip_baseline'][0].tolist(),rotation_in_tool=ctx['grip_baseline'][1].tolist())
                                stage.enter('grip_ready',step)
                                row,problems,contacts=sample(ctx,stage,step,time.perf_counter()-start,mujoco,np)
                            if stage.phase=='grip_settle' and stage.age(step)>=3.:problems.append('bilateral_grip_not_established')
                            if stage.phase in ('B_ready','grip_ready') and stage.age(step)>THRESHOLDS['maximum_wait_phase_s']:problems.append('key_wait_timeout')
                            if d.time>=THRESHOLDS['maximum_simulation_s']:problems.append('simulation_time_limit')
                            arm_saturation_s=arm_saturation_s+DT if row['max_torque_ratio']>1 else 0.
                            if arm_saturation_s>.25:problems.append('sustained_arm_saturation')
                            contact_loss_s=contact_loss_s+DT if ctx['grip_baseline'] is not None and not both else 0.
                            if contact_loss_s>.1:problems.append('bilateral_contact_lost')
                            hold_good_last=stage.phase=='lift_hold' and stage.age(step)>=1.-1e-10 and hold_valid(row)
                            hold.observe(hold_good_last,step)
                            if stage.phase!=last_phase:
                                print(f'PHASE: {stage.phase} | sim {d.time:.3f} s',flush=True);last_phase=stage.phase
                                report['phases'].append(dict(phase=stage.phase,sim_s=float(d.time),step=step))
                            if problems:
                                report['stop_diagnostics']=dict(problems=problems,contacts=contacts,row=row)
                                status='stopped';reason=','.join(problems);write(writer,row,True);stop=True;break
                            if stage.phase=='lift_hold' and stage.age(step)>=3.-1e-10:
                                status='completed';reason='lift_hold_window_completed'
                                report['grasp_hold_passed']=bool(hold_good_last and hold.since is not None and (step-hold.since)*DT>=THRESHOLDS['hold_s']-1e-10)
                                write(writer,row,True);stop=True;break
                            write(writer,row)
                            mujoco.mj_step(m,d);step+=1;report['steps']=step
                        stream.flush()
                    with viewer.lock():motion.draw(viewer,ctx,mujoco,np)
                    display=last if last is not None else first
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        'CONTACT GRASP + LIFT | NO TRANSFER / RELEASE\nS:approach  C:close  L:lift  Q:end | Camera 1/2/3\n'
                        f'{stage.phase} | sim {d.time:.2f} s\n'
                        f'Finger normal: {display["finger1_normal_n"]:.2f} / {display["finger2_normal_n"]:.2f} N\n'
                        f'Cube bottom above table: {display["cube_bottom_clearance_m"]*1000:.1f} mm\n'
                        'C only at B_ready; L only at grip_ready. Premature keys are ignored.',''))
                    before=(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time))
                    viewer.sync()
                    if (not np.array_equal(d.qpos,before[0]) or not np.array_equal(d.qvel,before[1]) or not np.array_equal(d.ctrl,before[2]) or d.time!=before[3]
                        or np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or abs(m.opt.timestep-DT)>1e-12
                        or not np.array_equal(m.opt.gravity,[0,0,-9.81]) or int(m.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)):
                        raise RuntimeError('Viewer changed model/state; use only lesson keys/camera.')
                    if stop:time.sleep(1.2);break
                    time.sleep(max(0.,(4*DT if loop_start is not None else .02)-(time.perf_counter()-tick)))
        finally:
            if not terminal_saved:
                try:
                    stage.advance(step);row,_,_=sample(ctx,stage,step,time.perf_counter()-start,mujoco,np);write(writer,row,True)
                except Exception:report['terminal_record_error']=traceback.format_exc()
            report.update(steps=step,simulation_s=float(d.time),metrics=metrics,
                hold_longest_valid_s=hold.maximum_s,physics_loop_wall_s=None if loop_start is None else time.perf_counter()-loop_start,
                final_qpos=d.qpos.tolist(),final_qvel=d.qvel.tolist())
    report.update(status=status,end_reason=reason)
    report['tracking_thresholds_passed']=bool(status=='completed' and metrics['max_position_error_m']<=THRESHOLDS['position_warning_m'] and metrics['max_orientation_error_rad']<=THRESHOLDS['orientation_warning_rad'])
    return 0 if status in ('completed','interrupted') else 2


def main():
    guard()
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args();out=args.output.resolve() if args.output else ROOT/'outputs'/('grasp_lift_user_'+stamp())
    if out.parent!=(ROOT/'outputs').resolve() or not out.name.startswith('grasp_lift_user_'):raise ValueError('Use a fresh project grasp_lift_user_ output')
    if not args.inspect:out.mkdir(exist_ok=False)
    os.chdir(ROOT/'outputs' if args.inspect else out)
    report=dict(kind='grasp_lift_dynamics',status='started',model_loaded=False,viewer_started=False,physics_started=False,
        steps=0,data_saved_samples=0,simulation_s=0.,events=[],phases=[],grasp_hold_passed=False,tracking_thresholds_passed=False,learner_verified=False,
        transport_validated=False,release_validated=False,thresholds=THRESHOLDS,controller='PD_plus_zero_velocity_gravity',
        finger_open_m=OPEN,finger_closed_target_m=CLOSED,kp=motion.KP,kd=motion.KD,software_caps=motion.CAP,
        executable=sys.executable,temp=tempfile.gettempdir(),snapshots=[],
        sample_semantics='Pre-step actual state, reference and current motor commands. Contact forces from mj_forward with current controls, before integration. Terminal has no future control fields.',
        contact_semantics='Sum of positive normal forces over active cube contacts for each finger; N, not motor force. Relative cube/tool transform fixed once after settled bilateral contact; missing before this baseline.',
        scope='Contact close and lift/hold only. No attachment, object teleportation, transfer, release or full v0.1 acceptance.')
    started=time.perf_counter();code=0
    try:
        import numpy as np
        import mujoco
        ctx=setup(mujoco,np);inspection=inspect(ctx,mujoco,np)
        report.update(model_loaded=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,dt_s=DT,static_inspection=inspection,source_reference=str(ctx['reference_file']))
        if args.inspect:print(json.dumps(inspection,ensure_ascii=False,indent=2));return 0
        files=[(ctx['folder']/n,'source/'+n) for n in ('scene.xml','config.json','static_audit.json')]
        files.append((ctx['reference_file'],'source/path.json'))
        files.append((ctx['contact_geometry_receipt'],'source/contact_geometry_receipt.json'))
        files += [(p,'code/'+p.name) for p in Path(__file__).parent.glob('*.py')]
        files += [(Path(__file__).with_name(n),'code/'+n) for n in ('reachable_preview_source.json','reachable_motion_source.json','grasp_lift_source.json','Run-GraspLift.ps1','Show-GraspLift.ps1')]
        files.append((ROOT/'repos/g1_grasp_assets/asset_manifest.json','source/asset_manifest.json'))
        for source,relative in files:
            dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            report['snapshots'].append(dict(path=relative,sha256=sha(dest)))
        print('GRASP LIFT RUN:',out,flush=True);code=run(ctx,report,out,mujoco,np)
    except KeyboardInterrupt:report.update(status='interrupted',end_reason='KeyboardInterrupt',grasp_hold_passed=False);code=130
    except Exception:
        report.update(status='error',end_reason='exception',grasp_hold_passed=False,error=traceback.format_exc());code=1;print(report['error'],file=sys.stderr)
        if not args.inspect:(out/'error.txt').write_text(report['error'],encoding='utf-8')
    finally:
        report['wall_s']=time.perf_counter()-started
        if not args.inspect:
            if (out/'grasp.csv').exists():report['grasp_csv_sha256']=sha(out/'grasp.csv')
            save_json(out/'summary.json',report);print('SUMMARY:',out/'summary.json',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
