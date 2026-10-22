"""User-operated local Cartesian keyboard practice; no grasp or route replay.

References are planned in separate data. Only motor controls and mj_step move
the live arm after setup. --inspect never opens a viewer or integrates physics.
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
import carry_to_tray as carry
import grasp_lift as base
import grasp_solver_runtime as frozen
from pickplace_layout_preview import set_camera
from cartesian_keyboard_target import TargetPlanner, prepare_initial_pose, SETTINGS as PLANNER_SETTINGS

DT = base.DT
STEP_M = .005
THRESHOLDS = dict(base.THRESHOLDS,
    maximum_cube_xy_displacement_m=.003, maximum_cube_z_displacement_m=.005,
    ready_arm_speed_rad_s=.05, ready_finger_speed_m_s=.01,
    ready_finger_error_m=.003, ready_stable_s=.25,
    settling_timeout_s=5., maximum_simulation_s=120., maximum_physics_wall_s=240.)
KEY_COMMANDS = {265: 'X+', 264: 'X-', 263: 'Y+', 262: 'Y-',
                266: 'Z+', 267: 'Z-', ord('6'): 'GRIPPER_TOGGLE'}
DELTAS = {'X+': (.005,0.,0.), 'X-': (-.005,0.,0.),
          'Y+': (0.,.005,0.), 'Y-': (0.,-.005,0.),
          'Z+': (0.,0.,.005), 'Z-': (0.,0.,-.005)}
KEY_NAMES = {265:'Up',264:'Down',263:'Left',262:'Right',266:'PageUp',267:'PageDown',
             ord('6'):'6',ord('7'):'7',ord('8'):'8',ord('9'):'9'}
SUCCESS = ROOT/'outputs/carry_user_20261021_103558_374'
SUCCESS_SHA = '6f05af11b27f618cb9dc5ede67d79e0b4bc0465ca0b46fee154d64483144691f'
KEY_EVIDENCE = dict(
    native_repeat='MuJoCo 3.3.7 glfw_adapter.cc IsKeyDownEvent accepts GLFW_PRESS only; repeat/release ignored.',
    arrows='simulate.cc Left/Right history stepping requires !is_passive_; this uses launch_passive.',
    page_up='PageUp changes selected parent only; viewer perturbation selection is cleared each frame.',
    gripper='6 is unassigned in this version; C and O are visualization shortcuts and are not used.',
    sources=['https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/simulate/glfw_adapter.cc',
             'https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/python/mujoco/simulate.cc',
             'https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/simulate/simulate.cc',
             'https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/src/engine/engine_vis_init.c'])


STATIC_VIEWER_HELP = (
    'LOCAL CARTESIAN KEYBOARD | EMPTY HAND | NOT GRASPING\n'
    '7:start  8:camera  9:end  6:gripper toggle\n'
    'Up/Down: world +/-X; Left/Right: world +/-Y\n'
    'PageUp/PageDown: world +/-Z (NOT screen directions)\n'
    'One short press = 5 mm. Wait for the READY label by the hand.\n'
    'Pink: accepted goal; blue: start. Busy keys rejected.\n'
    'This help stays fixed; live status is next to the hand.')


def update_status_scene(viewer, ctx, session, display, mujoco, np):
    """Called under viewer.lock; labels avoid the native overlay request queue."""
    viewer.perturb.select = 0
    viewer.perturb.active = 0
    scene = viewer.user_scn
    scene.ngeom = 0
    if scene.maxgeom < 3:
        raise RuntimeError('Viewer scene needs three status geometries')
    for index, (point, color) in enumerate(((ctx['keyboard_anchor'], [.3,.6,1.,.9]),
                                           (session.goal, [1.,.1,.5,.9]))):
        mujoco.mjv_initGeom(scene.geoms[index], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([.004,.004,.004]), point, np.eye(3).ravel(), np.asarray(color,dtype=np.float32))
    label = scene.geoms[2]
    position = np.asarray([display['actual_x_m'], display['actual_y_m'], display['actual_z_m']]) + [0.,0.,.07]
    mujoco.mjv_initGeom(label, mujoco.mjtGeom.mjGEOM_LABEL,
        np.zeros(3), position, np.eye(3).ravel(), np.array([1.,1.,1.,1.],dtype=np.float32))
    label.label = (f'{session.phase.upper()} | sim {ctx["data"].time:.2f}s | '
                   f'error {display["position_error_m"]*1000:.2f} mm')[:95]
    scene.ngeom = 3


def setup(mujoco, np):
    if sha(SUCCESS/'summary.json') != SUCCESS_SHA:
        raise ValueError('Accepted carry summary changed')
    accepted = json.loads((SUCCESS/'summary.json').read_text(encoding='utf-8-sig'))
    if not (accepted['status']=='completed' and accepted['placement_passed']
            and accepted['tracking_thresholds_passed']):
        raise ValueError('Expected the accepted full fixed-reference carry run')
    snapshots = {item['path']:item['sha256'] for item in accepted['snapshots']}
    for name in ('carry_to_tray.py','carry_runtime.py','carry_route.py','grasp_contact_geometry.py'):
        expected = snapshots['code/'+name]
        if sha(SUCCESS/'code'/name)!=expected or sha(Path(__file__).with_name(name))!=expected:
            raise ValueError('Frozen carry dependency changed: '+name)
    ctx = carry.setup(mujoco,np)
    # Verify the original A-state protocol before initializing this different lesson.
    ctx['keyboard_base_inspection'] = carry.inspect(ctx,mujoco,np)
    prepare_initial_pose(ctx,mujoco,np)
    return ctx


class KeyboardSession:
    """Pure command state. Busy commands are rejected, never queued."""
    def __init__(self, ctx, planner, np):
        self.np = np
        self.planner = planner
        self.phase = 'waiting_start'
        self.q_hold = ctx['data'].qpos[ctx['qa']].copy()
        self.goal = ctx['keyboard_anchor'].copy()
        self.plan = None
        self.start_step = 0
        self.settle_step = 0
        self.ready_since = None
        self.command_id = 0
        self.completed_ids = []
        self.pending_id = None
        self.command_records = []
        self.generation = 0

    def start(self, step):
        if self.phase != 'waiting_start':
            return False,'already_started'
        self.phase = 'settling'
        self.settle_step = step
        return True,'started_hold_at_keyboard_anchor'

    def advance(self, step):
        if self.plan is not None and step-self.start_step >= len(self.plan['rows'])-1:
            self.q_hold = self.np.asarray(self.plan['rows'][-1]['q_ref9'],dtype=float).copy()
            self.plan = None
            self.phase = 'settling'
            self.settle_step = step
            self.ready_since = None

    def reference(self, step):
        if self.plan is None:
            return self.q_hold.copy(),self.np.zeros(9)
        index=max(0,min(step-self.start_step,len(self.plan['rows'])-1))
        row=self.plan['rows'][index]
        return self.np.asarray(row['q_ref9'],dtype=float),self.np.asarray(row['dq_ref9'],dtype=float)

    def observe(self, row, step):
        if self.phase not in ('settling','ready'):
            self.ready_since=None
            return
        if ready_valid(row):
            if self.ready_since is None:
                self.ready_since=step
            if (step-self.ready_since)*DT>=THRESHOLDS['ready_stable_s']-1e-10:
                self.phase='ready'
                if self.pending_id is not None:
                    self.completed_ids.append(self.pending_id)
                    self.pending_id=None
        else:
            if self.phase=='ready':
                self.settle_step=step
            self.phase='settling'
            self.ready_since=None

    def request(self, command, step):
        if self.phase != 'ready':
            return dict(accepted=False,reason='busy_or_not_ready_no_queue',command=command)
        # The callback records this phase/generation before expensive planning.
        # Events received while planning or left over from a previous command
        # cannot become future commands merely because processing was delayed.
        self.phase='planning'
        self.generation+=1
        destination=None
        if command == 'GRIPPER_TOGGLE':
            destination=base.CLOSED if self.q_hold[7]>(base.OPEN+base.CLOSED)/2 else base.OPEN
            plan=self.planner.plan_fingers(destination,self.q_hold.copy())
        else:
            plan=self.planner.plan_delta(DELTAS[command],self.q_hold.copy())
        if not plan['accepted']:
            self.phase='ready'
            return plan
        rows=plan['rows']
        if len(rows)<2 or not self.np.allclose(rows[0]['q_ref9'],self.q_hold,rtol=0,atol=1e-10):
            raise ValueError('New reference does not begin at the held reference')
        self.command_id+=1
        self.pending_id=self.command_id
        self.command_records.append(dict(command_id=self.command_id,command=command,
            finger_target_m=destination))
        self.plan=plan
        self.goal=self.np.asarray(plan['target_xyz'],dtype=float).copy()
        self.start_step=step
        self.phase='moving'
        self.ready_since=None
        return plan


def ready_valid(row):
    return bool(row['position_error_m']<=THRESHOLDS['position_warning_m']
        and row['orientation_error_rad']<=THRESHOLDS['orientation_warning_rad']
        and row['target_position_error_m']<=THRESHOLDS['position_warning_m']
        and row['target_orientation_error_rad']<=THRESHOLDS['orientation_warning_rad']
        and row['max_joint_speed_rad_s']<=THRESHOLDS['ready_arm_speed_rad_s']
        and row['max_finger_speed_m_s']<=THRESHOLDS['ready_finger_speed_m_s']
        and row['max_finger_error_m']<=THRESHOLDS['ready_finger_error_m']
        and row['unexpected_contact_count']==0)


def stale_event(session, received_phase, received_generation):
    return received_phase!='ready' or received_generation!=session.generation


def completion_checks(session, reason, metrics, step):
    completed=[r for r in session.command_records if r['command_id'] in session.completed_ids]
    directions=sorted({r['command'] for r in completed if r['command'] in DELTAS})
    finger_targets=[r['finger_target_m'] for r in completed if r['command']=='GRIPPER_TOGGLE']
    tracking=bool(step>0 and metrics['max_position_error_m']<=THRESHOLDS['position_warning_m']
        and metrics['max_orientation_error_rad']<=THRESHOLDS['orientation_warning_rad'])
    closed=any(abs(x-base.CLOSED)<1e-12 for x in finger_targets)
    opened=any(abs(x-base.OPEN)<1e-12 for x in finger_targets)
    passed=bool(reason=='user_9' and session.phase=='ready' and tracking
        and set(directions)==set(DELTAS) and closed and opened)
    return dict(practice_completed=passed,tracking_thresholds_passed=tracking,
        completed_directions=directions,gripper_closed_completed=closed,gripper_opened_completed=opened,
        completed_command_records=completed,
        practice_requirement='Each of six world directions and gripper closed/open endpoints completed; ready at user 9 exit; tracking thresholds passed.')


def sample(ctx, session, step, wall_s, mujoco, np):
    m,d,r=ctx['model'],ctx['data'],ctx['reference']
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        raise RuntimeError('Nonfinite actual state')
    qref,qdref=session.reference(step)
    pd,gravity,raw,tau=base.motion.control_terms(ctx,qref,qdref,mujoco,np)
    d.ctrl[ctx['aids']]=tau
    mujoco.mj_forward(m,d)
    if not np.allclose(d.qfrc_actuator[ctx['va']],tau,rtol=0,atol=1e-9):
        raise RuntimeError('Recorded command differs from actuator force')
    r.qpos[:]=ctx['base_qpos'];r.qpos[ctx['qa']]=qref;r.qvel[:]=0
    mujoco.mj_forward(m,r)
    actual=d.site_xpos[ctx['site']].copy();ref=r.site_xpos[ctx['site']].copy()
    actual_q=np.empty(4);ref_q=np.empty(4)
    mujoco.mju_mat2Quat(actual_q,d.site_xmat[ctx['site']])
    mujoco.mju_mat2Quat(ref_q,r.site_xmat[ctx['site']])
    orientation=float(2*np.arccos(np.clip(abs(float(actual_q@ref_q)),0.,1.)))
    orientation_task=float(2*np.arccos(np.clip(abs(float(actual_q@ctx['keyboard_orientation'])),0.,1.)))
    q=d.qpos[ctx['qa']].copy();qd=d.qvel[ctx['va']].copy()
    finger_gap=min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
                   for g,v in ctx['finger_meshes'])-ctx['cfg']['table']['top_z_m']
    unexpected=[]
    for contact in d.contact[:d.ncon]:
        pair={int(contact.geom1),int(contact.geom2)}
        if pair!={ctx['cube_geom'],ctx['table_geom']} and contact.dist<=0:
            unexpected.append(dict(geoms=sorted(pair),distance_m=float(contact.dist)))
    cube=d.geom_xpos[ctx['cube_geom']].copy();delta=cube-ctx['cube_initial']
    row=dict(sim_s=float(d.time),wall_s=wall_s,phase=session.phase,terminal_state=0,
        command_id=session.command_id,busy=int(session.phase in ('planning','moving','settling')),ready=int(session.phase=='ready'),
        **dict(zip(('target_x_m','target_y_m','target_z_m'),map(float,session.goal))),
        **dict(zip(('ref_x_m','ref_y_m','ref_z_m'),map(float,ref))),
        **dict(zip(('actual_x_m','actual_y_m','actual_z_m'),map(float,actual))),
        position_error_m=float(np.linalg.norm(actual-ref)),orientation_error_rad=orientation,
        target_position_error_m=float(np.linalg.norm(actual-session.goal)),
        reference_target_error_m=float(np.linalg.norm(ref-session.goal)),target_orientation_error_rad=orientation_task,
        finger_ref_m=float(qref[7]),finger1_q_m=float(q[7]),finger2_q_m=float(q[8]),
        min_finger_table_gap_m=finger_gap,unexpected_contact_count=len(unexpected),
        max_joint_error_rad=float(np.max(np.abs(q[:7]-qref[:7]))),
        max_joint_speed_rad_s=float(np.max(np.abs(qd[:7]))),
        max_finger_speed_m_s=float(np.max(np.abs(qd[7:]))),max_finger_error_m=float(np.max(np.abs(q[7:]-qref[7:]))),
        max_torque_ratio=float(np.max(np.abs(raw[:7])/ctx['cap'][:7])),
        max_finger_force_ratio=float(np.max(np.abs(raw[7:])/ctx['cap'][7:])),
        cube_displacement_m=float(np.linalg.norm(delta)),
        **dict(zip(('cube_x_m','cube_y_m','cube_z_m'),map(float,cube))))
    for i in range(9):
        for key,vector in (('q_ref',qref),('q',q),('qd_ref',qdref),('qd',qd),('pd',pd),('gravity',gravity),('tau_raw',raw),('tau',tau)):
            row[f'{key}_{i}']=float(vector[i])
    problems=[]
    if abs(d.time-step*DT)>1e-8:problems.append('unexpected_time')
    if row['position_error_m']>THRESHOLDS['position_stop_m'] or orientation>THRESHOLDS['orientation_stop_rad']:problems.append('arm_pose_tracking_stop')
    if row['max_joint_error_rad']>THRESHOLDS['max_joint_error_rad'] or row['max_joint_speed_rad_s']>THRESHOLDS['max_joint_speed_rad_s']:problems.append('arm_joint_tracking_stop')
    if np.any(q[:7]<=ctx['limits'][:,0]+.02) or np.any(q[:7]>=ctx['limits'][:,1]-.02):problems.append('arm_limit_stop')
    if np.any(q[7:]<ctx['finger_limits'][:,0]-.0005) or np.any(q[7:]>ctx['finger_limits'][:,1]+.0005) or row['max_finger_speed_m_s']>THRESHOLDS['max_finger_speed_m_s']:problems.append('finger_limit_or_speed_stop')
    if finger_gap<THRESHOLDS['min_finger_table_gap_m']:problems.append('finger_table_clearance_stop')
    if unexpected:problems.append('unexpected_contact')
    if np.linalg.norm(delta[:2])>THRESHOLDS['maximum_cube_xy_displacement_m'] or abs(delta[2])>THRESHOLDS['maximum_cube_z_displacement_m']:problems.append('cube_displacement_stop')
    if any(w.number for w in d.warning):problems.append('mujoco_warning')
    return row,problems,unexpected


def inspect(ctx,mujoco,np):
    original=ctx['keyboard_base_inspection']
    before=(ctx['data'].qpos.copy(),ctx['data'].qvel.copy(),ctx['data'].time)
    planner=TargetPlanner(ctx,mujoco,np)
    session=KeyboardSession(ctx,planner,np)
    row,problems,_=sample(ctx,session,0,0.,mujoco,np)
    if problems:raise RuntimeError('Keyboard static input check: '+','.join(problems))
    if not np.array_equal(before[0],ctx['data'].qpos) or not np.array_equal(before[1],ctx['data'].qvel) or before[2]!=ctx['data'].time:
        raise RuntimeError('Static input check changed integrated state')
    return planner,dict(status='keyboard_static_input_check_passed',physics_steps=0,
        original_carry_inspection=original['status'],initial_row=row,
        anchor_position_world_m=ctx['keyboard_anchor'].tolist(),target_fixed_orientation_wxyz=ctx['keyboard_orientation'].tolist(),
        step_m=STEP_M,thresholds=THRESHOLDS,key_evidence=KEY_EVIDENCE,planner_settings=PLANNER_SETTINGS,
        local_bounds_m=dict(relative_min=[-PLANNER_SETTINGS['workspace_half_extent_m']]*3,
            relative_max=[PLANNER_SETTINGS['workspace_half_extent_m']]*3,
            world_min=(ctx['keyboard_anchor']-PLANNER_SETTINGS['workspace_half_extent_m']).tolist(),
            world_max=(ctx['keyboard_anchor']+PLANNER_SETTINGS['workspace_half_extent_m']).tolist()),
        limitation='No keyboard movement, gripper physical motion or GUI has been run.')


def jsonable(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def run(ctx,planner,report,out,mujoco,np):
    import mujoco.viewer
    m,d=ctx['model'],ctx['data']
    session=KeyboardSession(ctx,planner,np)
    commands=queue.SimpleQueue()
    started=time.perf_counter();physics_started=None
    step=0;last=None;terminal_saved=False;camera_index=0
    status,reason='interrupted','window_closed'
    arm_saturation_s=0.;stop=False
    options=frozen.option_snapshot(m,np)
    report['frozen_physics_options']=options
    first,problems,_=sample(ctx,session,0,0.,mujoco,np)
    if problems:raise RuntimeError('Initial sample failed: '+str(problems))
    metrics=dict(max_position_error_m=0.,max_orientation_error_rad=0.,max_joint_error_rad=0.,
        max_torque_ratio=0.,max_finger_force_ratio=0.,max_cube_displacement_m=0.,max_unexpected_contact_count=0)

    def write(writer,row,terminal=False):
        nonlocal last,terminal_saved
        if terminal:
            row=base.motion.terminal_row(row);terminal_saved=True
        writer.writerow(row);last=row
        report['data_saved_samples']+=1
        for key,col in (('max_position_error_m','position_error_m'),('max_orientation_error_rad','orientation_error_rad'),
                        ('max_joint_error_rad','max_joint_error_rad'),('max_torque_ratio','max_torque_ratio'),
                        ('max_finger_force_ratio','max_finger_force_ratio'),('max_cube_displacement_m','cube_displacement_m'),
                        ('max_unexpected_contact_count','unexpected_contact_count')):
            if row[col]!='':metrics[key]=max(metrics[key],row[col])

    def callback(key):
        commands.put((int(key),time.perf_counter(),session.phase,session.generation))

    with (out/'keyboard.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(first));writer.writeheader()
        try:
            with mujoco.viewer.launch_passive(m,d,key_callback=callback,show_left_ui=False,show_right_ui=False) as viewer:
                report['viewer_started']=True
                with viewer.lock():
                    set_camera(viewer.cam,ctx['cfg'],frozen.CAMERAS[camera_index])
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False
                # MuJoCo 3.3.7 SetTexts waits on a pending request while holding
                # the GIL; a key callback can wait for that GIL on the UI thread.
                # Publish help only once for this fresh viewer, never in the loop.
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT, STATIC_VIEWER_HELP, ''))
                report['viewer_status_mode'] = 'one_static_overlay_and_locked_scene_label'
                last_display_phase = None
                print('Click scene. 7 starts; arrows world XY, PageUp/Down world Z; 6 toggle gripper; 8 camera; 9 finish.',flush=True)
                print('One short press = 5 mm / 1 simulation s. Wait for READY. Busy keys are rejected, not queued.',flush=True)
                while viewer.is_running():
                    tick=time.perf_counter()
                    frozen.assert_options_unchanged(m,options,np)
                    if tick-started>600 or (physics_started is not None and tick-physics_started>THRESHOLDS['maximum_physics_wall_s']):
                        reason='wall_time_limit';break
                    while not commands.empty():
                        key,received,received_phase,received_generation=commands.get()
                        event=dict(key=KEY_NAMES.get(key,str(key)),command=KEY_COMMANDS.get(key,'OTHER'),
                            sim_s=float(d.time),wall_s=received-started,accepted=False,reason='unsupported_key',
                            command_id=session.command_id,accepted_target_xyz=session.goal.tolist(),
                            received_phase=received_phase,received_generation=received_generation,
                            process_delay_wall_s=time.perf_counter()-received)
                        with (out/'events.jsonl').open('a',encoding='utf-8') as event_stream:
                            event_stream.write(json.dumps(jsonable(dict(stage='received',**event)),ensure_ascii=False)+'\n')
                        if key==ord('9'):
                            event.update(accepted=True,command='END',reason='user_end')
                            reason='user_9';stop=True
                        elif key==ord('8'):
                            camera_index=(camera_index+1)%len(frozen.CAMERAS)
                            with viewer.lock():set_camera(viewer.cam,ctx['cfg'],frozen.CAMERAS[camera_index])
                            event.update(accepted=True,command='CAMERA',reason='camera_changed')
                        elif key==ord('7'):
                            accepted,note=session.start(step)
                            event.update(accepted=accepted,command='START',reason=note)
                            if accepted:
                                physics_started=tick;report['physics_started']=True
                        elif key in KEY_COMMANDS:
                            command=KEY_COMMANDS[key]
                            event['requested_xyz']=(session.goal+np.asarray(DELTAS[command])).tolist() if command in DELTAS else session.goal.tolist()
                            stale=stale_event(session,received_phase,received_generation)
                            if session.phase=='ready' and not stale:
                                current,current_problems,_=sample(ctx,session,step,time.perf_counter()-started,mujoco,np)
                                if current_problems or not ready_valid(current):
                                    session.phase='settling';session.settle_step=step;session.ready_since=None
                            planning_start=time.perf_counter()
                            if stale:
                                result=dict(accepted=False,reason='received_busy_or_stale_no_queue')
                            else:
                                if session.phase=='ready':print('PLANNING:',command,'(simulation time paused)',flush=True)
                                result=session.request(command,step)
                            event.update(accepted=bool(result['accepted']),reason=result['reason'],command_id=session.command_id,
                                accepted_target_xyz=session.goal.tolist(),diagnostics=jsonable(result.get('diagnostics',{})),
                                planning_wall_s=time.perf_counter()-planning_start)
                            if result['accepted']:
                                path=out/'command_references'/f'{session.command_id:04d}.json'
                                save_json(path,jsonable(dict(command=command,start_step=step,**result)))
                                event.update(reference_path=str(path.relative_to(out)),reference_sha256=sha(path))
                        report['events'].append(event)
                        with (out/'events.jsonl').open('a',encoding='utf-8') as event_stream:
                            event_stream.write(json.dumps(jsonable(dict(stage='processed',**event)),ensure_ascii=False)+'\n')
                        print('KEY:',event['key'],event['reason'],flush=True)
                        if stop:break
                    if stop:break
                    if session.phase!='waiting_start':
                        for _ in range(4):
                            frozen.assert_options_unchanged(m,options,np)
                            session.advance(step)
                            row,problems,contacts=sample(ctx,session,step,time.perf_counter()-started,mujoco,np)
                            session.observe(row,step)
                            row.update(phase=session.phase,busy=int(session.phase in ('moving','settling')),ready=int(session.phase=='ready'))
                            arm_saturation_s=arm_saturation_s+DT if row['max_torque_ratio']>1. else 0.
                            if arm_saturation_s>.25:problems.append('sustained_arm_saturation')
                            if session.phase=='settling' and (step-session.settle_step)*DT>THRESHOLDS['settling_timeout_s']:
                                problems.append('settling_timeout')
                            if problems:
                                report['stop_diagnostics']=dict(problems=problems,contacts=contacts,row=row)
                                status,reason='stopped',','.join(problems);write(writer,row,True);stop=True;break
                            if d.time>=THRESHOLDS['maximum_simulation_s']-1e-10:
                                reason='simulation_time_limit';write(writer,row,True);stop=True;break
                            write(writer,row)
                            mujoco.mj_step(m,d)
                            step+=1;report['steps']=step
                        stream.flush()
                    display=last if last is not None else first
                    with viewer.lock():
                        update_status_scene(viewer,ctx,session,display,mujoco,np)
                    if session.phase != last_display_phase:
                        print('PHASE:',session.phase.upper(),f'| sim {d.time:.3f}s',flush=True)
                        last_display_phase = session.phase
                    before=(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time))
                    viewer.sync()
                    frozen.assert_options_unchanged(m,options,np)
                    if (not np.array_equal(d.qpos,before[0]) or not np.array_equal(d.qvel,before[1])
                        or not np.array_equal(d.ctrl,before[2]) or d.time!=before[3]
                        or np.any(d.xfrc_applied) or np.any(d.qfrc_applied)):
                        raise RuntimeError('Viewer changed actual state or applied forces')
                    if stop:break
                    time.sleep(max(0.,(4*DT if physics_started is not None else .02)-(time.perf_counter()-tick)))
        finally:
            if not terminal_saved:
                try:
                    row,terminal_problems,terminal_contacts=sample(ctx,session,step,time.perf_counter()-started,mujoco,np)
                    if terminal_problems:
                        status,reason='stopped','terminal:'+','.join(terminal_problems)
                        report['stop_diagnostics']=dict(problems=terminal_problems,contacts=terminal_contacts,row=row)
                    if session.phase=='ready' and not ready_valid(row):
                        session.phase='settling'
                        row.update(phase=session.phase,busy=1,ready=0)
                    write(writer,row,True)
                except Exception:
                    status,reason='error','terminal_record_failed'
                    report['terminal_record_error']=traceback.format_exc()
            report.update(steps=step,simulation_s=float(d.time),metrics=metrics,
                completed_command_ids=session.completed_ids,final_phase=session.phase,
                final_qpos=d.qpos.tolist(),final_qvel=d.qvel.tolist(),final_qacc_warmstart=d.qacc_warmstart.tolist(),
                physics_loop_wall_s=None if physics_started is None else time.perf_counter()-physics_started)
    checks=completion_checks(session,reason,metrics,step)
    if checks['practice_completed']:status='completed'
    elif reason=='user_9':status='partial'
    report.update(status=status,end_reason=reason,**checks,
        planning_wall_s_total=sum(e.get('planning_wall_s',0.) for e in report['events']))
    return 0 if status in ('completed','partial','interrupted') else 2


def main():
    guard()
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve() if args.output else ROOT/'outputs'/('keyboard_user_'+stamp())
    if out.parent!=(ROOT/'outputs').resolve() or not out.name.startswith('keyboard_user_'):
        raise ValueError('Use a fresh flat outputs/keyboard_user_ directory')
    if not args.inspect:out.mkdir(exist_ok=False)
    os.chdir(ROOT/'outputs' if args.inspect else out)
    report=dict(kind='cartesian_keyboard_practice',status='started',model_loaded=False,viewer_started=False,
        physics_started=False,steps=0,data_saved_samples=0,simulation_s=0.,events=[],snapshots=[],
        practice_completed=False,grasp_passed=False,placement_passed=False,full_v01_accepted=False,
        tracking_thresholds_passed=False,thresholds=THRESHOLDS,key_evidence=KEY_EVIDENCE,
        keyboard_step_m=STEP_M,controller='PD_plus_zero_velocity_gravity',kp=base.motion.KP,kd=base.motion.KD,
        software_caps=base.motion.CAP,executable=sys.executable,temp=tempfile.gettempdir(),
        accepted_carry_path=str(SUCCESS),accepted_carry_summary_sha256=SUCCESS_SHA,
        scope='User-selected local Cartesian goals around the derived high anchor, preserving endpoint orientation; empty hand only. Joint interpolation is not an exact Cartesian line.',
        sample_semantics='Pre-step samples with target accepted endpoint, current reference FK and actual physical position; terminal future controls blank.')
    started=time.perf_counter();code=0
    try:
        import numpy as np
        import mujoco
        ctx=setup(mujoco,np)
        planner,inspection=inspect(ctx,mujoco,np)
        report.update(model_loaded=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,dt_s=DT,
            static_inspection=inspection,anchor_position_world_m=ctx['keyboard_anchor'].tolist(),
            target_fixed_orientation_wxyz=ctx['keyboard_orientation'].tolist(),
            local_bounds_m=inspection['local_bounds_m'],planner_settings=PLANNER_SETTINGS,
            initial_qpos=ctx['data'].qpos.tolist(),initial_qvel=ctx['data'].qvel.tolist(),
            effective_solver_options=ctx['solver_effective_options'],
            keyboard_initialization=ctx['keyboard_initialization'])
        if args.inspect:
            print(json.dumps(jsonable(inspection),ensure_ascii=False,indent=2));return 0
        carry.snapshot(ctx,out,report)
        for source,relative in [(SUCCESS/'summary.json','accepted_carry/summary.json'),
            (Path(ctx['keyboard_initialization']['path']),'keyboard_reference/anchor.json')]+[
            (Path(__file__).with_name(name),'code/'+name) for name in ('Run-CartesianKeyboard.ps1','Show-CartesianKeyboard.ps1')]:
            dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            report['snapshots'].append(dict(path=relative,sha256=sha(dest)))
        (out/'command_references').mkdir()
        print('KEYBOARD RUN:',out,flush=True)
        # CPython 3.13.9 predates gh-140815's invalid-frame dump fix.
        # Keep the flushed CSV and key events; do not inspect live stacks
        # asynchronously while the viewer and simulation threads are running.
        report['diagnostic_mode']='csv_and_events_no_async_stack_sampler'
        code=run(ctx,planner,report,out,mujoco,np)
    except KeyboardInterrupt:
        report.update(status='interrupted',end_reason='KeyboardInterrupt',practice_completed=False);code=130
    except Exception:
        report.update(status='error',end_reason='exception',practice_completed=False,error=traceback.format_exc());code=1
        print(report['error'],file=sys.stderr)
        if not args.inspect:(out/'error.txt').write_text(report['error'],encoding='utf-8')
    finally:
        report['wall_s']=time.perf_counter()-started
        if not args.inspect:
            if (out/'keyboard.csv').exists():report['keyboard_csv_sha256']=sha(out/'keyboard.csv')
            save_json(out/'summary.json',jsonable(report));print('SUMMARY:',out/'summary.json',flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
