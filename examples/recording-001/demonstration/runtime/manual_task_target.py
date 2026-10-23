"""Per-key Cartesian planning for the bounded, fixed-layout manual task.

All motion targets originate in user key increments. No saved carry route is
executed. The original model/controller are unchanged. Object attachment below
is a SCRATCH-DATA geometric envelope only; the live free object is never set.
"""
import json
from pathlib import Path
import time

import cartesian_keyboard_target as local
from lesson_common import ROOT, save_json, sha

DT, OPEN, CLOSED = local.DT, local.OPEN, local.CLOSED
SETTINGS = dict(local.SETTINGS, workspace_min_offset_from_B_m=[-.020, -.020, 0.],
    workspace_max_offset_from_B_m=[.120, .175, .090], minimum_environment_gap_m=.003,
    path_line_deviation_m=.001,
    minimum_payload_gap_m=.003, minimum_finger_table_gap_m=.001,
    cube_support_penetration_tolerance_m=.0002,
    minimum_payload_floor_gap_m=.004, minimum_open_payload_floor_gap_m=.002,
    jog_maximum_delta_m=.0015,
    jog_duration_s=.12, support_escape_monotonic_tolerance_m=1e-7)
SETTINGS.pop('workspace_half_extent_m')
# Derived by bounded static IK from the original A pose, not by dynamics.
INITIAL_Q9 = [-.7024653553213133,-.5730631094387038,-.6941401030898114,
    .6755982153407619,-1.0554993590673116,1.1128062836247814,
    1.1653887890399983, OPEN, OPEN]


def prepare_initial_pose(ctx, mujoco, np):
    """Explicit new lesson initialization at B + world Z 75 mm, once only."""
    d = ctx['data']
    if d.time != 0. or ctx.get('manual_initialization') is not None or not np.array_equal(d.qpos, ctx['base_qpos']):
        raise ValueError('Manual task initialization is only allowed once at fresh setup')
    q = np.asarray(INITIAL_Q9)
    scratch = mujoco.MjData(ctx['model'])
    scratch.qpos[:] = d.qpos
    scratch.qpos[ctx['qa']] = q
    mujoco.mj_forward(ctx['model'], scratch)
    target = np.asarray(ctx['points']['B']['position_world_m'])+[0., 0., .075]
    quat = np.empty(4); mujoco.mju_mat2Quat(quat, scratch.site_xmat[ctx['site']])
    expected = np.asarray(ctx['points']['B']['orientation_wxyz'])
    angle = 2*np.arccos(np.clip(abs(float(quat@expected)), 0., 1.))
    if np.linalg.norm(scratch.site_xpos[ctx['site']]-target) > 1e-6 or angle > 1e-5:
        raise ValueError('Manual task initial FK disagrees with frozen B-relative target')
    d.qpos[ctx['qa']] = q
    mujoco.mj_forward(ctx['model'], d)
    record = dict(q_ref9=q.tolist(), position_world_m=target.tolist(), orientation_wxyz=expected.tolist(),
        source_summary=str(local.SOURCE), source_summary_sha256=local.SOURCE_SHA,
        source_scene_sha256=sha(ctx['folder']/'scene.xml'), source_config_sha256=sha(ctx['folder']/'config.json'),
        kind='new_lesson_initialization_B_plus_75mm_world_Z', physics_steps=0,
        initialization_is_not_executed_movement=True)
    B=np.asarray(ctx['points']['B']['position_world_m']);C=np.asarray(ctx['points']['C']['position_world_m'])
    ctx.update(manual_initialization=record, manual_anchor=target.copy(), manual_orientation=expected.copy(),
        manual_targets=dict(B=B.tolist(),C=C.tolist(),high=target.tolist(),
            release_guide=(B+[.090,.145,.015]).tolist(),above_tray_guide=(B+[.090,.145,.075]).tolist()))
    return record


class TaskPlanner(local.TargetPlanner):
    """Same solve interface, task-scale bounds and contact-aware scratch checks.

    mode='held' requires a measured grip baseline tuple(position,rotation), or
    dict containing position_in_tool_m and rotation_in_tool. The runtime may
    supply it only after its measured grasp gate; it does not weld the object.
    Each prediction follows the current actual cube using the REFERENCE hand's
    increment. mode='released' keeps the actual cube fixed for geometry.
    """
    def __init__(self, ctx, mujoco, np):
        super().__init__(ctx, mujoco, np)
        self.B = np.asarray(ctx['points']['B']['position_world_m']).copy()
        self.C = np.asarray(ctx['points']['C']['position_world_m']).copy()
        self.orientation_wxyz = np.asarray(ctx['points']['B']['orientation_wxyz']).copy()
        for label in ('A', 'C'):
            if abs(abs(float(self.orientation_wxyz@np.asarray(ctx['points'][label]['orientation_wxyz'])))-1.) > 1e-10:
                raise ValueError('Frozen task poses no longer share a common orientation')
        self.origin_xyz = self.B+np.array([0.,0.,.075])
        self.target_xyz = self.origin_xyz.copy()
        self.initial_q9 = np.asarray(INITIAL_Q9).copy()
        self.last_q9 = self.initial_q9.copy()
        self.workspace_low = self.B+np.asarray(SETTINGS['workspace_min_offset_from_B_m'])
        self.workspace_high = self.B+np.asarray(SETTINGS['workspace_max_offset_from_B_m'])
        self.mode, self.payload, self.allow_finger_cube = 'empty', None, False
        self.grasp_confirmed = False
        self._support_escape_ids = set()
        self.environment = [g for g in self.obstacles if g != ctx['cube_geom']]
        self.support_ids = {ctx['table_geom'], int(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'tray_floor'))}
        # The tray is narrow near fully open fingers. Use every compiled finger
        # collision-mesh vertex, as in the accepted carry geometry, instead of
        # including the empty corners of a local enclosing box. Projections of
        # ALL vertices enclose the complete convex collision hull; never sample
        # an inscribed subset or shrink the actual shape to manufacture a gap.
        full_fingers={g:v.copy() for g,v in ctx['finger_meshes']}
        if set(full_fingers)!=set(ctx['finger_geoms']):raise ValueError('Missing full finger collision meshes')
        self.meshes=[(g,full_fingers.get(g,v)) for g,v in self.meshes]
        self._forward(self.initial_q9)
        if np.linalg.norm(self.scratch.site_xpos[ctx['site']]-self.origin_xyz)>1e-6:
            raise ValueError('Manual initial reference mismatch')

    def _forward(self, q9):
        # Parent constructor calls this before task-specific fields exist.
        local.TargetPlanner._forward(self, q9)
        if getattr(self, 'mode', 'empty') == 'held' and self.payload is not None:
            d, np = self.scratch, self.np
            R = d.site_xmat[self.ctx['site']].reshape(3,3).copy()
            p = d.site_xpos[self.ctx['site']].copy()
            a = self.ctx['cube_adr']
            d.qpos[a:a+3] = p+R@self.payload['position']
            quat=np.empty(4); self.mj.mju_mat2Quat(quat,(R@self.payload['rotation']).ravel())
            d.qpos[a+3:a+7]=quat
            # Closed command is a force-producing target, not actual jaw width.
            d.qpos[self.ctx['qa'][7:]]=np.maximum(q9[7:],self.payload['finger_q'])
            self.mj.mj_forward(self.model,d)
            self.forward_calls += 1

    def _configure(self, mode, grip_baseline, kind, grasp_confirmed=False):
        np = self.np
        if mode not in ('empty','closing','held','released'):return 'invalid_task_mode'
        if not isinstance(grasp_confirmed, (bool, np.bool_)):return 'invalid_grasp_confirmation'
        self.mode, self.payload = mode, None
        self.grasp_confirmed = bool(grasp_confirmed)
        self._support_escape_ids = set()
        self.allow_finger_cube = mode == 'held' or kind == 'fingers' or mode == 'released'
        if mode == 'held':
            try:
                if isinstance(grip_baseline,dict):
                    p=np.asarray(grip_baseline['position_in_tool_m'],dtype=float)
                    R=np.asarray(grip_baseline['rotation_in_tool'],dtype=float)
                else:
                    p=np.asarray(grip_baseline[0],dtype=float);R=np.asarray(grip_baseline[1],dtype=float)
                fingers=np.asarray(self.ctx['data'].qpos[self.ctx['qa'][7:]],dtype=float)
            except (KeyError,IndexError,TypeError,ValueError):return 'missing_or_invalid_grip_baseline'
            if p.shape!=(3,) or R.shape!=(3,3) or fingers.shape!=(2,) or not np.isfinite(np.r_[p,R.ravel(),fingers]).all():
                return 'invalid_grip_baseline_shape'
            if not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0) or np.linalg.det(R)<.999999:
                return 'invalid_grip_rotation'
            if np.linalg.norm(p)>.10 or np.any(fingers<self.ctx['finger_limits'][:,0]) or np.any(fingers>self.ctx['finger_limits'][:,1]+1e-9):
                return 'invalid_grip_baseline_range'
            self.payload=dict(position=p.copy(),rotation=R.copy(),finger_q=fingers.copy())
        return None

    def _start(self,q_start9):
        q,error=local.TargetPlanner._start(self,q_start9)
        if error or self.mode!='held':return q,error
        # Reanchor only scratch prediction at actual cube pose. Reference error
        # must not introduce a false instantaneous object jump at command start.
        np=self.np;d=self.scratch;live=self.ctx['data'];cube=self.ctx['cube_geom']
        R=d.site_xmat[self.ctx['site']].reshape(3,3).copy();p=d.site_xpos[self.ctx['site']].copy()
        self.payload['position']=R.T@(live.geom_xpos[cube]-p)
        self.payload['rotation']=R.T@live.geom_xmat[cube].reshape(3,3)
        self._forward(q)
        return q,None

    def _geometry(self):
        np,m,d=self.np,self.model,self.scratch
        cube=int(self.ctx['cube_geom']); fingers=set(self.ctx['finger_geoms'])
        # An unheld cube may rest on a support. A held cube must clear BOTH
        # supports: reporting the gap to walls while excluding the floor hid
        # the pre-release floor collision in the 20261021 user run.
        allowed=([{cube,g} for g in self.support_ids] if self.mode!='held' else
                 [{cube,g} for g in self._support_escape_ids])
        if self.allow_finger_cube:allowed.extend([{cube,int(g)} for g in fingers])
        unexpected=[]
        for c in d.contact[:d.ncon]:
            pair={int(c.geom1),int(c.geom2)}
            if c.dist<=0 and pair not in allowed:
                unexpected.append(dict(geom_ids=[int(c.geom1),int(c.geom2)],distance_m=float(c.dist)))
        best=float('inf');pair=None;finger_gap=float('inf')
        for g,vertices in self.meshes:
            world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            if g in fingers: finger_gap=min(finger_gap,float(world[:,2].min())-self.ctx['cfg']['table']['top_z_m'])
            obs=self.environment if (g in fingers and self.allow_finger_cube) else self.obstacles
            poses=d.geom_xpos[obs]; rotations=d.geom_xmat[obs].reshape(-1,3,3); sizes=m.geom_size[obs]
            local_v=np.einsum('ovi,oij->ovj',world[None,:,:]-poses[:,None,:],rotations)
            gap=np.maximum(-sizes-local_v.max(axis=1),local_v.min(axis=1)-sizes)
            bounds=np.linalg.norm(np.maximum(gap,0.),axis=1)
            j=int(np.argmin(bounds))
            if bounds[j]<best:best=float(bounds[j]);pair=[int(g),int(obs[j])]
        payload_gap=None;payload_pair=None;support_bottom=None;support_gaps={}
        if self.mode=='held':
            from grasp_contact_geometry import box_vertices,separated_box_lower_bound
            vertices=box_vertices(m,d,cube,np)
            payload_gap=float('inf')
            support_bottom=float(vertices[:,2].min())-self.ctx['cfg']['table']['top_z_m']
            for g in self.environment:
                if g in self.support_ids:
                    # Initial lift may start at real compliant support contact.
                    top=float(d.geom_xpos[g,2]+m.geom_size[g,2])
                    value=float(vertices[:,2].min())-top
                    # Only treat a surface as a support underneath projected box.
                    xy=(vertices-d.geom_xpos[g])@d.geom_xmat[g].reshape(3,3)
                    overlaps=bool(np.all(xy[:,:2].min(axis=0)<=m.geom_size[g,:2]) and np.all(xy[:,:2].max(axis=0)>=-m.geom_size[g,:2]))
                    if overlaps: support_gaps[int(g)]=value
                    if overlaps and value < -SETTINGS['cube_support_penetration_tolerance_m']:
                        unexpected.append(dict(geom_ids=[cube,int(g)],distance_m=value,check='payload_below_support'))
                    continue
                value=separated_box_lower_bound(vertices,m,d,g,np)
                if value<payload_gap:payload_gap=value;payload_pair=[cube,int(g)]
        return dict(minimum_environment_gap_lower_bound_m=best,nearest_geom_ids=pair,
            minimum_finger_table_gap_m=finger_gap, unexpected_contacts=unexpected,
            minimum_payload_gap_lower_bound_m=payload_gap,nearest_payload_geom_ids=payload_pair,
            payload_bottom_table_gap_m=support_bottom, payload_support_gaps_m=support_gaps)

    def _segment(self,start,end,target,duration,kind,ik):
        np=self.np;steps=round(duration/DT); rows=[]
        self._forward(start);p0=self.scratch.site_xpos[self.ctx['site']].copy()
        gap_min=table_min=float('inf');payload_min=floor_min=None;max_angle=max_line=max_gravity=0.
        floor_id=min(self.support_ids-{int(self.ctx['table_geom'])})
        initial_geom=self._geometry()
        initial_gaps=initial_geom['payload_support_gaps_m']
        previous_gaps=initial_gaps.copy()
        delta=target-p0
        stationary_open=(kind=='fingers' and np.array_equal(start[:7],end[:7]) and
                         np.array_equal(end[7:],np.asarray([OPEN,OPEN])))
        # Runtime permits release at a measured 2--10 mm floor gap. Opening
        # at an unchanged arm reference is not a lateral/downward move: use
        # that same 2 mm lower gate, while retaining all finger/wall checks.
        required_floor_gap=(SETTINGS['minimum_open_payload_floor_gap_m'] if stationary_open
                            else SETTINGS['minimum_payload_floor_gap_m'])
        upwards=(kind in ('translate','jog') and delta[2]>1e-6 and np.linalg.norm(delta[:2])<SETTINGS['position_tolerance_m'])
        # A too-small existing gap must not trap the user: permit ONLY an
        # upward, monotonically separating segment. This also permits the
        # first lift from compliant table support, without a lateral exception.
        if self.mode=='held' and upwards:
            self._support_escape_ids={g for g,value in initial_gaps.items()
                if -SETTINGS['cube_support_penetration_tolerance_m']<=value<required_floor_gap
                # Once grasp hold has passed, touching the table is no longer
                # the initial supported pickup, even if an upward key is used.
                and not (self.grasp_confirmed and g==int(self.ctx['table_geom']) and value<=0.)}
        for i in range(steps+1):
            u=i/steps;s=u**3*(10+u*(-15+6*u));ds=30*u*u*(1-u)**2/duration
            q=start+s*(end-start);dq=ds*(end-start)
            if i==0:q=start.copy()
            if i==steps:q=end.copy()
            self._forward(q);p=self.scratch.site_xpos[self.ctx['site']].copy()
            angle=float(np.linalg.norm(self._rotation_error()));line=float(np.linalg.norm(p-(p0+s*(target-p0))))
            geom=self._geometry();gap=geom['minimum_environment_gap_lower_bound_m']
            gravity=float(np.max(np.abs(self.scratch.qfrc_bias[self.ctx['va']])/self.ctx['cap']))
            reasons=[]
            if not np.isfinite(np.r_[q,dq,p,angle,line,gap,gravity]).all():reasons.append('nonfinite_reference')
            if np.any(q[:7]<self.lo) or np.any(q[:7]>self.hi):reasons.append('joint_limit_margin')
            if np.any(q[7:]<self.ctx['finger_limits'][:,0]) or np.any(q[7:]>self.ctx['finger_limits'][:,1]+1e-12):reasons.append('finger_limits')
            if np.max(np.abs(dq[:7]))>SETTINGS['maximum_arm_reference_speed_rad_s']:reasons.append('arm_speed')
            if np.max(np.abs(dq[7:]))>SETTINGS['maximum_finger_reference_speed_m_s']:reasons.append('finger_speed')
            if angle>SETTINGS['path_orientation_tolerance_rad']:reasons.append('reference_orientation')
            if line>SETTINGS['path_line_deviation_m']:reasons.append('reference_line_deviation')
            if np.any(p<self.workspace_low-SETTINGS['position_tolerance_m']) or np.any(p>self.workspace_high+SETTINGS['position_tolerance_m']):reasons.append('reference_workspace')
            if gap<SETTINGS['minimum_environment_gap_m']:reasons.append('environment_clearance')
            if geom['minimum_finger_table_gap_m']<SETTINGS['minimum_finger_table_gap_m']:reasons.append('finger_table_clearance')
            if geom['minimum_payload_gap_lower_bound_m'] is not None and geom['minimum_payload_gap_lower_bound_m']<SETTINGS['minimum_payload_gap_m']:reasons.append('payload_clearance')
            for g,value in geom['payload_support_gaps_m'].items():
                if g in self._support_escape_ids:
                    if value<previous_gaps.get(g,value)-SETTINGS['support_escape_monotonic_tolerance_m']:
                        reasons.append('floor_margin_escape_not_monotonic')
                    if i==steps and value<=initial_gaps[g]+.5*delta[2]:
                        reasons.append('floor_margin_escape_not_separating')
                elif value<required_floor_gap:
                    reasons.append('floor_margin')
                previous_gaps[g]=value
            floor_gap=geom['payload_support_gaps_m'].get(floor_id)
            if floor_gap is not None: floor_min=min(floor_min if floor_min is not None else float('inf'),floor_gap)
            if geom['unexpected_contacts']:reasons.append('unexpected_contact')
            if gravity>SETTINGS['maximum_static_gravity_to_cap']:reasons.append('gravity_headroom')
            if reasons:return self._reject('floor_margin' if any(x.startswith('floor_margin') for x in reasons) else 'reference_sample_rejected',dict(kind=kind,rejected_step=i,checks=reasons,geometry=geom,minimum_payload_floor_gap_m=floor_min,required_payload_floor_gap_m=required_floor_gap,stationary_open=stationary_open,orientation_error_rad=angle,line_deviation_m=line,gravity_to_cap=gravity,ik=ik))
            rows.append(dict(step=i,sim_s=i*DT,q_ref9=q.tolist(),dq_ref9=dq.tolist(),position_world_m=p.tolist(),
                orientation_wxyz=self._quaternion().tolist(),minimum_environment_gap_lower_bound_m=gap,
                minimum_finger_table_gap_m=geom['minimum_finger_table_gap_m'],
                minimum_payload_gap_lower_bound_m=geom['minimum_payload_gap_lower_bound_m'],
                payload_floor_gap_m=floor_gap))
            gap_min=min(gap_min,gap);table_min=min(table_min,geom['minimum_finger_table_gap_m'])
            if geom['minimum_payload_gap_lower_bound_m'] is not None:payload_min=min(payload_min if payload_min is not None else float('inf'),geom['minimum_payload_gap_lower_bound_m'])
            max_angle=max(max_angle,angle);max_line=max(max_line,line);max_gravity=max(max_gravity,gravity)
        endpoint=float(np.linalg.norm(np.asarray(rows[-1]['position_world_m'])-target))
        if endpoint>SETTINGS['position_tolerance_m'] or angle>SETTINGS['orientation_tolerance_rad']:
            return self._reject('endpoint_pose_rejected',dict(position_error_m=endpoint,orientation_error_rad=angle))
        self.target_xyz,self.last_q9=target.copy(),end.copy()
        return dict(accepted=True,reason='static_reference_accepted',kind=kind,target_xyz=target.tolist(),
            target_orientation=self.orientation_wxyz.tolist(),duration_s=duration,rows=rows,
            planning_wall_s=time.perf_counter()-self._command_started,
            diagnostics=dict(ik=ik,endpoint_position_error_m=endpoint,endpoint_orientation_error_rad=angle,
                maximum_path_orientation_error_rad=max_angle,maximum_line_deviation_m=max_line,
                minimum_environment_gap_lower_bound_m=gap_min,minimum_finger_table_gap_m=table_min,
                minimum_payload_gap_lower_bound_m=payload_min,maximum_gravity_to_cap=max_gravity,
                minimum_payload_floor_gap_m=floor_min,required_payload_floor_gap_m=required_floor_gap,
                stationary_open=stationary_open,
                upward_support_escape=bool(self._support_escape_ids),grasp_confirmed=self.grasp_confirmed,
                checked_samples=len(rows),discrete_only=True,mode=self.mode,payload_geometry_is_not_physics=True,
                payload_prediction='current_actual_cube_following_reference_hand_increment',
                clearance_representation='full_finger_collision_mesh_vertices_and_other_arm_enclosures'))

    def plan_delta(self,delta_world_m,q_start9,mode='empty',grip_baseline=None,grasp_confirmed=False):
        """Original 5 mm / 1 s discrete reference interface."""
        return self._plan_translation(delta_world_m,q_start9,mode,grip_baseline,grasp_confirmed,
            SETTINGS['maximum_delta_m'],SETTINGS['motion_duration_s'],'translate')

    def plan_jog(self,delta_world_m,q_start9,mode='empty',grip_baseline=None,grasp_confirmed=False):
        """Short user chunk; shrink ONLY on reference-speed rejection.

        At most three candidates (requested size, half, quarter), each still
        0.12 s. Geometry/IK refusals stop immediately; there is no collision
        search or threshold relaxation. Failed candidates cannot commit a goal.
        """
        started=time.perf_counter();attempts=[]
        try:requested=self.np.asarray(delta_world_m,dtype=float)
        except (TypeError,ValueError):requested=None
        candidates=(1.,.5,.25) if requested is not None and requested.shape==(3,) else (1.,)
        for scale in candidates:
            delta=delta_world_m if requested is None else requested*scale
            result=self._plan_translation(delta,q_start9,mode,grip_baseline,grasp_confirmed,
                SETTINGS['jog_maximum_delta_m'],SETTINGS['jog_duration_s'],'jog')
            checks=result.get('diagnostics',{}).get('checks',[])
            attempts.append(dict(scale=scale,delta_world_m=None if requested is None else delta.tolist(),
                accepted=result['accepted'],reason=result['reason'],checks=checks,
                planning_wall_s=result.get('planning_wall_s')))
            if result['accepted'] or not checks or any(c not in ('arm_speed','finger_speed') for c in checks):
                break
        result.update(requested_delta_world_m=None if requested is None else requested.tolist(),
            accepted_delta_world_m=delta.tolist() if result['accepted'] else None,
            jog_speed_attempts=attempts,planning_wall_s=time.perf_counter()-started)
        return result

    def _plan_translation(self,delta_world_m,q_start9,mode,grip_baseline,grasp_confirmed,maximum_delta,duration,kind):
        self._command_started=time.perf_counter();np=self.np
        error=self._configure(mode,grip_baseline,'translate',grasp_confirmed)
        if error:return self._reject(error)
        start,error=self._start(q_start9)
        if error:return self._reject(error)
        try:delta=np.asarray(delta_world_m,dtype=float)
        except (TypeError,ValueError):return self._reject('invalid_delta')
        if delta.shape!=(3,) or not np.isfinite(delta).all():return self._reject('invalid_delta')
        if np.count_nonzero(np.abs(delta)>1e-12)!=1:return self._reject('one_world_axis_per_key')
        if np.linalg.norm(delta)>maximum_delta+1e-12:return self._reject('delta_size')
        target=self.target_xyz+delta
        if np.any(target<self.workspace_low-1e-12) or np.any(target>self.workspace_high+1e-12):return self._reject('workspace_boundary')
        end,info=self._solve(start,target)
        if end is None:return self._reject('ik_failed',info)
        return self._segment(start,end,target,duration,kind,info)

    def plan_fingers(self,finger_target,q_start9,mode='empty',grip_baseline=None,grasp_confirmed=False):
        self._command_started=time.perf_counter()
        error=self._configure(mode,grip_baseline,'fingers',grasp_confirmed)
        if error:return self._reject(error)
        start,error=self._start(q_start9)
        if error:return self._reject(error)
        if not isinstance(finger_target,(float,int,self.np.floating)) or not any(abs(float(finger_target)-v)<1e-12 for v in (OPEN,CLOSED)):
            return self._reject('finger_target_not_open_or_closed')
        if self.np.allclose(start[7:],finger_target,atol=1e-12,rtol=0):return self._reject('fingers_already_at_target')
        end=start.copy();end[7:]=finger_target
        return self._segment(start,end,self.target_xyz.copy(),SETTINGS['finger_duration_s'],'fingers',None)

    def plan_manual_fingers(self,finger_target,q_start9):
        """Direct manual opening/closing, independent of task-score milestones.

        Opening releases the object; it does not keep a rigid payload attached
        in scratch prediction. Check the moving fingers and environment at the
        current object pose. The actual object remains a free physical body.
        Existing limits, speeds and finger/environment geometry still apply.
        """
        self._command_started=time.perf_counter()
        if not isinstance(finger_target,(float,int,self.np.floating)) or not any(
                abs(float(finger_target)-v)<1e-12 for v in (OPEN,CLOSED)):
            return self._reject('finger_target_not_open_or_closed')
        mode='released' if float(finger_target)==OPEN else 'closing'
        error=self._configure(mode,None,'fingers')
        if error:return self._reject(error)
        start,error=self._start(q_start9)
        if error:return self._reject(error)
        if self.np.allclose(start[7:],finger_target,atol=1e-12,rtol=0):
            return self._reject('fingers_already_at_target')
        end=start.copy();end[7:]=finger_target
        result=self._segment(start,end,self.target_xyz.copy(),SETTINGS['finger_duration_s'],'fingers',None)
        result.setdefault('diagnostics',{}).update(manual_gripper_command=True,
            task_milestone_gate=False,payload_prediction='static_current_object_pose_for_geometry_only')
        return result
