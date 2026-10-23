"""Read-only state capture shared by collection and reference replay."""

FORMAT = 'g1.demonstration.v0.2.0'
SESSION_FIELDS = ('phase', 'q_hold', 'goal', 'plan', 'start_step', 'settle_step',
    'ready_since', 'command_id', 'completed_ids', 'pending_id', 'command_records',
    'generation', 'closed_requested', 'pending_gripper_closed', 'notice',
    'jog_reference_completed_ids', 'last_plan_kind')
PLANNER_FIELDS = ('origin_xyz', 'target_xyz', 'initial_q9', 'last_q9',
    'orientation_wxyz', 'mode', 'payload', 'allow_finger_cube', 'grasp_confirmed')


def serialized(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serialized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialized(v) for v in value]
    if hasattr(value, '__dict__'):
        return serialized(vars(value))
    return value


def capture_initialization(manual, ctx, session, planner, mujoco, np):
    m, d = ctx['model'], ctx['data']
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, state, spec)
    return dict(format=FORMAT, scope='same_fixed_package_initial_state',
        capture_point='after fresh setup and software construction, before first controller sample',
        state_spec=spec, integration_state_size=len(state), integration_state=state.tolist(),
        time_s=float(d.time), model_dimensions={name:int(getattr(m,name)) for name in
            ('nq','nv','nu','na','nmocap','neq','nuserdata','nplugin')},
        qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist(),
        qacc_warmstart=d.qacc_warmstart.tolist(),
        empty_fields={'act':'no actuator dynamics','mocap':'no mocap bodies',
                      'eq_active':'no equality constraints','userdata':'zero dimension','plugin_state':'no plugins'},
        software=dict(
            session={key:serialized(getattr(session,key)) for key in SESSION_FIELDS},
            planner={key:serialized(getattr(planner,key)) for key in PLANNER_FIELDS},
            task_observer=serialized(session.checks),
            input_policy=dict(mode='foreground_hold_to_jog', armed=False, need_release=True,
                pending_keys=[], saturation_s=0.0, grip_baseline=serialized(ctx['grip_baseline']))))


def enrich_state(row, ctx, step, mujoco, np):
    """Append current measurements without forwarding or changing any state."""
    m, d = ctx['model'], ctx['data']
    row['step_id'] = int(step)
    row.update({f'qpos_{j}':float(v) for j,v in enumerate(d.qpos)})
    row.update({f'qvel_{j}':float(v) for j,v in enumerate(d.qvel)})
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, d.site_xmat[ctx['site']])
    row.update({'actual_quat_'+a:float(v) for a,v in zip('wxyz',quat)})
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'task_cube_free')
    qadr, vadr = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
    row.update({'cube_quat_'+a:float(v) for a,v in zip('wxyz',d.qpos[qadr+3:qadr+7])})
    velocity = np.empty(6)
    # MuJoCo spatial velocity is angular then linear, at the site, global axes.
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, ctx['site'], velocity, 0)
    for prefix, values in (('actual_angular_velocity_world',velocity[:3]),
                           ('actual_linear_velocity_world',velocity[3:]),
                           ('cube_linear_velocity_world',d.qvel[vadr:vadr+3]),
                           ('cube_angular_velocity_local',d.qvel[vadr+3:vadr+6])):
        row.update({prefix+'_'+a:float(v) for a,v in zip('xyz',values)})
    return row


FIELD_SEMANTICS = dict(
    state_action_timing='row k measures s_k and controller reference for [k*dt,(k+1)*dt); terminal N has no action',
    joint_order='configs/config.json joint_order; first seven rad/rad_s, fingers m/m_s',
    tool_orientation='actual_quat_wxyz: unit quaternion, site frame in world',
    cube_orientation='cube_quat_wxyz: free-body unit quaternion in world',
    tool_velocity='actual_*_velocity_world_xyz: site origin linear m/s, angular rad/s, world axes',
    cube_velocity='cube_linear_velocity_world_xyz m/s; cube_angular_velocity_local_xyz rad/s in cube local axes',
    full_state='qpos includes free-body quaternion; qvel is generalized velocity and is not componentwise qpos derivative',
    input_events='events.jsonl: polled input transitions and discrete receive/process; received wall clock distinct from processing sim_s',
    actions='actions.jsonl: pre-request software context, requests/rejections and accepted segment intervals; repeated toggle intents resolve to explicit states; post-commit targets are not pre-action observations',
    planned_vs_executed='accepted end is planned exclusive endpoint; only 0..N-1 pre-step rows were executed',
    gripper_position='finger joint coordinate, not total jaw width')
