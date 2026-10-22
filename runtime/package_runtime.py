"""Portable setup/snapshot adapter; frozen control/planning functions unchanged.

Only setup assigns a fresh initial integrated state. No trajectory is replayed.
"""
import json
import shutil
from lesson_common import ROOT, sha

def read(name):
    return json.loads((ROOT/'configs'/name).read_text(encoding='utf-8-sig'))

def integrity():
    manifest=read('runtime_integrity.json')
    for relative, expected in manifest['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=expected:
            raise ValueError('Package runtime integrity mismatch: '+relative)
    return manifest

def setup(mujoco,np):
    integrity()
    from reachable_motion import KP,KD,CAP
    from grasp_solver_runtime import option_snapshot
    from manual_task_target import INITIAL_Q9
    folder=ROOT/'configs'
    cfg=read('config.json'); frozen=read('frozen_context.json')
    if mujoco.__version__!=frozen['mujoco_version'] or np.__version__!=frozen['numpy_version']:
        raise ValueError('Use the versions pinned in requirements.txt.')
    if KP!=frozen['kp'] or KD!=frozen['kd'] or CAP!=frozen['cap']:
        raise ValueError('Frozen controller constants changed')
    if sha(folder/'scene.xml')!=cfg['scene_sha256']:
        raise ValueError('Configuration/scene mismatch')
    mapping=json.loads((ROOT/'assets/vfs_map.json').read_text(encoding='utf-8'))
    assets={key:(ROOT/value).read_bytes() for key,value in mapping.items()}
    model=mujoco.MjModel.from_xml_string((folder/'scene.xml').read_text(encoding='utf-8'),assets=assets)
    model.opt.noslip_iterations=3
    if option_snapshot(model,np)!=frozen['effective_solver_options']:
        raise ValueError('Frozen solver options changed')
    if (model.nq,model.nv,model.nu)!=(16,15,9) or model.neq or model.nmocap:
        raise ValueError('Expected fixed-base free-cube model without attachments')
    jid=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    gid=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
    arm_ids=[jid(n) for n in cfg['arm_joint_order']]
    finger_ids=[jid(n) for n in cfg['finger_joint_order']]
    joint_ids=arm_ids+finger_ids
    if min(joint_ids)<0:raise ValueError('Missing controlled joint')
    aids=[]
    for i,j in enumerate(joint_ids):
        found=np.where((model.actuator_trnid[:,0]==j)&(model.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT)))[0]
        if len(found)!=1:raise ValueError('Expected one direct motor per controlled joint')
        a=int(found[0]);aids.append(a)
        if (int(model.actuator_gaintype[a])!=0 or int(model.actuator_biastype[a])!=0 or int(model.actuator_dyntype[a])!=0
            or not np.array_equal(model.actuator_gear[a],[1,0,0,0,0,0]) or model.actuator_gainprm[a,0]!=1
            or not model.actuator_ctrllimited[a] or model.actuator_ctrlrange[a,0]>-CAP[i] or model.actuator_ctrlrange[a,1]<CAP[i]):
            raise ValueError('Unexpected motor semantics')
    data=mujoco.MjData(model)
    # Explicit fresh initialization, never applied again after setup.
    data.qpos[:]=frozen['initial_qpos'];data.qvel[:]=frozen['initial_qvel']
    data.ctrl[:]=frozen['initial_ctrl'];data.qacc_warmstart[:]=frozen['initial_qacc_warmstart']
    mujoco.mj_forward(model,data)
    data.qacc_warmstart[:]=frozen['initial_qacc_warmstart']
    qa=model.jnt_qposadr[joint_ids].copy();va=model.jnt_dofadr[joint_ids].copy()
    if not np.array_equal(data.qpos[qa],INITIAL_Q9):raise ValueError('Manual initial reference mismatch')
    meshes=[]
    for name in ('right_dex1_finger_1_collision','right_dex1_finger_2_collision'):
        g=gid(name);mid=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid])
        meshes.append((g,model.mesh_vert[start:start+count].astype(float).copy()))
    cube_adr=int(model.jnt_qposadr[jid('task_cube_free')])
    return dict(model=model,data=data,site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,cfg['site']),
        arm_ids=arm_ids,arm_adr=model.jnt_qposadr[arm_ids].copy(),finger_adr=model.jnt_qposadr[finger_ids].copy(),
        cube_adr=cube_adr,base_qpos=np.asarray(cfg['preview_qpos'],dtype=float),folder=folder,cfg=cfg,
        points={p['id']:p for p in cfg['poses']},reference_file=folder/'reference.json',
        qa=qa,va=va,aids=np.asarray(aids),limits=model.jnt_range[arm_ids].copy(),
        gravity=mujoco.MjData(model),reference=mujoco.MjData(model),kp=np.asarray(KP),kd=np.asarray(KD),cap=np.asarray(CAP),
        finger_meshes=meshes,finger_geoms=[g for g,_ in meshes],finger_limits=model.jnt_range[finger_ids].copy(),
        cube_geom=gid('task_cube_geom'),table_geom=gid('task_table_top'),cube_initial=data.qpos[cube_adr:cube_adr+3].copy(),
        grip_baseline=None,solver_effective_options=frozen['effective_solver_options'],noslip_iterations=3,
        carry_route={'tray':frozen['carry_tray']},manual_initialization=frozen['manual_initialization'],
        manual_anchor=np.asarray(frozen['manual_anchor']),manual_orientation=np.asarray(frozen['manual_orientation']),
        manual_targets=frozen['manual_targets'])

def snapshot(ctx,out,report):
    for folder in ('runtime','configs'):
        for source in sorted((ROOT/folder).glob('*')):
            if not source.is_file():continue
            relative=folder+'/'+source.name
            dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            report['snapshots'].append(dict(path=relative,sha256=sha(dest)))
    for relative in ('assets/manifest.json','assets/vfs_map.json'):
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
        report['snapshots'].append(dict(path=relative,sha256=sha(dest)))
    report['asset_location']='Package assets; each file bound by assets/manifest.json'
    report['setup_adapter']='portable_frozen_initialization_v1'
