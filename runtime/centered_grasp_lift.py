"""Geometry-only grasp comparison. User operates S/C/L; the original controller is reused."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

from lesson_common import ROOT, guard, save_json, sha, stamp
import grasp_lift as base
import reachable_motion as motion
from reachable_pose_preview import model_context, apply_and_check

INDEX = Path(__file__).with_name('centered_grasp_source.json')


def setup(mujoco, np):
    # Retain all previously validated motor, reference and original-source checks.
    old = base.setup(mujoco, np)
    index = json.loads(INDEX.read_text(encoding='utf-8-sig'))
    folder = (ROOT/index['source_directory']).resolve()
    if folder.parent != (ROOT/'outputs').resolve():
        raise ValueError('Centered source must be a project output directory')
    for name in ('scene.xml', 'config.json', 'receipt.json'):
        if sha(folder/name) != index['sha256'][name]:
            raise ValueError('Centered geometry source changed: '+name)
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8-sig'))
    receipt = json.loads((folder/'receipt.json').read_text(encoding='utf-8-sig'))
    if not receipt.get('static_geometry_prepared') or receipt.get('physical_grasp_validated'):
        raise ValueError('Expected prepared geometry, without a claimed physical grasp')
    if cfg['scene_sha256'] != sha(folder/'scene.xml'):
        raise ValueError('Centered configuration does not bind its scene')
    for name, path in [('original_scene',old['folder']/'scene.xml'),
                       ('original_config',old['folder']/'config.json'),
                       ('reference',old['reference_file']),
                       ('controller',Path(base.__file__)),('motion',Path(motion.__file__))]:
        if index['unchanged_sha256'][name] != sha(path):
            raise ValueError('Comparison dependency changed: '+name)
    fresh = model_context(folder,cfg,mujoco,np)
    m,previous = fresh['model'],old['model']
    # Layout positions and table-leg lengths may differ; mechanics must not.
    for key in ('nq','nv','nu','nbody','ngeom','njnt','nmesh','neq','nmocap'):
        if getattr(m,key)!=getattr(previous,key): raise ValueError('Topology changed: '+key)
    for key in ('jnt_type','jnt_qposadr','jnt_dofadr','jnt_range','jnt_axis','jnt_pos',
                'jnt_limited','jnt_actfrclimited','jnt_actfrcrange',
                'dof_damping','dof_frictionloss','dof_armature','body_mass','body_inertia',
                'geom_type','geom_dataid','geom_quat','geom_friction','geom_condim',
                'geom_solref','geom_solimp','geom_contype','geom_conaffinity',
                'actuator_trnid','actuator_trntype','actuator_gear','actuator_gainprm',
                'actuator_biasprm','actuator_gaintype','actuator_biastype','actuator_dyntype',
                'actuator_ctrllimited','actuator_ctrlrange','actuator_forcelimited','actuator_forcerange',
                'mesh_vert','mesh_face','site_pos','site_quat'):
        if not np.array_equal(getattr(m,key),getattr(previous,key)):
            raise ValueError('Geometry comparison changed mechanics: '+key)
    if (m.opt.timestep!=previous.opt.timestep or m.opt.integrator!=previous.opt.integrator
            or not np.array_equal(m.opt.gravity,previous.opt.gravity) or m.neq or m.nmocap):
        raise ValueError('Dynamics options or free-object model changed')
    points={p['id']:p for p in cfg['poses']}
    for key in ('A','B','C'):
        for field in ('q_arm_rad','position_world_m','orientation_wxyz'):
            if not np.array_equal(points[key][field],old['points'][key][field]):
                raise ValueError('Arm pose changed: '+key+' '+field)
    ctx={**old,**fresh}
    apply_and_check(ctx,points['A'],mujoco,np)
    ctx.update(folder=folder,cfg=cfg,points=points,gravity=mujoco.MjData(m),reference=mujoco.MjData(m),
               cube_initial=ctx['data'].qpos[ctx['cube_adr']:ctx['cube_adr']+3].copy(),
               contact_geometry_receipt=folder/'receipt.json',geometry_receipt=receipt,
               original_folder=old['folder'],original_contact_receipt=old['contact_geometry_receipt'],
               comparison_index=index,grip_baseline=None)
    return ctx


def snapshot(ctx,out,report):
    files=[(ctx['folder']/n,'source/'+n) for n in ('scene.xml','config.json','receipt.json')]
    files += [(ctx['reference_file'],'source/path.json'),
              (ctx['original_contact_receipt'],'original_source/contact_geometry_receipt.json'),
              (ROOT/'repos/g1_grasp_assets/asset_manifest.json','source/asset_manifest.json')]
    files += [(ctx['original_folder']/n,'original_source/'+n) for n in ('scene.xml','config.json','static_audit.json')]
    files += [(p,'code/'+p.name) for p in Path(__file__).parent.glob('*.py')]
    files += [(Path(__file__).with_name(n),'code/'+n) for n in
              ('centered_grasp_source.json','reachable_preview_source.json','reachable_motion_source.json',
               'grasp_lift_source.json','Run-CenteredGraspLift.ps1','Show-GraspLift.ps1')]
    for source,relative in files:
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        report['snapshots'].append(dict(path=relative,sha256=sha(dest)))


def main():
    guard()
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve() if args.output else ROOT/'outputs'/('centered_grasp_user_'+stamp())
    if out.parent!=(ROOT/'outputs').resolve() or not out.name.startswith('centered_grasp_user_'):
        raise ValueError('Use a fresh centered_grasp_user_ project output')
    if not args.inspect:out.mkdir(exist_ok=False)
    os.chdir(ROOT/'outputs' if args.inspect else out)
    report=dict(kind='grasp_lift_dynamics',variant='contact_centered_layout',status='started',
        model_loaded=False,viewer_started=False,physics_started=False,steps=0,data_saved_samples=0,simulation_s=0.,
        events=[],phases=[],grasp_hold_passed=False,tracking_thresholds_passed=False,learner_verified=False,
        transport_validated=False,release_validated=False,thresholds=base.THRESHOLDS,
        object_stop_thresholds=dict(translation_m=.015,rotation_rad=.25),
        controller='PD_plus_zero_velocity_gravity',finger_open_m=base.OPEN,finger_closed_target_m=base.CLOSED,
        kp=motion.KP,kd=motion.KD,software_caps=motion.CAP,executable=sys.executable,temp=tempfile.gettempdir(),snapshots=[],
        sample_semantics='Pre-step actual/reference/current motor commands. Terminal future control fields blank.',
        contact_semantics='Active positive cube/finger normal force sums; separate from motor forces. Cube/tool baseline captured once after settled bilateral contact.',
        scope='Geometry comparison; unchanged arm references, controller, friction and gates. No transfer/release/attachment. No prior success claimed.')
    started=time.perf_counter();code=0
    try:
        import numpy as np
        import mujoco
        ctx=setup(mujoco,np);inspection=base.inspect(ctx,mujoco,np)
        report.update(model_loaded=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,dt_s=base.DT,
            static_inspection=inspection,source_reference=str(ctx['reference_file']),
            layout_translation_world_m=ctx['comparison_index']['layout_translation_world_m'],
            original_comparison_run=str(ROOT/'outputs/grasp_lift_user_20261018_153545_741'))
        if args.inspect:
            print(json.dumps(dict(inspection=inspection,variant=report['variant'],layout_translation_world_m=report['layout_translation_world_m']),indent=2));return 0
        snapshot(ctx,out,report)
        print('CENTERED GRASP RUN:',out,flush=True)
        print('Geometry-only comparison. Physical grasp outcome is not yet known.',flush=True)
        code=base.run(ctx,report,out,mujoco,np)
        if 'stop_diagnostics' in report:
            row=report['stop_diagnostics']['row']
            report['object_stop_components']=dict(
                translation_exceeded=row['relative_slip_m']!='' and row['relative_slip_m']>.015,
                rotation_exceeded=row['relative_rotation_rad']!='' and row['relative_rotation_rad']>.25)
    except KeyboardInterrupt:
        report.update(status='interrupted',end_reason='KeyboardInterrupt',grasp_hold_passed=False);code=130
    except Exception:
        report.update(status='error',end_reason='exception',grasp_hold_passed=False,error=traceback.format_exc());code=1
        print(report['error'],file=sys.stderr)
        if not args.inspect:(out/'error.txt').write_text(report['error'],encoding='utf-8')
    finally:
        report['wall_s']=time.perf_counter()-started
        if not args.inspect:
            if (out/'grasp.csv').exists():report['grasp_csv_sha256']=sha(out/'grasp.csv')
            save_json(out/'summary.json',report);print('SUMMARY:',out/'summary.json',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
