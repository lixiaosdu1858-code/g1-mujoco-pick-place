"""User-operated v0.2 recorder around the frozen, validated manual runtime.

Only process-local sampling/session adapters add records. Controls, planning,
physics and task success criteria remain the frozen manual implementation.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

from inspect_episode import digest, read_json, require
from episode_state import FORMAT, FIELD_SEMANTICS, capture_initialization, enrich_state, serialized
from portable_paths import output_path, contained_path

PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def recording_adapters(manual, ctx, planner, out, mujoco, np):
    """Capture before/after requests without changing their return or scheduling."""
    original_session, original_sample = manual.ManualSession, manual.sample
    sequence = 0
    started = time.perf_counter()

    def log(kind, step, result, **fields):
        nonlocal sequence
        sequence += 1
        record = dict(event_id=sequence, kind=kind, request_step=int(step),
            request_sim_s=int(step)*manual.DT, wall_s=time.perf_counter()-started,
            accepted=bool(result['accepted']), reason=result['reason'], **fields)
        with (out/'actions.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(serialized(record), ensure_ascii=True, allow_nan=False)+'\n')

    class CollectionSession(original_session):
        def __init__(self, run_ctx, run_planner, run_np):
            super().__init__(run_ctx, run_planner, run_np)
            save(out/'initialization.json', capture_initialization(manual, run_ctx, self, run_planner, mujoco, np))

        def _request_context(self):
            # Copy before mutation: post-commit goals are action targets, not
            # observations available when the operator made this request.
            return serialized(dict(phase=self.phase, q_hold=self.q_hold,
                goal=self.goal, closed_requested=self.closed_requested,
                pending_gripper_closed=self.pending_gripper_closed,
                planner_mode=planner.mode, generation=self.generation))

        def start(self, step):
            before = self._request_context()
            accepted, reason = super().start(step)
            log('start', step, dict(accepted=accepted, reason=reason), command='START', request_context=before)
            return accepted, reason

        def queue_gripper_toggle(self):
            before = self._request_context()
            previous_pending = self.pending_gripper_closed
            result = super().queue_gripper_toggle()
            log('gripper_intent', round(ctx['data'].time/manual.DT), result,
                command='GRIPPER_INTENT', desired_closed=result.get('desired_closed'),
                request_context=before,
                superseded_pending_closed=previous_pending,
                effective_start_step=None, effective_end_step_exclusive=None)
            return result

        def _plan_fields(self, result, step):
            if not result['accepted']:
                return dict(command_id=None, effective_start_step=None, effective_end_step_exclusive=None)
            return dict(command_id=self.command_id, effective_start_step=int(step),
                effective_end_step_exclusive=int(step)+len(result['rows'])-1,
                target_xyz_m=self.goal.tolist(), phase_after_commit=self.phase)

        def request_jog(self, command, step, row, still_held):
            before = self._request_context()
            result = super().request_jog(command, step, row, still_held)
            log('request_jog', step, result, command=command,
                request_context=before,
                diagnostics=result.get('diagnostics',{}), **self._plan_fields(result,step))
            return result

        def apply_pending_gripper(self, step, row):
            before = self._request_context()
            requested = self.pending_gripper_closed
            result = super().apply_pending_gripper(step,row)
            if result is not None:
                log('gripper_commit',step,result,command='GRIPPER_TARGET',desired_closed=requested,
                    request_context=before,
                    finger_target_m=manual.base.CLOSED if requested else manual.base.OPEN,
                    diagnostics=result.get('diagnostics',{}), **self._plan_fields(result,step))
            return result

    def sample(run_ctx, session, step, wall_s, mj, numpy):
        row, problems, contacts = original_sample(run_ctx, session, step, wall_s, mj, numpy)
        enrich_state(row,run_ctx,step,mj,numpy)
        return row,problems,contacts

    return CollectionSession, sample


def report_template(manual):
    return dict(kind='g1_v02_manual_demonstration', episode_format=FORMAT,
        status='started', end_reason='not_started', model_loaded=False, viewer_started=False,
        physics_started=False, steps=0, data_saved_samples=0, simulation_s=0., events=[], snapshots=[],
        manual_task_completed=False, grasp_hold_passed=False, transfer_completed=False,
        release_started=False, placement_passed=False, tracking_thresholds_passed=False,
        full_v01_accepted=False, learner_verified=False, thresholds=manual.THRESHOLDS,
        placement_thresholds=manual.task.PLACEMENT_THRESHOLDS, noslip_iterations=3,
        input_mode='foreground_hold_to_jog', jog_maximum_segment_m=.0015, jog_segment_s=.12,
        gripper_input_mode='latest_intent_at_segment_boundary_no_task_milestone_lock',
        controller='PD_plus_zero_velocity_gravity', kp=manual.base.motion.KP, kd=manual.base.motion.KD,
        software_caps=manual.base.motion.CAP, executable=sys.executable, temp=tempfile.gettempdir(),
        package_version='0.2.0_candidate_frozen_runtime_with_recording_adapter',
        scope='User-controlled fixed-scene demonstration; no learned policy or automatic task route',
        field_semantics=FIELD_SEMANTICS, diagnostic_mode='csv_and_events_no_async_stack_sampler')


def prepare(args, report, out):
    # Verify packaged file lists before importing their code; all writes stay out of it.
    integrity = read_json(PACKAGE/'configs/runtime_integrity.json')
    for relative, expected in integrity['files'].items():
        path=contained_path(PACKAGE,relative)
        require(path.is_relative_to(PACKAGE.resolve()) and digest(path)==expected, 'Frozen package changed: '+relative)
    for item in read_json(PACKAGE/'assets/manifest.json')['files']:
        path=contained_path(PACKAGE,item['package_path'])
        require(path.is_relative_to(PACKAGE.resolve()) and digest(path)==item['sha256'], 'Mesh changed')
    sys.path.insert(0,str(PACKAGE/'runtime'))
    manual=importlib.import_module('manual_pickplace')
    manual.guard()
    import numpy as np
    import mujoco
    if args.inspect:
        def blocked(*a,**kw):
            raise RuntimeError('Physics stepping is prohibited during collector inspection')
        mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=blocked
    report.update(report_template(manual))
    # Preflight operates on a separate MjData, so its forward/control calls do
    # not alter the integration state of the future user recording.
    probe=manual.setup(mujoco,np)
    _,inspection=manual.inspect(probe,mujoco,np)
    ctx=manual.setup(mujoco,np)
    planner=manual.TaskPlanner(ctx,mujoco,np)
    m,d=ctx['model'],ctx['data']
    require(m.na==m.nmocap==m.neq==m.nuserdata==m.nplugin==0, 'Unsupported active model state')
    report.update(model_loaded=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,
        python_version=sys.version.split()[0],dt_s=manual.DT,static_inspection=inspection,
        initial_qpos=d.qpos.tolist(),initial_qvel=d.qvel.tolist(),
        initial_ctrl=d.ctrl.tolist(),initial_qacc_warmstart=d.qacc_warmstart.tolist(),
        frozen_physics_options=manual.frozen.option_snapshot(m,np),
        anchor_position_world_m=ctx['manual_anchor'].tolist(),target_fixed_orientation_wxyz=ctx['manual_orientation'].tolist(),
        planner_settings=manual.PLANNER_SETTINGS,manual_initialization=ctx['manual_initialization'])
    package=importlib.import_module('package_runtime')
    package.snapshot(ctx,out,report)
    save(out/'manual_initialization.json',serialized(ctx['manual_initialization']))
    report['snapshots'].append(dict(path='manual_initialization.json',sha256=digest(out/'manual_initialization.json')))
    (out/'command_references').mkdir()
    (out/'collection_code').mkdir()
    report['collection_code']=[]
    for name in ('collect_episode.py','episode_state.py','episode_schema.py',
                 'inspect_episode.py','portable_paths.py'):
        source=Path(__file__).with_name(name)
        dest=out/'collection_code'/name
        shutil.copy2(source,dest)
        report['collection_code'].append(dict(path='collection_code/'+name,sha256=digest(dest)))
    for name in ('actions.jsonl','events.jsonl','jog_references.jsonl'):
        (out/name).touch(exist_ok=False)
    save(out/'field-semantics.json',FIELD_SEMANTICS)
    return manual,ctx,planner,mujoco,np,inspection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    out=output_path(args.output,exists=False)
    require(out.parent==(PROJECT/'outputs').resolve() and out.name.startswith('demonstration_'),
        'Use a fresh project outputs/demonstration_* directory')
    out.mkdir(exist_ok=False); os.chdir(out)
    report=dict(kind='g1_v02_manual_demonstration',status='preparing',steps=0,manual_task_completed=False)
    started=time.perf_counter(); code=1
    try:
        manual,ctx,planner,mj,np,inspection=prepare(args,report,out)
        cls,sample=recording_adapters(manual,ctx,planner,out,mj,np)
        if args.inspect:
            session=cls(ctx,planner,np)
            before=(ctx['data'].qpos.copy(),ctx['data'].qvel.copy(),float(ctx['data'].time))
            row,problems,_=sample(ctx,session,0,0.,mj,np)
            require(not problems,'Initial conditions failed: '+str(problems))
            differences={k:abs(float(row[k])-float(v)) for k,v in inspection['initial_sample'].items()
                if isinstance(v,(int,float)) and not isinstance(v,bool)}
            require(max(differences.values(),default=0.)<=1e-9,'Recording adapter changed initial measurements')
            require(np.array_equal(before[0],ctx['data'].qpos) and np.array_equal(before[1],ctx['data'].qvel)
                and before[2]==ctx['data'].time==0,'Inspection changed integrated state')
            save(out/'initial-sample.json',serialized(row))
            report.update(status='prepared_static_only',end_reason='inspect_only',new_physics_steps=0,
                new_windows=0,initial_measurement_max_difference=max(differences.values(),default=0.),
                recorded_state_columns=len(row),complete_v02_episode=False)
            print('COLLECTION PREPARATION: PASS | 0 physics steps | 0 windows',flush=True)
            code=0
        else:
            # These adapters live only in this Python process. Frozen files and
            # existing controllers/runs are never modified.
            manual.ManualSession=cls; manual.sample=sample
            print('DEMONSTRATION RUN:',out,flush=True)
            code=manual.run(ctx,planner,report,out,mj,np)
    except KeyboardInterrupt:
        report.update(status='interrupted',end_reason='KeyboardInterrupt',manual_task_completed=False);code=130
    except Exception:
        report.update(status='error',end_reason='exception',manual_task_completed=False,error=traceback.format_exc())
        save(out/'error.json',dict(error=report['error']))
        print(report['error'],file=sys.stderr);code=1
    finally:
        report['wall_s']=time.perf_counter()-started
        for filename,key in (('manual.csv','manual_csv_sha256'),('jog_references.jsonl','jog_references_sha256'),
                             ('initialization.json','initialization_sha256'),('actions.jsonl','actions_sha256')):
            if (out/filename).exists():report[key]=digest(out/filename)
        save(out/'summary.json',serialized(report))
        print('SUMMARY:',out/'summary.json',flush=True)
    if not args.inspect and (out/'manual.csv').exists():
        try:
            from episode_schema import finalize_episode
            index=finalize_episode(out)
            print('EPISODE FORMAT:', 'PASS' if index.get('complete_v02_episode') else 'INCOMPLETE',flush=True)
            print('EPISODE CHECK:',out/'episode',flush=True)
            if not index.get('complete_v02_episode') and code==0: code=3
        except Exception:
            (out/'episode-finalization-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print(traceback.format_exc(),file=sys.stderr)
            if code==0:code=3
    print('DEMONSTRATION RUN:',out,flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
