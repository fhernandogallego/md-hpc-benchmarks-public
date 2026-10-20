#!/usr/bin/env python3
"""One-command ATLAS preprocessing and train-only refitting on SCAYLE."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import time
import traceback

from common import (FAILED, GROUP, SYSTEM, atomic_json, execute, fresh_directory, inspect_archive,
                    job_ids, jobs_outcome, load_json, read_csv, require, sha,
                    slurm_states, stamp, status_dict, unpack_archive, write_csv)


def default_work():
    return Path(os.environ.get('SCRATCH', str(Path.home()/'scratch')))/'md-hpc-benchmarks'


def initial_base(work):
    base=work/'results/paper2_hito0'
    result={
        'sources':base/'atlas222_assembly_20261015_v1/sources.csv',
        'assignment':base/'atlas222_grouped_development_v1/system_assignment.csv',
        'inventory':base/'atlas222_source_inventory_20261015_125033/source_inventory.csv',
        'sequences':base/'atlas222_sequences_20261016_121738/sequences.fasta',
        'future_rows':base/'atlas222_future_rmsd_train_3239158/train_future_rmsd_rows.csv',
        'source_hashes':base/'atlas222_future_rmsd_train_3239158/source_hashes.json',
    }
    expected={'assignment':'79b983deb8c1f857fb7a22ce05dcdd107dbb1ff4fd37ee2971c046e7564e2149',
              'inventory':'8877f3fcb12e53af21ae609d1b1a1176965f2584fd75ad03a2db21b6ab5ca3ce',
              'sequences':'a6da7bd4d2c2a56fb9d7f2256f09678690e72455602eee7f393cd7f14cf1c4e6',
              'future_rows':'3caa09b9171ea5d12dccc3d9b9f44b0e2eff622083c849e31f623c876a53c1d9'}
    for key,path in result.items():
        require(path.is_file(),f'Missing ATLAS222 foundation: {path}')
        if key in expected:
            require(sha(path)==expected[key],f'Changed ATLAS222 foundation: {path}')
    require(len(read_csv(result['sources']))==222,'Unexpected foundation size')
    return {**{k:str(v) for k,v in result.items()},'acquired_exclusions':['3txs_C']}


def resolve_base(config):
    work=Path(config['work'])
    latest=work/'results/paper2_automation/LATEST_SUCCESS.json'
    if config.get('base'):
        path=Path(config['base']).expanduser().resolve()
        if path.is_dir():
            path=path/'cohort_state.json'
        state=load_json(path)
    elif latest.is_file():
        state=load_json(Path(load_json(latest)['cohort_state']))
    else:
        state=initial_base(work)
    for key in ('sources','assignment','inventory','sequences','future_rows','source_hashes'):
        require(Path(state[key]).is_file(),f'Missing base artifact: {key}: {state[key]}')
    for key,expected in state.get('sha256',{}).items():
        require(sha(state[key])==expected,f'Changed base artifact: {key}')
    return state


def discover(work, base, group=None, manifest=None):
    raw=Path(work)/'raw/atlas'
    known={r['system'] for r in read_csv(base['sources'])}|set(base.get('acquired_exclusions',[]))
    requested=None
    if manifest:
        rows=read_csv(manifest)
        require(rows and {'system','dataset_group'}<=set(rows[0]),'Input CSV needs system,dataset_group')
        requested={(r['dataset_group'],r['system']):r for r in rows}
        require(len(requested)==len(rows),'Duplicate input CSV rows')
    roots=[raw/group] if group else sorted(p for p in raw.iterdir() if p.is_dir())
    result={}
    for root in roots:
        require(GROUP.fullmatch(root.name),f'Unsupported dataset_group: {root.name}')
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir()):
            system=directory.name
            if not directory.is_dir() or not SYSTEM.fullmatch(system) or system in known:
                continue
            if requested is not None and (root.name,system) not in requested:
                continue
            require(system[:4]==system[:4].lower(),f'Use canonical lowercase PDB id: {system}')
            archives=directory/'archives'
            if not all((archives/f'{system}_{kind}.zip').is_file() for kind in ('analysis','protein')):
                continue
            require(system not in result,f'System appears in two raw groups; select --group or --manifest: {system}')
            r=requested.get((root.name,system),{}) if requested is not None else {}
            result[system]=dict(system=system,dataset_group=root.name,raw=str(directory.resolve()),
                expected_analysis_sha256=r.get('analysis_sha256',''),expected_protein_sha256=r.get('protein_sha256',''))
    if requested is not None:
        missing=[f'{g}/{s}' for g,s in requested if s not in known and s not in result]
        require(not missing,'Missing/incomplete requested archives: '+','.join(missing))
    return [result[s] for s in sorted(result)]


def snapshot_pipeline(project, destination, work, selection):
    project,destination=Path(project),Path(destination)
    for directory in ('src','slurm'):
        shutil.copytree(project/directory,destination/directory,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (destination/'manifests').mkdir()
    write_csv(destination/'manifests/atlas_systems.csv',[
        dict(system=r['system'],enabled='true',expected_replicas=3,
             dataset_group=r['dataset_group'],status='staged') for r in selection])
    for name in ('submit_atlas_system_pipeline.sh','atlas_pipeline_stage.sbatch'):
        path=destination/'slurm'/name
        text=path.read_text()
        for key,value in [('PROJECT',destination),('WORK',work)]:
            text,count=re.subn(r'^'+key+r'=.*$',key+'='+shlex.quote(str(value)),text,flags=re.M)
            require(count==1,f'Cannot safely pin {key} in {path}')
        path.write_text(text)
        execute(['bash','-n',path])


def verify_package(runroot, item, log):
    import numpy as np
    runroot=Path(runroot)
    status=status_dict(runroot/'PIPELINE_STATUS.txt')
    expected={'pipeline_status':'PASS','system':item['system'],'dataset_group':item['dataset_group'],
              'n_replicas':'3','prefix_end_ns':'20','expected_steps':'201','expected_step_ps':'100',
              'normalized':'false','imputed':'false','official_basic_comparison':'true',
              'official_rmsf_comparison':'true','package_audit':'true','cross_file_audit':'true'}
    require(all(status.get(k)==v for k,v in expected.items()),f'Invalid final status: {runroot}')
    execute(['sha256sum','--check','--quiet','checksums.sha256'],log,cwd=runroot)
    release=runroot/'processed'/item['dataset_group']/item['system']/'prefix_sequences'
    with np.load(release/'prefix_sequences_raw.npz',allow_pickle=False) as z:
        require(z['X_raw'].shape==(3,201,8) and np.isfinite(z['X_raw']).all(),'Invalid feature tensor')
    return str(release)


def pipeline_command(config,item,dry=False):
    command=['bash',str(Path(config['pipeline_code'])/'slurm/submit_atlas_system_pipeline.sh'),
             '--system',item['system'],'--dataset-group',item['dataset_group'],'--run-id',config['run_id']]
    return command+(['--dry-run'] if dry else [])


def wait_jobs(ids, log_label, poll):
    unknown_since=None
    while True:
        states,active=slurm_states(ids)
        outcome=jobs_outcome(ids,states,active)
        if outcome=='PASS':
            return
        require(outcome!='FAILED',f'{log_label} failed: {states}')
        if outcome=='UNKNOWN':
            unknown_since=unknown_since or time.time()
            require(time.time()-unknown_since<300,f'{log_label}: accounting unknown for 5 minutes. Inspect Slurm; no duplicate submitted.')
        else:
            unknown_since=None
        print(f'WAIT {log_label} {outcome}',flush=True)
        time.sleep(poll)


def completed_step(path):
    marker=Path(path)/'STEP_COMPLETE.json'
    if not marker.is_file():
        return False
    for name,expected in load_json(marker)['files'].items():
        require(sha(Path(path)/name)==expected,f'Changed completed stage: {path}/{name}')
    return True


def finish_step(path):
    path=Path(path)
    atomic_json(path/'STEP_COMPLETE.json',{'files':{
        str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()
        and p.name!='STEP_COMPLETE.json'}})


def train(config, root):
    from cohort import audit_training, training_bundle
    root=Path(root)
    directory=root/'training'
    if not (directory/'manifest.json').is_file():
        fresh_directory(directory)
        training_bundle(config,root/'cohort',directory)
    else:
        for name,value in load_json(directory/'manifest.json')['files'].items():
            require(sha(directory/name)==value,f'Changed training input: {name}')
    worker=directory/'worker.sbatch'
    if not worker.is_file():
        worker.write_text('#!/bin/bash\nset -euo pipefail\n'+
            f'cd {shlex.quote(str(directory/"code"))}\n'+
            f'export PYTHONPATH={shlex.quote(str(directory/"code"))}\n'+
            f'{shlex.quote(sys.executable)} -u worker.py {shlex.quote(str(directory))} "${{SLURM_ARRAY_TASK_ID:-aggregate}}"\n')
    jobfile=directory/'array_job.json'
    if jobfile.is_file():
        prior=load_json(jobfile)
        if prior.get('intent') and not prior.get('id'):
            raise RuntimeError('Uncertain training submission. Inspect squeue before resolving array_job.json; no duplicate submitted.')
        if prior.get('id'):
            states,active=slurm_states([prior['id']])
            outcome=jobs_outcome([prior['id']],states,active)
            if outcome in ('ACTIVE','UNKNOWN'):
                wait_jobs([prior['id']],'training array',config['poll'])
    missing=[f for f in range(5) if not (directory/f'fold_{f}/PASS').is_file()]
    if missing:
        for f in missing:
            if (directory/f'fold_{f}').exists():
                fresh_directory(directory/f'fold_{f}')
                (directory/f'fold_{f}').rmdir()  # Worker itself creates its fresh directory.
        atomic_json(jobfile,{'intent':True,'folds':missing})
        job=execute(['sbatch','--parsable','--account='+config['account'],
            '--partition='+config['partition'],'--qos='+config['qos'],
            '--job-name=paper2_cv','--cpus-per-task=1','--mem=6G','--time=02:00:00',
            '--array='+','.join(map(str,missing))+'%5',
            '--output='+str(directory/'cv_%A_%a.out'),worker]).strip().split(';')[0]
        require(job.isdigit(),'Cannot parse training job id')
        atomic_json(jobfile,{'intent':False,'id':job,'folds':missing})
        print('CV_ARRAY_JOB='+job,flush=True)
        wait_jobs([job],'training array',config['poll'])
    execute([sys.executable,directory/'code/worker.py',directory,'aggregate'],directory/'aggregate.log',cwd=directory/'code')
    report=audit_training(directory)
    if not (directory/'full_train/metadata.json').is_file():
        if (directory/'full_train').exists():
            fresh_directory(directory/'full_train')
            (directory/'full_train').rmdir()
        execute([sys.executable,directory/'code/full_fit.py',directory],directory/'full_fit.log',cwd=directory/'code')
    require((directory/'full_train/gru_model.pt').is_file() and (directory/'full_train/ridge_model.npz').is_file(),
            'Missing full-train checkpoints')
    return report


def model_smoke_check(config,root):
    """Exercise the installed scientific runtime on synthetic inputs, before real jobs."""
    import numpy as np
    import pandas as pd
    import pymbar
    import torch
    sys.path.insert(0,str(Path(config['tool_code'])/'refs'))
    import run_hito0_gru_rmsd as gru
    rng=np.random.default_rng(7)
    x=rng.normal(size=(8,201,8))
    y=x[:,-1,0]+np.linspace(-1,1,8)
    fit=np.arange(6); evaluation=np.array([6,7])
    mean,std,scale=gru.prepare(x,y,fit)
    gru.setup_seed(7)
    model=gru.ResidualGRU(32)
    zero=gru.predict(model,x,evaluation,mean,std,scale)
    require(np.array_equal(zero,x[evaluation,-1,0]),'Epoch-zero model must equal persistence')
    fitted=gru.fit_model(x,y,fit,mean,std,scale,7,2,32)
    predicted=gru.predict(fitted,x,evaluation,mean,std,scale)
    require(predicted.shape==(2,) and np.isfinite(predicted).all(),'Synthetic GRU check failed')
    atomic_json(Path(root)/'runtime_check.json',dict(status='PASS',torch=torch.__version__,
                numpy=np.__version__,pandas=pd.__version__,pymbar=pymbar.__version__,
                synthetic_only=True))
    print('SCIENTIFIC_RUNTIME_CHECK=PASS',flush=True)


def controller(root):
    from cohort import prepare
    root=Path(root).resolve()
    config=load_json(root/'config.json')
    config['tool_code']=str(root/'tool')
    config['pipeline_code']=str(root/'pipeline_code')
    work=Path(config['work'])
    lockpath=work/'maintenance/paper2_auto.lock'
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if config.get('input_manifest_sha256'):
            require(sha(config['manifest'])==config['input_manifest_sha256'],'Input manifest changed')
        for name,value in load_json(root/'tool_manifest.json').items():
            require(sha(root/'tool'/name)==value,f'Controller code changed: {name}')
        model_smoke_check(config,root)
        if (root/'base.json').is_file():
            base=load_json(root/'base.json')
            for key,expected in base['sha256'].items():
                require(sha(base[key])==expected,f'Changed base input: {key}')
        else:
            base=resolve_base(config)
            base['sha256']={k:sha(base[k]) for k in ('sources','assignment','inventory','sequences','future_rows','source_hashes')}
            atomic_json(root/'base.json',base)
        if (root/'selection.json').is_file():
            selection=load_json(root/'selection.json')
        else:
            selection=discover(work,base,config.get('group'),config.get('manifest'))
            require(selection,'No new complete archive pairs found; use --group or --manifest to select the upload directory')
            atomic_json(root/'selection.json',selection)
        if not (root/'pipeline_manifest.json').is_file():
            require(not (root/'pipeline_code').exists(),'Partial pipeline snapshot; inspect before resuming')
            snapshot_pipeline(config['project'],root/'pipeline_code',work,selection)
            if config.get('optional_oracle_diagnostics'):
                from recover_empty_oracle import install_extractor
                install_extractor(root/'pipeline_code')
            atomic_json(root/'pipeline_manifest.json',{
                str(p.relative_to(root/'pipeline_code')):sha(p)
                for p in (root/'pipeline_code').rglob('*') if p.is_file()})
        for name,value in load_json(root/'pipeline_manifest.json').items():
            require(sha(root/'pipeline_code'/name)==value,f'Pipeline snapshot changed: {name}')
        state=load_json(root/'state.json') if (root/'state.json').is_file() else {'systems':{}}
        def save():
            atomic_json(root/'state.json',state)
        print(f'SELECTED_SYSTEMS={len(selection)} PARALLEL_SYSTEMS={config["parallel"]}',flush=True)
        # Freeze and validate all archive pairs before submitting any processing.
        for item in selection:
            system=item['system']
            record=state['systems'].setdefault(system,{})
            for kind in ('analysis','protein'):
                path=Path(item['raw'])/'archives'/f'{system}_{kind}.zip'
                old=record.get(kind)
                if old:
                    require(stamp(path)==old['stamp'] and sha(path)==old['sha256'],f'Input archive changed: {path}')
                else:
                    record[kind]=inspect_archive(path)
                    expected=item.get(f'expected_{kind}_sha256')
                    if expected:
                        require(record[kind]['sha256']==expected,f'Upload checksum differs: {path}')
                    record[kind]['external_upload_hash_checked']=bool(expected)
                    save()
            print('ARCHIVES_VERIFIED '+system,flush=True)
        # Preflight and dry-run always precede real pipeline submissions.
        for item in selection:
            system=item['system']; record=state['systems'][system]
            if record.get('prepared'):
                continue
            for kind in ('analysis','protein'):
                unpack_archive(Path(item['raw'])/'archives'/f'{system}_{kind}.zip',
                               Path(item['raw'])/'unpacked'/f'{system}_{kind}',record[kind])
            directory=fresh_directory(root/'preflight'/system)
            execute([sys.executable,root/'pipeline_code/src/pipeline/preflight_atlas_system.py',
                '--system',system,'--replicas','1,2,3','--unpacked-dir',Path(item['raw'])/'unpacked',
                '--output-dir',directory,'--analysis-step-ps','100','--protein-step-ps','10',
                '--expected-end-ns','100','--time-tolerance-ps','1e-3'],root/f'preflight_{system}.log')
            execute(pipeline_command(config,item,True),root/f'dry_run_{system}.log')
            record['prepared']=True; save()
            print('PREFLIGHT_AND_DRY_RUN_PASS '+system,flush=True)
        if config['plan_only']:
            print('PAPER2_AUTO_PLAN=PASS; no processing or training jobs submitted',flush=True)
            return
        remaining=list(selection)
        while remaining:
            active=sum(1 for r in state['systems'].values() if r.get('submitted') and not r.get('terminal'))
            progress=False
            for item in list(remaining):
                system=item['system']; record=state['systems'][system]
                runroot=work/'results/atlas_pipeline_runs'/item['dataset_group']/system/config['run_id']
                if record.get('passed'):
                    remaining.remove(item); continue
                if record.get('failure'):
                    remaining.remove(item); continue
                idsfile=runroot/'job_ids.txt'
                if record.get('intent') and not record.get('submitted'):
                    require(idsfile.is_file(),f'Uncertain submission: {system}. Inspect its submission log and squeue; no duplicate submitted.')
                    record['ids']=job_ids(idsfile)
                    require('finalize' in record['ids'],'Recovered job register is incomplete; no duplicate submitted')
                    record['submitted']=True; save()
                if not record.get('submitted'):
                    if active>=config['parallel']:
                        continue
                    require(not runroot.exists(),f'Run already exists without this controller\'s state: {runroot}')
                    record['intent']=True; save()
                    execute(pipeline_command(config,item),root/f'submission_{system}.log')
                    require(idsfile.is_file(),f'Missing job_ids.txt: {system}')
                    record['ids']=job_ids(idsfile)
                    require('finalize' in record['ids'],'Incomplete pipeline job register')
                    record['submitted']=True; save(); active+=1; progress=True
                    print('PIPELINE_SUBMITTED '+system,flush=True)
                states,queued=slurm_states(record['ids'].values())
                outcome=jobs_outcome(record['ids'].values(),states,queued)
                if outcome=='UNKNOWN':
                    record.setdefault('unknown_since',time.time())
                    require(time.time()-record['unknown_since']<300,
                            f'{system}: accounting unknown for 5 minutes. Inspect Slurm; no duplicate submitted.')
                else:
                    record.pop('unknown_since',None)
                # If an upstream stage failed, cancel only pending dependents of this owned pipeline.
                failing=[(j,v) for j,v in states.items() if v[0] in FAILED or
                         (v[0]=='COMPLETED' and v[1]!='0:0')]
                if failing and queued:
                    pending=execute(['squeue','-h','-j',','.join(record['ids'].values()),'-t','PENDING','-o','%i']).split()
                    if pending:
                        execute(['scancel',*pending])
                if outcome=='PASS':
                    record['release']=verify_package(runroot,item,root/f'verify_{system}.log')
                    record.update(passed=True,terminal=True); active-=1
                    remaining.remove(item); save(); progress=True
                    print('PIPELINE_PASS '+system,flush=True)
                elif outcome=='FAILED':
                    record.update(failure=states,terminal=True); active-=1
                    remaining.remove(item); save(); progress=True
                    print('PIPELINE_FAILED '+system,flush=True)
            print('PIPELINE_PROGRESS='+str(sum(r.get('passed',False) for r in state['systems'].values()))+'/'+str(len(selection)),flush=True)
            if remaining and not progress:
                time.sleep(config['poll'])
        failed=[s for s,r in state['systems'].items() if r.get('failure')]
        require(not failed,'Independent systems processed, but training blocked by failures: '+','.join(failed)+'. See state.json and pipeline logs.')
        new_sources=[{'system':s,'release':r['release']} for s,r in state['systems'].items()]
        if not completed_step(root/'cohort'):
            directory=fresh_directory(root/'cohort')
            prepare(config,base,new_sources,selection,directory)
            finish_step(directory)
        report=train(config,root)
        if config['cleanup_unpacked']:
            from cleanup import cleanup
            cleanup(root,selection,state)
        cohort=root/'cohort'
        base_state={key:str(cohort/name) for key,name in {
            'sources':'sources.csv','assignment':'system_assignment.csv',
            'inventory':'source_inventory.csv','sequences':'sequences.fasta',
            'future_rows':'train_future_rmsd_rows.csv','source_hashes':'source_hashes.json'}.items()}
        base_state['sha256']={k:sha(v) for k,v in base_state.items()}
        base_state['acquired_exclusions']=base.get('acquired_exclusions',[])
        atomic_json(root/'cohort_state.json',base_state)
        atomic_json(work/'results/paper2_automation/LATEST_SUCCESS.json',{'cohort_state':str(root/'cohort_state.json')})
        atomic_json(root/'summary.json',{'status':'COMPLETE','new_systems':len(selection),
            'cohort':load_json(cohort/'summary.json'),'paired_audit':report,
            'full_train_model':str(root/'training/full_train/gru_model.pt'),
            'validation_test_evaluated':False,'HITO0_passed':False})
        (root/'PRISM_PROMPT.txt').write_text('Update paper2 only. Document this new exploratory train-only future-RMSD grouped CV as a separate experiment. '
            'Use summary.json and training/system_metrics.csv for counts, scores and paired whole-group intervals. '
            'Disclose conditional bootstrap uncertainty, sequence grouping, epoch-zero persistence fallback and missing static-structure/mdCATH controls. '
            'Do not claim HITO 0, test performance, anytime-valid coverage or measured HPC savings.\n')
        print('PAPER2_AUTO=PASS\nOUTPUT='+str(root)+'\nMODEL='+str(root/'training/full_train/gru_model.pt'),flush=True)


def launch(args,resume=False):
    if resume:
        from resume_frozen import launch as resume_with_queue_fix
        return resume_with_queue_fix(args.run)
    else:
        require(args.account,'Set PAPER2_SLURM_ACCOUNT or pass --account')
        require(args.partition,'Set PAPER2_SLURM_PARTITION or pass --partition')
        work=Path(args.work).expanduser().resolve()
        name=args.run_id or 'auto_'+time.strftime('%Y%m%d_%H%M%S')
        require(GROUP.fullmatch(name),'Unsafe run id')
        root=work/'results/paper2_automation'/name
        root.mkdir(parents=True,exist_ok=False)
        project=Path(args.project).expanduser().resolve() if args.project else work/'results/paper2_hito0/wave3_remaining117_20261014_v1/code'
        require((project/'slurm/submit_atlas_system_pipeline.sh').is_file(),f'Missing validated pipeline: {project}; pass --project')
        config=dict(work=str(work),run_id=name,project=str(project),group=args.group,
            manifest=str(Path(args.manifest).expanduser().resolve()) if args.manifest else None,
            base=args.base,parallel=args.parallel,poll=args.poll,plan_only=args.plan_only,
            cleanup_unpacked=args.cleanup_unpacked,account=args.account,partition=args.partition,
            qos=args.qos,blast_module=args.blast_module)
        config['optional_oracle_diagnostics']=True
        require(args.parallel>=1 and args.poll>=5,'parallel>=1 and poll>=5 required')
        if args.group:
            require(GROUP.fullmatch(args.group),'Unsafe dataset group')
        if args.manifest:
            shutil.copy2(config['manifest'],root/'input_manifest.csv')
            config['manifest']=str(root/'input_manifest.csv')
            config['input_manifest_sha256']=sha(root/'input_manifest.csv')
        atomic_json(root/'config.json',config)
        shutil.copytree(Path(__file__).parent,root/'tool',ignore=shutil.ignore_patterns('__pycache__','tests','*.pyc'))
        atomic_json(root/'tool_manifest.json',{
            str(p.relative_to(root/'tool')):sha(p) for p in (root/'tool').rglob('*') if p.is_file()})
    job=root/'controller_job.json'
    if job.is_file():
        prior=load_json(job)
        require(prior.get('id'),'Uncertain controller submission: inspect squeue; no duplicate submitted')
        states,active=slurm_states([prior['id']])
        require(jobs_outcome([prior['id']],states,active) not in ('ACTIVE','UNKNOWN'),'Controller still active or accounting unknown')
    script=root/'controller.sbatch'
    script.write_text('#!/bin/bash\nset -euo pipefail\n'+
        'exec bash '+shlex.quote(str(root/'tool/run.sh'))+' controller --run '+shlex.quote(str(root))+'\n')
    atomic_json(job,{'intent':True})
    identifier=execute(['sbatch','--parsable','--account='+config['account'],
        '--partition='+config['partition'],'--qos='+config['qos'],'--job-name=paper2_auto',
        '--cpus-per-task=1','--mem=6G','--time=12:00:00','--output='+str(root/'controller_%j.out'),script]).strip().split(';')[0]
    require(identifier.isdigit(),'Cannot parse controller job id')
    atomic_json(job,{'id':identifier})
    print('CONTROLLER_JOB='+identifier+'\nRUN='+str(root)+'\nLOG='+str(root/f'controller_{identifier}.out'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('launch',help='Discover uploads and submit the complete workflow')
    p.add_argument('--work',default=str(default_work()))
    p.add_argument('--group'); p.add_argument('--manifest'); p.add_argument('--project'); p.add_argument('--base')
    p.add_argument('--run-id'); p.add_argument('--parallel',type=int,default=8)
    p.add_argument('--poll',type=int,default=20)
    p.add_argument('--account',default=os.environ.get('PAPER2_SLURM_ACCOUNT'))
    p.add_argument('--partition',default=os.environ.get('PAPER2_SLURM_PARTITION'))
    p.add_argument('--qos',default='normal'); p.add_argument('--blast-module',default='BLAST+/2.14.0-gompi-2022b')
    p.add_argument('--plan-only',action='store_true',help='Verify, unpack, preflight and dry-run only')
    p.add_argument('--cleanup-unpacked',action='store_true',help='Verify archive recovery and remove only new unpacked XTC after successful training')
    for name in ('controller','resume','status'):
        p=sub.add_parser(name); p.add_argument('--run',required=True)
    for name in ('repair-oracle','repair-oracle-worker'):
        p=sub.add_parser(name)
        p.add_argument('--run',required=True); p.add_argument('--system',required=True)
    args=parser.parse_args()
    try:
        if args.command=='launch':
            launch(args)
        elif args.command=='resume':
            launch(args,True)
        elif args.command=='controller':
            controller(args.run)
        elif args.command in ('repair-oracle','repair-oracle-worker'):
            from recover_empty_oracle import launch as repair_launch, perform
            (repair_launch if args.command=='repair-oracle' else perform)(args.run,args.system)
        else:
            root=Path(args.run).expanduser()
            for name in ('controller_job.json','state.json','summary.json'):
                if (root/name).is_file():
                    print(name+'\n'+(root/name).read_text())
    except Exception as error:
        if getattr(args,'run',None) and Path(args.run).is_dir():
            atomic_json(Path(args.run)/'last_error.json',{'error':str(error),'traceback':traceback.format_exc()})
        print('PAPER2_AUTO=FAILED: '+str(error),file=sys.stderr,flush=True)
        raise


if __name__=='__main__':
    main()
