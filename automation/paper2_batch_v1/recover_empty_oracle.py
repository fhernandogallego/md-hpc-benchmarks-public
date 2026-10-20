"""Versioned recovery of empty full-oracle diagnostics, preserving prefix features."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
from pathlib import Path
import shlex
import shutil
import sys
import time

from common import (FAILED, SYSTEM, atomic_json, execute, job_ids, jobs_outcome, load_json,
                    read_csv, require, sha, slurm_states)


def install_extractor(destination):
    destination=Path(destination)
    source=Path(__file__).parent/'recovery/extract_atlas_core_rmsd.py'
    target=destination/'src/pipeline/extract_atlas_core_rmsd.py'
    # Only the executed reference version or this version is supported.
    require(sha(target) in (REFERENCE_SHA,sha(source)),
            'Core extractor differs from the supported reference; refusing an unreviewed replacement')
    shutil.copyfile(source,target)
    stage=destination/'slurm/atlas_pipeline_stage.sbatch'
    text=stage.read_text()
    needle='"$PROJECT/src/pipeline/extract_atlas_core_rmsd.py" \\\n'
    require(text.count(needle)==1,'Unexpected core-RMSD stage layout')
    if '--allow-empty-oracle' not in text:
        text=text.replace(needle,needle+'          --allow-empty-oracle \\\n')
        stage.write_text(text)
    execute(['bash','-n',stage])


REFERENCE_SHA='6b0d325b60473f9354915edc8b6512f074a85d34d802ddf42f785a992d052e77'


def wait_recovery(ids,poll):
    unknown_since=None
    while True:
        states,active=slurm_states(ids)
        failing=any(s in FAILED or (s=='COMPLETED' and c!='0:0') for s,c in states.values())
        if failing and active:
            pending=execute(['squeue','-h','-j',','.join(ids),'-t','PENDING','-o','%i']).split()
            if pending:
                execute(['scancel',*pending])
        outcome=jobs_outcome(ids,states,active)
        require(not failing,'Recovery pipeline failed; original cohort state remains unchanged: '+str(states))
        if outcome=='PASS':
            return
        if outcome=='UNKNOWN':
            unknown_since=unknown_since or time.time()
            require(time.time()-unknown_since<300,'Recovery accounting unknown; no duplicate submitted')
        else:
            unknown_since=None
        print('WAIT_ORACLE_RECOVERY='+outcome,flush=True)
        time.sleep(poll)


def diagnosis(root,system,item,config,record):
    import pandas as pd
    old=Path(config['work'])/'results/atlas_pipeline_runs'/item['dataset_group']/system/config['run_id']
    identifier=record['ids']['core_rmsd']
    logs=list((old/'logs').glob('*_'+identifier+'.err'))
    require(len(logs)==1,'Cannot identify the original core-RMSD error log')
    message=logs[0].read_text(errors='replace')
    require('RuntimeError:' in message and any(
        'el núcleo '+name+' está vacío.' in message
        for name in ('replica_full_oracle','consensus_full_oracle')),
        'Recovery is restricted to a confirmed empty full-oracle diagnostic')
    source=Path(__file__).parent/'recovery/extract_atlas_core_rmsd.py'
    spec=importlib.util.spec_from_file_location('optional_oracle_extractor',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tables=old/'processed'/item['dataset_group']/system/'core_overlap'
    replica_path=tables/'core_replica_residues.csv'
    consensus_path=tables/'core_consensus_residues.csv'
    replicas=pd.read_csv(replica_path); consensus=pd.read_csv(consensus_path)
    module.validate_core_tables(replicas,consensus,system,[1,2,3])
    missing=[]; counts=[]
    for r in (1,2,3):
        for q in (.3,.4,.5):
            sets=module.selected_residue_indices(replicas,consensus,r,q,allow_empty_oracle=True)
            counts.append(dict(replica=r,quantile=q,sizes={v:len(x) for v,x in sets.items()}))
            missing.extend(dict(replica=r,quantile=q,variant=v) for v,x in sets.items() if len(x)==0)
    require(missing,'No empty oracle diagnostic remains; inspect before recovery')
    require(all(r['variant'].endswith('_full_oracle') for r in missing),'Empty prefix core')
    return dict(original_run_root=str(old),original_job_ids=record['ids'],
        original_failure=record['failure'],error_log=str(logs[0]),error_log_sha256=sha(logs[0]),
        core_table_sha256={str(p):sha(p) for p in (replica_path,consensus_path)},
        core_sizes=counts,unavailable_diagnostics=missing,
        policy='Only empty full-oracle diagnostics are unavailable; all prefix cores remain mandatory',
        model_features_changed=False)


def resume_original(root):
    root=Path(root)
    if (root/'summary.json').is_file():
        print('ORIGINAL_RUN_ALREADY_COMPLETE',flush=True)
        return
    if (root/'controller_job.json').is_file():
        prior=load_json(root/'controller_job.json')
        require(prior.get('id'),'Uncertain controller submission; inspect before resume')
        states,active=slurm_states([prior['id']])
        outcome=jobs_outcome([prior['id']],states,active)
        if outcome=='ACTIVE':
            print('ORIGINAL_CONTROLLER_ALREADY_ACTIVE='+prior['id'],flush=True)
            return
        require(outcome!='UNKNOWN','Controller accounting unknown; no duplicate submitted')
    print(execute(['bash',root/'tool/run.sh','resume','--run',root]),flush=True)


def perform(root,system):
    from controller import pipeline_command, snapshot_pipeline, verify_package
    root=Path(root).resolve()
    config=load_json(root/'config.json'); work=Path(config['work'])
    repair=root/'recovery'/(system+'_oracle_v1')
    lockpath=work/'maintenance/paper2_auto.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for manifest,directory in (('tool_manifest.json','tool'),('pipeline_manifest.json','pipeline_code')):
            for name,value in load_json(root/manifest).items():
                require(sha(root/directory/name)==value,'Changed frozen source: '+directory+'/'+name)
        for name,value in load_json(repair/'tool_manifest.json').items():
            require(sha(repair/'tool'/name)==value,'Changed recovery tool: '+name)
        state=load_json(root/'state.json'); record=state['systems'][system]
        item=next(r for r in load_json(root/'selection.json') if r['system']==system)
        if not record.get('passed'):
            require(record.get('failure') and record.get('terminal'),'Expected a terminal failed pipeline')
            _,active=slurm_states(record['ids'].values())
            require(not active,'Original system pipeline still active')
            evidence=diagnosis(root,system,item,config,record)
            atomic_json(repair/'diagnosis.json',evidence)
            snapshot=repair/'pipeline_code'
            if not (repair/'pipeline_manifest.json').is_file():
                require(not snapshot.exists(),'Incomplete recovery snapshot; inspect before retry')
                snapshot_pipeline(root/'pipeline_code',snapshot,work,[item])
                install_extractor(snapshot)
                atomic_json(repair/'pipeline_manifest.json',{
                    str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()})
            for name,value in load_json(repair/'pipeline_manifest.json').items():
                require(sha(snapshot/name)==value,'Changed recovery pipeline: '+name)
            current={**config,'pipeline_code':str(snapshot),'run_id':config['run_id']+'_oracle_v1'}
            runroot=work/'results/atlas_pipeline_runs'/item['dataset_group']/system/current['run_id']
            register=repair/'submission.json'
            if not register.is_file():
                require(not runroot.exists(),'Recovery run already exists without submission record')
                execute(pipeline_command(current,item,True),repair/'dry_run.log')
                atomic_json(register,{'intent':True,'run_root':str(runroot)})
                execute(pipeline_command(current,item),repair/'submission.log')
            require((runroot/'job_ids.txt').is_file(),'Uncertain recovery submission; inspect logs; no duplicate submitted')
            ids=job_ids(runroot/'job_ids.txt')
            require('finalize' in ids,'Incomplete recovery job register; no duplicate submitted')
            atomic_json(register,{'ids':ids,'run_root':str(runroot)})
            wait_recovery(list(ids.values()),config['poll'])
            release=verify_package(runroot,item,repair/'verify.log')
            metadata=load_json(runroot/'processed'/item['dataset_group']/system/'core_rmsd/core_rmsd.json')
            unavailable=[dict(replica=r['replica'],quantile=r['quantile'],variant=v)
                for r in metadata['trajectory_validation']
                for v,s in r['diagnostic_status'].items() if s=='UNAVAILABLE_EMPTY_ORACLE_CORE']
            require(unavailable and all(r['variant'].endswith('_full_oracle') for r in unavailable),
                    'Expected a documented missing oracle and complete prefix cores')
            prior={k:v for k,v in record.items() if k in ('ids','failure','terminal','submitted','release','passed')}
            state_path=root/'state.json'
            backup=repair/'state_before_recovery.json'
            if not backup.is_file():
                shutil.copyfile(state_path,backup)
            evidence.update(recovery_run_root=str(runroot),release=release,recovery_job_ids=ids,
                unavailable_diagnostics=unavailable,extractor_sha256=sha(snapshot/'src/pipeline/extract_atlas_core_rmsd.py'))
            atomic_json(repair/'completion.json',evidence)
            record.setdefault('recovery_history',[]).append(dict(previous=prior,evidence=str(repair/'completion.json')))
            record.pop('failure',None); record.pop('unknown_since',None)
            record.update(passed=True,terminal=True,submitted=True,ids=ids,release=release,
                          recovery_run_root=str(runroot))
            atomic_json(state_path,state)
            print('ORACLE_RECOVERY=PASS\nSYSTEM='+system+'\nRELEASE='+release,flush=True)
        else:
            require((repair/'completion.json').is_file(),'Passed system is not this recovery; no mutation allowed')
    resume_original(root)


def launch(root,system):
    root=Path(root).expanduser().resolve()
    require(SYSTEM.fullmatch(system),'Unsafe system id')
    config=load_json(root/'config.json')
    require(not (root/'summary.json').is_file(),'Original run already complete')
    record=load_json(root/'state.json')['systems'][system]
    require(record.get('failure') or record.get('recovery_history'),'Expected a recorded failed system')
    identifier=load_json(root/'controller_job.json').get('id')
    require(identifier,'Original controller job ID missing')
    states,active=slurm_states([identifier])
    require(jobs_outcome([identifier],states,active) not in ('ACTIVE','UNKNOWN'),
            'Original controller active or accounting unknown')
    repair=root/'recovery'/(system+'_oracle_v1')
    repair.mkdir(parents=True,exist_ok=True)
    jobfile=repair/'repair_job.json'
    if jobfile.is_file():
        prior=load_json(jobfile)
        require(prior.get('id'),'Uncertain repair submission; inspect squeue before retry')
        states,active=slurm_states([prior['id']])
        outcome=jobs_outcome([prior['id']],states,active)
        if outcome=='ACTIVE':
            print('RECOVERY_JOB='+prior['id']+'\nLOG='+str(repair/f'recovery_{prior["id"]}.out'))
            return
        require(outcome!='UNKNOWN','Repair accounting unknown; no duplicate submitted')
    if not (repair/'tool_manifest.json').is_file():
        require(not (repair/'tool').exists(),'Partial recovery tool snapshot')
        shutil.copytree(Path(__file__).parent,repair/'tool',ignore=shutil.ignore_patterns('__pycache__','*.pyc','tests'))
        atomic_json(repair/'tool_manifest.json',{
            str(p.relative_to(repair/'tool')):sha(p) for p in (repair/'tool').rglob('*') if p.is_file()})
    script=repair/'repair.sbatch'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexec bash '+
        shlex.quote(str(repair/'tool/run.sh'))+' repair-oracle-worker --run '+shlex.quote(str(root))+
        ' --system '+shlex.quote(system)+'\n')
    atomic_json(jobfile,{'intent':True})
    identifier=execute(['sbatch','--parsable','--account='+config['account'],
        '--partition='+config['partition'],'--qos='+config['qos'],'--job-name=paper2_oracle_recovery',
        '--cpus-per-task=1','--mem=6G','--time=04:00:00','--output='+str(repair/'recovery_%j.out'),script]).strip().split(';')[0]
    require(identifier.isdigit(),'Cannot parse repair job ID')
    atomic_json(jobfile,{'id':identifier})
    print('RECOVERY_JOB='+identifier+'\nLOG='+str(repair/f'recovery_{identifier}.out'),flush=True)
