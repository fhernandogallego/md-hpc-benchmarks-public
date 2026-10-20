"""Resume a frozen controller with a separately recorded Slurm I/O runtime fix."""
from pathlib import Path
import shlex
import shutil

from common import atomic_json, execute, jobs_outcome, load_json, require, sha, slurm_states


def launch(root):
    root=Path(root).expanduser().resolve()
    if (root/'summary.json').is_file():
        print('ORIGINAL_RUN_ALREADY_COMPLETE',flush=True)
        return
    config=load_json(root/'config.json')
    jobfile=root/'controller_job.json'
    if jobfile.is_file():
        prior=load_json(jobfile)
        require(prior.get('id'),'Uncertain controller submission; inspect before resume')
        states,active=slurm_states([prior['id']])
        outcome=jobs_outcome([prior['id']],states,active)
        if outcome=='ACTIVE':
            print('ORIGINAL_CONTROLLER_ALREADY_ACTIVE='+prior['id'],flush=True)
            return
        require(outcome!='UNKNOWN','Controller accounting unknown; no duplicate submitted')
    original=root/'tool'
    original_manifest=load_json(root/'tool_manifest.json')
    for name,value in original_manifest.items():
        require(sha(original/name)==value,'Changed frozen source: '+name)
    updated=Path(__file__).parent/'common.py'
    # The scientific code remains byte-for-byte the frozen version; common.py
    # is the dependency-free I/O helper. Original files/manifests stay intact.
    runtime=root/'controller_runtimes'/('slurm_queue_'+sha(updated)[:12])
    manifest=runtime/'manifest.json'
    if not manifest.is_file():
        require(not runtime.exists(),'Incomplete Slurm runtime snapshot; inspect before retry')
        runtime.mkdir(parents=True)
        shutil.copytree(original,runtime/'tool',ignore=shutil.ignore_patterns('__pycache__','*.pyc','tests'))
        shutil.copyfile(updated,runtime/'tool/common.py')
        for name,value in original_manifest.items():
            if name!='common.py':
                require(sha(runtime/'tool'/name)==value,'Changed scientific source: '+name)
        metadata=dict(
            original_manifest_sha256=sha(root/'tool_manifest.json'),
            original_common_sha256=original_manifest['common.py'],
            updated_common_sha256=sha(updated),
            change='Query active user jobs before looking up requested IDs in sacct',
            scientific_code_changed=False,
            files={str(p.relative_to(runtime)):sha(p) for p in (runtime/'tool').rglob('*') if p.is_file()})
        atomic_json(manifest,metadata)
    evidence=load_json(manifest)
    require(evidence['original_manifest_sha256']==sha(root/'tool_manifest.json'),'Changed runtime parent')
    for name,value in evidence['files'].items():
        require(sha(runtime/name)==value,'Changed Slurm runtime: '+name)
    # Validate the immutable runtime again on the compute node before execution.
    checker=runtime/'verify_and_run.py'
    checker.write_text(
        'from pathlib import Path\nimport hashlib,json,os,sys\n'
        'here=Path(__file__).resolve().parent\n'
        'for name,wanted in json.loads((here/"manifest.json").read_text())["files"].items():\n'
        '    actual=hashlib.sha256((here/name).read_bytes()).hexdigest()\n'
        '    if actual!=wanted: raise RuntimeError("Changed runtime: "+name)\n'
        'os.execv("/bin/bash",["bash",str(here/"tool/run.sh"),"controller","--run",sys.argv[1]])\n')
    script=root/'controller.sbatch'
    script.write_text('#!/bin/bash\nset -euo pipefail\n'+
        'exec '+shlex.quote(__import__('sys').executable)+' '+shlex.quote(str(checker))+' '+shlex.quote(str(root))+'\n')
    atomic_json(jobfile,{'intent':True,'runtime_manifest':str(manifest)})
    identifier=execute(['sbatch','--parsable','--account='+config['account'],
        '--partition='+config['partition'],'--qos='+config['qos'],'--job-name=paper2_auto',
        '--cpus-per-task=1','--mem=6G','--time=12:00:00','--output='+str(root/'controller_%j.out'),script]).strip().split(';')[0]
    require(identifier.isdigit(),'Cannot parse resumed controller job ID')
    atomic_json(jobfile,{'id':identifier,'runtime_manifest':str(manifest)})
    print('CONTROLLER_JOB='+identifier+'\nRUN='+str(root)+'\nLOG='+str(root/f'controller_{identifier}.out'),flush=True)
