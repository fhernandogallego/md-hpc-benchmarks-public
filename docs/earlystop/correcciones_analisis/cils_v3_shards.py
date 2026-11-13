#!/usr/bin/env python3
"""CILS v3: frozen dev-cohort target shards, immutable checkpoints, verified merge.

No MD simulations and no validation/test input. Run selftest before any Slurm job.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WORK = Path('/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks')
CODE = WORK / 'maintenance/paper2_cils_v3_20261112_101652/package/calculos'
INV = WORK / 'results/paper2_miti_20261109/inputs/inventory_miti.csv'
TRAIN = WORK / 'results/paper2_automation/auto_20261030_091302/training'
COHORT = WORK / 'results/paper2_automation/auto_20261030_091302/cohort/system_assignment.csv'
PACK = WORK / 'maintenance/paper2_cils_v3_20261112_101652'
STATE = WORK / 'maintenance/paper2_cils_v3_relaunch_v1'
NAMES = ('core_rmsd_q30', 'helix', 'rg', 'rmsd', 'sasa', 'strand')
POLICIES = ('include', 'exclude')
FIELDS = ('constant_window', 'constant_policy', 'sample_id', 'system_id',
          'replica', 'target', 't0_ns', 'future_valid', 'future_target',
          'future_duration_ns', 'future_neff', 'future_g')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            raise ValueError('CSV sin cabecera: ' + str(path))
        return rd.fieldnames, list(rd)


def write_csv(path, names, entries):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=names)
        wr.writeheader()
        wr.writerows(entries)


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(row):
    return row.get('system') or row.get('system_id') or ''


def ensure(x, message):
    if not x:
        raise ValueError(message)


def setup(inventory=INV, train=TRAIN, cohort=COHORT, code=CODE,
          pack=PACK, state=STATE, n_expected=757, chunk=10):
    inventory, train, cohort, code, pack, state = map(Path,
        (inventory, train, cohort, code, pack, state))
    ensure(chunk >= 1, 'shard size invalido')
    deps = {'inventory': inventory, 'folds': train / 'outer_folds.csv',
            'cohort': cohort, 'builder': code / 'build_targets.py',
            'config': code / 'targets_config.json',
            'package_manifest': pack / 'paquete_revision_scayle_v3.manifest.json',
            'package_tar': pack / 'paquete_revision_scayle_v3.tar.gz'}
    for key, path in deps.items():
        ensure(path.is_file(), 'Falta %s: %s' % (key, path))
    manifest = load_json(deps['package_manifest'])
    ensure(sha(deps['package_tar']) == manifest['archive_sha256'], 'Tar v3 alterado')
    for name, p in [('calculos/build_targets.py', deps['builder']),
                    ('calculos/targets_config.json', deps['config'])]:
        ensure(sha(p) == manifest['files_sha256'][name], 'Codigo v3 alterado: ' + name)
    config = load_json(deps['config'])
    ensure(tuple(config['targets']) == NAMES, 'Se cambio la configuracion de 6 objetivos')
    ensure(config['targets']['core_rmsd_q30']['equilibration'] == 'detect'
           and config['targets']['core_rmsd_q30']['column'] == 'replica_prefix_rmsd_A',
           'Core q30 no usa equilibration propia')
    heads, invrows = rows(inventory)
    ids = [identity(r) for r in invrows]
    ensure(len(ids) == n_expected and all(ids) and len(set(ids)) == n_expected,
           'Inventario no contiene exactamente %d sistemas unicos' % n_expected)
    _, folds = rows(deps['folds'])
    foldids = [r['system_id'] for r in folds]
    ensure(len(foldids) == n_expected and set(foldids) == set(ids),
           'El inventario no coincide con outer_folds de development')
    grp = {}
    for r in folds:
        g = r['group_id']; k = r['outer_fold']
        ensure(g not in grp or grp[g] == k, 'Fuga de grupo en folds: '+g)
        grp[g] = k
    _, assignments = rows(cohort)
    trainset = {r['system_id'] for r in assignments if r.get('split') == 'train'}
    ensure(set(ids) <= trainset, 'Hay sistemas que no son train; no abrir validation/test')
    if state.exists():
        old = checked_plan(state)
        ensure(old['hashes'] == {k:sha(v) for k,v in deps.items()}
               and old['ids'] == ids and old['chunk_size'] == chunk,
               'Estado anterior no coincide con entradas congeladas')
        print('INIT_REUSED=PASS shards=%d' % len(old['shards']), flush=True)
        return old
    stage = Path(tempfile.mkdtemp(prefix='.init_', dir=state.parent))
    try:
        (stage / 'shards').mkdir()
        shutil.copy2(inventory, stage / 'frozen_inventory.csv')
        specs=[]
        for i, start in enumerate(range(0, len(ids), chunk)):
            subset=invrows[start:start+chunk]
            shardfile=stage / 'shards' / ('shard_%03d.csv' % i)
            write_csv(shardfile, heads, subset)
            specs.append({'index':i, 'name':shardfile.name,
                          'ids':ids[start:start+chunk], 'sha256':sha(shardfile)})
        plan = {'version':1, 'created':stamp(), 'ids':ids, 'expected_n':n_expected,
                'chunk_size':chunk, 'shards':specs,
                'paths':{k:str(p.resolve()) for k,p in deps.items()},
                'hashes':{k:sha(p) for k,p in deps.items()},
                'frozen_inventory_sha256':sha(stage/'frozen_inventory.csv')}
        write_json(stage / 'plan.json', plan)
        stage.rename(state)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print('INIT_PASS systems=%d shards=%d chunk=%d state=%s' %
          (len(ids),len(specs),chunk,state), flush=True)
    return plan


def checked_plan(state):
    state=Path(state)
    plan=load_json(state/'plan.json')
    ensure(plan['version'] == 1 and plan['expected_n'] == len(plan['ids']), 'Plan invalido')
    ensure(len(set(plan['ids'])) == len(plan['ids']), 'IDs de plan duplicados')
    for k,p in plan['paths'].items():
        ensure(Path(p).is_file() and sha(p) == plan['hashes'][k],
               'Entrada congelada ha cambiado: '+k)
    ensure(sha(state/'frozen_inventory.csv') == plan['frozen_inventory_sha256'],
           'Inventario congelado alterado')
    expected=[]
    for i,spec in enumerate(plan['shards']):
        ensure(spec['index'] == i, 'Indice de shard inconsistente')
        path=state/'shards'/spec['name']
        ensure(path.is_file() and sha(path) == spec['sha256'],
               'Fragmento alterado: %d' % i)
        _, chunkrows=rows(path)
        ensure([identity(r) for r in chunkrows] == spec['ids'],
               'Fragmento no contiene IDs previstos')
        expected.extend(spec['ids'])
    ensure(expected == plan['ids'], 'Fragmentos no reconstruyen inventario original')
    return plan


def validate_data(path, ids, policy):
    fields, data=rows(path)
    ensure(tuple(fields) == FIELDS, 'CSV de objetivos cambió de esquema')
    ordered=[(s,t,str(r)) for s in ids for t in NAMES for r in (1,2,3)]
    actual=[]
    for row in data:
        sid=row['system_id']; t=row['target']; rep=row['replica']
        ensure(row['sample_id'] == sid+'_R'+rep, 'sample_id incompatible')
        ensure(row['constant_policy'] == policy, 'Politica incorrecta')
        ensure(row['constant_window'].lower() in ('true','false'), 'Constante invalida')
        ensure(row['future_valid'].lower() in ('true','false'), 'Validez invalida')
        if policy == 'exclude' and row['constant_window'].lower() == 'true':
            ensure(row['future_valid'].lower() == 'false', 'Constante no excluida')
        if row['future_valid'].lower() == 'true':
            ensure(math.isfinite(float(row['future_target'])), 'Target valido no finito')
        actual.append((sid,t,rep))
    ensure(len(actual)==len(ordered), 'Faltan o sobran filas (%d vs %d)' %
           (len(actual),len(ordered)))
    ensure(len(set(actual))==len(actual), 'Filas duplicadas')
    ensure(actual==ordered, 'Orden, sistemas u objetivos distintos al plan')
    return data


def validate_pair(include, exclude):
    ensure(len(include)==len(exclude), 'Dos politicas de longitud diferente')
    for a,b in zip(include,exclude):
        ensure(a['constant_policy']=='include' and b['constant_policy']=='exclude',
               'Etiquetas de politica incorrectas')
        for k in FIELDS:
            if k == 'constant_policy':
                continue
            if k in ('future_valid','future_target') and a['constant_window'].lower()=='true':
                continue
            ensure(a[k] == b[k], 'Politicas difieren fuera de constantes: '+k+' '+a['sample_id'])


def check_part(state, plan, i, policy):
    spec=plan['shards'][i]
    folder=Path(state)/'parts'/('%s_%03d' % (policy,i))
    ensure(folder.is_dir(), 'FRAGMENTO_FALTANTE: '+folder.name)
    cp=load_json(folder/'complete.json')
    ensure(cp['version']==1 and cp['policy']==policy and cp['shard']==i,
           'Checkpoint discordante')
    ensure(cp['plan_sha256']==sha(Path(state)/'plan.json') and
           cp['shard_sha256']==spec['sha256'] and
           cp['builder_sha256']==plan['hashes']['builder'] and
           cp['config_sha256']==plan['hashes']['config'], 'Checkpoint obsoleto')
    for name in ('targets.csv','targets.summary.json','build.log'):
        ensure(sha(folder/name)==cp['files'][name], 'HASH_MISMATCH '+folder.name+'/'+name)
    data=validate_data(folder/'targets.csv',spec['ids'],policy)
    summary=load_json(folder/'targets.summary.json')
    ensure(summary['n_problems']==0 and summary['targets_sha256']==cp['files']['targets.csv']
           and summary['script_sha256']==plan['hashes']['builder'] and
           summary['config_sha256']==plan['hashes']['config'] and
           summary['inventory_sha256']==spec['sha256'], 'Summary inconsistente')
    for t in NAMES:
        target=[r for r in data if r['target']==t]
        counts=summary['per_target'][t]
        ensure(counts['replicas']==len(target) and
               counts['valid_replicas']==sum(r['future_valid'].lower()=='true' for r in target) and
               counts['valid_systems']==len({r['system_id'] for r in target if r['future_valid'].lower()=='true'}),
               'Summary por target no coincide: '+t)
    return cp, data


def run_one(state, i, policy, extra_env=None):
    ensure(policy in POLICIES, 'Politica invalida')
    plan=checked_plan(state)
    ensure(0<=i<len(plan['shards']), 'Indice de fragmento fuera de rango')
    state=Path(state); (state/'parts').mkdir(exist_ok=True)
    (state/'locks').mkdir(exist_ok=True)
    lockpath=state/'locks'/('%s_%03d.lock' % (policy,i))
    with open(lockpath,'a+b') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Fragmento en ejecucion por otro proceso: '+lockpath.name)
        final=state/'parts'/('%s_%03d'%(policy,i))
        if final.exists():
            cp,_=check_part(state,plan,i,policy)
            print('SHARD_REUSED=PASS shard=%03d policy=%s elapsed_sec=%s' %
                  (i,policy,cp['elapsed_seconds']),flush=True)
            return cp
        stage=Path(tempfile.mkdtemp(prefix='.partial_%s_%03d_'%(policy,i),dir=state/'parts'))
        t0=time.monotonic()
        usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            cmd=[sys.executable,plan['paths']['builder'], '--inventory',
                 str(state/'shards'/plan['shards'][i]['name']),
                 '--config', plan['paths']['config'],'--constant-policy',policy,
                 '--out',str(stage/'targets.csv')]
            env=os.environ.copy()
            if extra_env: env.update(extra_env)
            env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                        'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
            print('SHARD_START shard=%03d policy=%s log=%s' % (i, policy, stage/'build.log'), flush=True)
            with open(stage/'build.log','wb') as log:
                p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            if p.returncode != 0:
                raise RuntimeError('Build fallo rc=%d, ver %s' % (p.returncode,stage/'build.log'))
            data=validate_data(stage/'targets.csv',plan['shards'][i]['ids'],policy)
            summary=load_json(stage/'targets.summary.json')
            ensure(summary['n_problems']==0 and
                   summary['inventory_sha256']==plan['shards'][i]['sha256'] and
                   summary['script_sha256']==plan['hashes']['builder'] and
                   summary['config_sha256']==plan['hashes']['config'] and
                   summary['targets_sha256']==sha(stage/'targets.csv'),
                   'Summary de fragmento no valida')
            for t in NAMES:
                target=[r for r in data if r['target']==t]
                stats=summary['per_target'][t]
                ensure(stats['replicas']==len(target) and
                       stats['valid_replicas']==sum(r['future_valid'].lower()=='true' for r in target) and
                       stats['valid_systems']==len({r['system_id'] for r in target if r['future_valid'].lower()=='true'}),
                       'Summary no coincide: '+t)
            u=resource.getrusage(resource.RUSAGE_CHILDREN)
            files={name:sha(stage/name) for name in
                   ('targets.csv','targets.summary.json','build.log')}
            cp={'version':1,'shard':i,'policy':policy,'finished':stamp(),
                'plan_sha256':sha(state/'plan.json'),
                'shard_sha256':plan['shards'][i]['sha256'],
                'builder_sha256':plan['hashes']['builder'],
                'config_sha256':plan['hashes']['config'],
                'n_systems':len(plan['shards'][i]['ids']),
                'n_rows':len(data),'elapsed_seconds':round(time.monotonic()-t0,3),
                'child_cpu_seconds':round((u.ru_utime+u.ru_stime)-
                                              (usage0.ru_utime+usage0.ru_stime),3),
                'child_maxrss_kib':u.ru_maxrss,'files':files}
            write_json(stage/'complete.json',cp)
            stage.rename(final)
            print('SHARD_PASS shard=%03d policy=%s systems=%d rows=%d elapsed_sec=%.3f maxrss_kib=%s' %
                  (i,policy,cp['n_systems'],cp['n_rows'],cp['elapsed_seconds'],cp['child_maxrss_kib']),flush=True)
            return cp
        except BaseException:
            print('SHARD_INCOMPLETE stage=%s (no checkpoint; can retry)'%stage, file=sys.stderr,flush=True)
            raise


def consolidate(state):
    state=Path(state); plan=checked_plan(state)
    parts={}; missing=[]
    for policy in POLICIES:
        for i in range(len(plan['shards'])):
            try:
                _,part=check_part(state,plan,i,policy)
                parts[(policy,i)]=part
            except (FileNotFoundError,ValueError,KeyError) as e:
                missing.append('%s_%03d: %s'%(policy,i,e))
    ensure(not missing, 'CONSOLIDATE_BLOCKED:\n'+'\n'.join(missing[:40]))
    data={policy:[r for i in range(len(plan['shards'])) for r in parts[(policy,i)]]
          for policy in POLICIES}
    for policy in POLICIES:
        validate_ids=plan['ids']
        expected_len=len(validate_ids)*len(NAMES)*3
        ensure(len(data[policy])==expected_len,'Consolidacion incompleta')
        keys=[(r['system_id'],r['target'],r['replica']) for r in data[policy]]
        ensure(len(set(keys))==expected_len,'Filas duplicadas al consolidar')
        ensure([r['system_id'] for r in data[policy][::18]]==validate_ids,
               'Orden global de sistemas distinto al inventario')
    validate_pair(data['include'],data['exclude'])
    final=state/'merged'
    if final.exists():
        report=load_json(final/'MERGE_PASS.json')
        ensure(report['plan_sha256']==sha(state/'plan.json'), 'Merge previo obsoleto')
        for name,digest in report['sha256'].items():
            ensure(sha(final/name)==digest,'Merge alterado: '+name)
        print('MERGE_REUSED=PASS rows_per_policy=%d'%len(data['include']))
        return report
    stage=Path(tempfile.mkdtemp(prefix='.merge_partial_',dir=state))
    try:
        sha_out={}
        for policy in POLICIES:
            filename='targets_%s_constants.csv'%policy
            write_csv(stage/filename,list(FIELDS),data[policy])
            sha_out[filename]=sha(stage/filename)
            counts={}
            for t in NAMES:
                rr=[r for r in data[policy] if r['target']==t]
                counts[t]={'replicas':len(rr),
                           'valid_replicas':sum(r['future_valid'].lower()=='true' for r in rr),
                           'valid_systems':len({r['system_id'] for r in rr if r['future_valid'].lower()=='true'})}
            summary={'calculation':'revised_build_targets_sharded','constant_policy':policy,
                     'inventory_sha256':plan['hashes']['inventory'],
                     'config_sha256':plan['hashes']['config'],
                     'script_sha256':plan['hashes']['builder'],
                     'targets_sha256':sha_out[filename], 'n_problems':0,
                     'per_target':counts, 'shards':len(plan['shards'])}
            write_json(stage/('targets_%s_constants.summary.json'%policy),summary)
            sha_out['targets_%s_constants.summary.json'%policy]=sha(stage/('targets_%s_constants.summary.json'%policy))
        report={'version':1,'created':stamp(),'n_systems':len(plan['ids']),
                'n_shards':len(plan['shards']), 'rows_per_policy':len(data['include']),
                'plan_sha256':sha(state/'plan.json'),'sha256':sha_out,
                'cohort_scope':'train_only; no validation/test',
                'constant_policy_comparison':'PASS; differences confined to constant windows'}
        write_json(stage/'MERGE_PASS.json',report)
        stage.rename(final)
        print('MERGE_PASS systems=%d rows_per_policy=%d'%
              (len(plan['ids']),len(data['include'])),flush=True)
        return report
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


FAKE_BUILDER = r'''import argparse,csv,hashlib,json,math,os,sys
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('inventory','config','constant-policy','out'):p.add_argument('--'+name,required=True)
a=p.parse_args()
with open(a.inventory,newline='') as f: data=list(csv.DictReader(f))
if os.getenv('SIMULATE_ABORT')=='1' and any(row['system']=='s03' for row in data):
    print('SYNTHETIC_CRASH_BEFORE_OUTPUT',flush=True);sys.exit(37)
targets=list(json.loads(Path(a.config).read_text())['targets'])
cols=('constant_window','constant_policy','sample_id','system_id','replica','target','t0_ns','future_valid','future_target','future_duration_ns','future_neff','future_g')
res=[]
for row in data:
    s=row['system']
    for t in targets:
        for i in (1,2,3):
            const=(s=='s02' and t=='strand' and i==1)
            valid=not(const and a.constant_policy=='exclude')
            res.append(dict(constant_window=str(const),constant_policy=a.constant_policy,sample_id=f'{s}_R{i}',system_id=s,replica=str(i),target=t,t0_ns='0.0',future_valid=str(valid),future_target='1.0' if valid else 'nan',future_duration_ns='40.0',future_neff='20.0',future_g='1.0'))
with open(a.out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(res)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary={'n_problems':0,'script_sha256':sha(__file__),'config_sha256':sha(a.config),'inventory_sha256':sha(a.inventory),'targets_sha256':sha(a.out),
'per_target':{t:{'replicas':len([r for r in res if r['target']==t]),'valid_replicas':len([r for r in res if r['target']==t and r['future_valid']=='True']),'valid_systems':len({r['system_id'] for r in res if r['target']==t and r['future_valid']=='True'})} for t in targets}}
Path(a.out).with_suffix('.summary.json').write_text(json.dumps(summary))
print('SYNTHETIC_BUILDER_PASS')
'''


def selftest(report_path):
    tests={}
    with tempfile.TemporaryDirectory(prefix='earlystop_synth_') as scratch:
        root=Path(scratch); (root/'code').mkdir(); (root/'train').mkdir(); (root/'pack').mkdir()
        script=root/'code/build_targets.py'; script.write_text(FAKE_BUILDER)
        config=root/'code/targets_config.json'
        write_json(config,{'targets':{t:({'equilibration':'detect','column':'replica_prefix_rmsd_A'}
                 if t=='core_rmsd_q30' else {}) for t in NAMES}})
        inv=root/'inventory.csv'; ids=['s%02d'%i for i in range(1,6)]
        write_csv(inv,['system'],[{'system':s} for s in ids])
        write_csv(root/'train/outer_folds.csv',['system_id','group_id','outer_fold'],
                  [{'system_id':s,'group_id':s,'outer_fold':str(i%5)} for i,s in enumerate(ids)])
        write_csv(root/'cohort.csv',['system_id','split'],[{'system_id':s,'split':'train'} for s in ids])
        tar=root/'pack/paquete_revision_scayle_v3.tar.gz';tar.write_bytes(b'testonly')
        write_json(root/'pack/paquete_revision_scayle_v3.manifest.json',
            {'archive_sha256':sha(tar),'files_sha256':{
                'calculos/build_targets.py':sha(script),'calculos/targets_config.json':sha(config)}})
        state=root/'state'
        setup(inv,root/'train',root/'cohort.csv',root/'code',root/'pack',state,5,2)
        tests['plan_5_systems_3_shards']=len(checked_plan(state)['shards'])==3
        cp0=run_one(state,0,'include'); original=sha(state/'parts/include_000/targets.csv')
        try: run_one(state,1,'include',{'SIMULATE_ABORT':'1'})
        except RuntimeError: tests['interruption_detected']=True
        else: tests['interruption_detected']=False
        tests['completed_shard_preserved']=(sha(state/'parts/include_000/targets.csv')==original)
        tests['partial_shard_has_no_checkpoint']=not (state/'parts/include_001').exists()
        tests['resume_skips_verified_result']=run_one(state,0,'include')['files']['targets.csv']==original
        run_one(state,1,'include');run_one(state,2,'include')
        try: consolidate(state)
        except ValueError as e: tests['missing_shards_detected']='CONSOLIDATE_BLOCKED' in str(e)
        else: tests['missing_shards_detected']=False
        for i in range(3):run_one(state,i,'exclude')
        bad=rows(state/'parts/include_000/targets.csv')[1]
        try:validate_fake_rows(bad+[bad[0]],['s01','s02'],'include')
        except ValueError:tests['duplicate_row_detected']=True
        else:tests['duplicate_row_detected']=False
        try:validate_fake_rows(bad[:-1],['s01','s02'],'include')
        except ValueError:tests['missing_row_detected']=True
        else:tests['missing_row_detected']=False
        original_bytes=(state/'parts/include_000/targets.csv').read_bytes()
        (state/'parts/include_000/targets.csv').write_bytes(original_bytes+b'\n')
        try:check_part(state,checked_plan(state),0,'include')
        except ValueError:tests['hash_tampering_detected']=True
        else:tests['hash_tampering_detected']=False
        (state/'parts/include_000/targets.csv').write_bytes(original_bytes)
        result=consolidate(state)
        tests['merge_complete']=result['n_systems']==5 and result['rows_per_policy']==90
        tests['merge_resume_verifies']=consolidate(state)['sha256']==result['sha256']
    payload={'date':stamp(),'checks':tests,'success':all(tests.values()),'synthetic_only':True}
    report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
    write_json(report_path,payload)
    for name,passed in tests.items():print('TEST_%s=%s'%(name,'PASS' if passed else 'FAIL'))
    print('SELFTEST=%s report=%s'%('PASS' if payload['success'] else 'FAIL',report_path))
    ensure(payload['success'],'Tests fallaron')


def validate_fake_rows(fake_rows,ids,policy):
    with tempfile.NamedTemporaryFile(mode='w',suffix='.csv',newline='',delete=False) as f:
        path=Path(f.name)
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(fake_rows)
    try:return validate_data(path,ids,policy)
    finally:path.unlink(missing_ok=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('init')
    run=sub.add_parser('run');run.add_argument('--shard',type=int,required=True)
    run.add_argument('--policy',choices=POLICIES,required=True)
    sub.add_parser('verify');sub.add_parser('consolidate')
    tst=sub.add_parser('selftest');tst.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    if args.cmd=='selftest': selftest(args.report)
    elif args.cmd=='init':setup()
    elif args.cmd=='run':run_one(STATE,args.shard,args.policy)
    elif args.cmd=='verify':
        plan=checked_plan(STATE);ok=0;miss=[]
        for i in range(len(plan['shards'])):
            for policy in POLICIES:
                try:check_part(STATE,plan,i,policy);ok+=1
                except (FileNotFoundError,ValueError,KeyError) as e:miss.append('%s_%03d:%s'%(policy,i,e))
        print('VERIFIED_CHECKPOINTS=%d/%d'%(ok,2*len(plan['shards'])))
        print('NOT_VERIFIED='+str(miss[:30]))
        ensure(not miss,'No todos los fragmentos completos (esperable durante el piloto)')
    elif args.cmd=='consolidate':consolidate(STATE)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('ERROR: '+str(e),file=sys.stderr,flush=True)
        sys.exit(1)
