"""Versioned cohort construction; never selects or evaluates held-out labels."""
from __future__ import annotations

import ast
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import shlex
import shutil
import sys

from common import (atomic_json, execute, fresh_directory, load_json, read_csv,
                    require, sha, write_csv)

AA = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
              'ARNDCQEGHILKMFPSTWYV'))
ALIASES = {'HID':'HIS', 'HIE':'HIS', 'HIP':'HIS', 'HSD':'HIS', 'HSE':'HIS',
           'HSP':'HIS', 'ASH':'ASP', 'GLH':'GLU', 'LYN':'LYS', 'CYM':'CYS',
           'CYX':'CYS', 'MSE':'MET'}
MODELS = ['trained_mean', 'last_value_persistence', 'snapshot0_ridge',
          'ridge_summary', 'gru_residual_rmsd_pilot']


def pdb_sequence(path, system):
    chains = defaultdict(dict)
    for line in Path(path).read_text(errors='replace').splitlines():
        if line.startswith('ENDMDL'):
            break
        if line[:6].strip() not in ('ATOM', 'HETATM') or line[12:16].strip() != 'CA':
            continue
        name = line[17:20].strip()
        if line.startswith('HETATM') and name not in AA and name not in ALIASES:
            continue
        chain, residue = line[21:22], line[22:27]
        require(residue not in chains[chain] or chains[chain][residue] == name,
                f'Alternative residue identities: {path} {chain}:{residue}')
        chains[chain][residue] = name
    requested = system.split('_', 1)[1]
    chain = requested if requested in chains else next(iter(chains)) if len(chains) == 1 else None
    require(chain is not None, f'Ambiguous PDB chain: {path}; requested {requested}')
    residues = list(chains[chain].values())
    require(residues and all(AA.get(ALIASES.get(r, r)) for r in residues), f'Unknown/empty sequence: {path}')
    return ''.join(AA[ALIASES.get(r, r)] for r in residues)


def read_fasta(path):
    result, key = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]
            require(key not in result, f'Duplicate FASTA identifier: {key}')
            result[key] = ''
        elif line.strip():
            require(key is not None, 'FASTA sequence without identifier')
            result[key] += line.strip()
    require(result and all(result.values()), 'Empty FASTA')
    return result


def groups_from_pairs(systems, pairs, partitions):
    parent = {s:s for s in systems}
    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    def join(a, b):
        require(a in parent and b in parent, f'Unknown pair: {a}, {b}')
        a, b = root(a), root(b)
        parent[max(a,b)] = min(a,b)
    for a, b in pairs:
        join(a,b)
    pdbs = defaultdict(list)
    for s in systems:
        pdbs[s.split('_',1)[0]].append(s)
    for members in pdbs.values():
        for other in members[1:]:
            join(members[0], other)
    groups = defaultdict(list)
    for s in systems:
        groups[root(s)].append(s)
    conflicts = [sorted(m) for m in groups.values() if len({partitions[s] for s in m}) > 1]
    return {s:'grp_'+min(m) for m in groups.values() for s in m}, conflicts


def make_folds(rows, seed=20261016):
    import hashlib
    members, replicas = defaultdict(set), defaultdict(int)
    for row in rows:
        if str(row['future_valid']).lower() == 'true':
            members[row['group_id']].add(row['system_id'])
            replicas[row['system_id']] += 1
    require(len(members) >= 15, 'Fewer than 15 labeled groups; five-fold nested CV is not supported')
    ordered = sorted(members, key=lambda g:(-len(members[g]),
                      hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest(),g))
    bins, loads = [[] for _ in range(5)], [0]*5
    for group in ordered:
        f = min(range(5), key=lambda i:(loads[i],len(bins[i]),i))
        bins[f].append(group)
        loads[f] += len(members[group])
    return [dict(system_id=s, group_id=g, outer_fold=f, n_valid_replicas=replicas[s])
            for f, groups in enumerate(bins) for g in groups for s in sorted(members[g])]


def load_function(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generalize_worker(source, destination):
    """Only replace cohort sizes in the published, previously executed worker."""
    mapping = {170:'n_systems', 149:'n_groups', 288:'n_replicas', 666:'n_samples'}
    class Counts(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in mapping:
                return ast.copy_location(ast.Subscript(value=ast.Name(id='counts',ctx=ast.Load()),
                    slice=ast.Constant(mapping[node.value]),ctx=ast.Load()),node)
            if type(node.value) is int and node.value == 34:
                return ast.copy_location(ast.parse("counts['fold_systems'][str(fold)]",mode='eval').body,node)
            if isinstance(node.value,str) and 'systems=34' in node.value:
                return ast.copy_location(ast.Constant(node.value.replace(' systems=34','')),node)
            return node
    tree = Counts().visit(ast.parse(Path(source).read_text()))
    position = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='run' for t in n.targets)) + 1
    tree.body.insert(position,ast.parse("counts=json.loads((run/'counts.json').read_text())").body[0])
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + '\n'
    compile(code,str(destination),'exec')
    Path(destination).write_text(code)


def prepare(config, base, new_sources, selection, output):
    import numpy as np
    import pandas as pd
    from pymbar import timeseries
    output = Path(output)
    sources = {r['system']:dict(r) for r in read_csv(base['sources'])}
    previous = {r['system_id']:r for r in read_csv(base['assignment'])}
    require(set(sources)==set(previous),'Base assignment and source registry differ')
    for r in new_sources:
        require(r['system'] not in sources,'New system already in base')
        sources[r['system']] = dict(r)
        previous[r['system']] = {'system_id':r['system'],'split':'train','origin':config['run_id']}
    write_csv(output/'sources.csv',[sources[s] for s in sorted(sources)],['system','release'])
    command = [sys.executable,'-m','src.dataset.assemble_multisystem_dataset',
               '--output-dir',str(output/'dataset')]
    for s in sorted(sources):
        command += ['--source',s+'='+sources[s]['release']]
    execute(command,output/'assembly.log',cwd=config['pipeline_code'])
    sequences = read_fasta(base['sequences'])
    require(set(sequences)==set(sources)-{r['system'] for r in new_sources},'Base FASTA differs')
    pdb_evidence = []
    for item in selection:
        system = item['system']
        paths = [Path(item['raw'])/'unpacked'/f'{system}_{kind}'/f'{system}.pdb'
                 for kind in ('analysis','protein')]
        parsed = [pdb_sequence(p,system) for p in paths]
        require(len(set(parsed))==1,f'PDB sequences disagree: {system}')
        sequences[system]=parsed[0]
        pdb_evidence.extend({'system':system,'path':str(p),'sha256':sha(p)} for p in paths)
    atomic_json(output/'pdb_evidence.json',pdb_evidence)
    with (output/'sequences.fasta').open('w') as f:
        for system in sorted(sequences):
            f.write('>'+system+'\n'+sequences[system]+'\n')
    fasta, db = output/'sequences.fasta', output/'blast_db'
    shell = 'module purge\nmodule load '+shlex.quote(config['blast_module'])+'\n'
    shell += shlex.join(['makeblastdb','-in',str(fasta),'-dbtype','prot','-out',str(db)])+'\n'
    shell += shlex.join(['blastp','-version'])+'\n'
    shell += shlex.join(['blastp','-task','blastp','-query',str(fasta),'-db',str(db),
             '-num_threads','1','-seg','no','-evalue','100','-max_target_seqs',str(len(sources)),
             '-outfmt','6 qseqid sseqid pident length qlen slen qstart qend sstart send evalue bitscore',
             '-out',str(output/'all_alignments.tsv')])
    execute(['bash','-lc','set -e\n'+shell],output/'blast.log')
    pairs, self_hits = set(), set()
    for line in (output/'all_alignments.tsv').read_text().splitlines():
        q,s,pid,ln,ql,sl,qs,qe,ss,se,ev,bs=line.split()
        require(q in sources and s in sources,'BLAST identifier mismatch')
        if q==s:
            self_hits.add(q)
        elif float(pid)>=30 and (int(qe)-int(qs)+1)/int(ql)>=.8 and (int(se)-int(ss)+1)/int(sl)>=.8:
            pairs.add(tuple(sorted((q,s))))
    require(self_hits==set(sources),'Missing BLAST self-alignments')
    identical=defaultdict(list)
    for s,sequence in sequences.items():
        identical[sequence].append(s)
    for members in identical.values():
        pairs.update((members[0],s) for s in members[1:])
    group_ids,conflicts=groups_from_pairs(sources,pairs,{s:r['split'] for s,r in previous.items()})
    atomic_json(output/'group_conflicts.json',conflicts)
    require(not conflicts,'Sequence groups cross protected partitions; see group_conflicts.json. Training blocked.')
    assignment=[dict(system_id=s,split=previous[s]['split'],group_id=group_ids[s],
                     origin=previous[s].get('origin','base')) for s in sorted(sources)]
    write_csv(output/'system_assignment.csv',assignment)
    write_csv(output/'candidate_pairs.csv',[dict(system_a=a,system_b=b) for a,b in sorted(pairs)],
              ['system_a','system_b'])
    source_inventory={r['system']:dict(r) for r in read_csv(base['inventory'])}
    for r in new_sources:
        s,p=r['system'],Path(r['release']).parent
        source_inventory[s]=dict(system=s,master_targets_csv=str(p/'master_targets/master_targets_long.csv'),
            **{f'basic_R{i}_csv':str(p/'basic'/f'{s}_R{i}_basic_observables.csv') for i in (1,2,3)})
    inventory_rows=[]
    for s in sorted(sources):
        r={k:source_inventory[s].get(k,'') for k in
           ('system','master_targets_csv','basic_R1_csv','basic_R2_csv','basic_R3_csv')}
        r['system']=s
        r['scope']=previous[s]['split']
        inventory_rows.append(r)
    write_csv(output/'source_inventory.csv',inventory_rows)
    # Cache reuse is conditioned on hashes of all referenced original CSVs.
    old_rows={r['sample_id']:r for r in read_csv(base['future_rows'])}
    old_hashes=load_json(base['source_hashes'])
    builder=load_function(Path(config['tool_code'])/'refs/future_builder.py','future_reference')
    records,hashes=[],{}
    for row in inventory_rows:
        s=row['system']
        if row['scope']!='train':
            continue
        master=Path(row['master_targets_csv'])
        hashes[s]={'master_csv':str(master),'master_sha256':sha(master),'replicas':{}}
        target_rows=[r for r in read_csv(master) if r['system']==s and r['target_name']=='rmsd']
        target_map={int(r['replica']):r for r in target_rows}
        require(len(target_rows)==3 and set(target_map)=={1,2,3},f'RMSD master incomplete: {s}')
        for replica in (1,2,3):
            path=Path(row[f'basic_R{replica}_csv'])
            digest=sha(path)
            hashes[s]['replicas'][str(replica)]={'path':str(path),'sha256':digest}
            sid=f'{s}_R{replica}'
            if sid in old_rows:
                old=old_hashes[s]
                require(old['master_sha256']==hashes[s]['master_sha256'] and
                        old['replicas'][str(replica)]['sha256']==digest,
                        f'Historical source changed: {sid}; refusing stale target reuse')
                r=dict(old_rows[sid])
                for flag in ('original_valid','future_valid','common_valid'):
                    r[flag]=builder.valid_flag(r[flag])
                for key in ('original_t0_ns','original_target_A','original_mean_delta_A',
                            'new_t0_ns','future_duration_ns','future_g','future_neff','future_target_A'):
                    r[key]=float(r[key])
                r['replica']=int(r['replica'])
                r['future_n_frames']=int(r['future_n_frames'])
            else:
                data=pd.read_csv(path,usecols=['time_ns','rmsd_backbone_A'])
                old=target_map[replica]
                v=builder.valid_flag(old['target_valid'])
                if v:
                    require(math.isfinite(float(old['target_value'])) and
                            abs(float(old['target_value'])-float(old['automatic_window_value']))<=1e-6,
                            f'Original target inconsistent: {sid}')
                result=builder.recalculate(data['time_ns'].to_numpy(dtype=np.float64),
                    data['rmsd_backbone_A'].to_numpy(dtype=np.float64),float(old['t0_ns']),
                    float(old['automatic_window_value']),timeseries.statistical_inefficiency)
                r=dict(system_id=s,replica=replica,sample_id=sid,
                       original_t0_ns=float(old['t0_ns']),original_valid=v,
                       original_target_A=float(old['target_value']) if v else math.nan,
                       **result,common_valid=v and result['future_valid'])
            r.update(group_id=group_ids[s],origin=previous[s].get('origin','base'))
            records.append(r)
    require(len(records)==3*sum(a['split']=='train' for a in assignment),'Train replica count mismatch')
    fields=['system_id','replica','sample_id','group_id','origin','original_t0_ns','original_valid',
            'original_target_A','original_mean_delta_A','new_t0_ns','future_duration_ns',
            'future_n_frames','future_g','future_neff','future_valid','future_target_A','common_valid']
    write_csv(output/'train_future_rmsd_rows.csv',records,fields)
    atomic_json(output/'source_hashes.json',hashes)
    folds=make_folds(records)
    write_csv(output/'outer_folds.csv',folds)
    atomic_json(output/'summary.json',dict(systems=len(sources),samples=3*len(sources),
        train_systems=sum(a['split']=='train' for a in assignment),groups=len(set(group_ids.values())),
        labeled_systems=len(folds),labeled_groups=len({r['group_id'] for r in folds}),
        labeled_replicas=sum(r['future_valid'] for r in records),
        validation_test_evaluated=False,base_future_targets_reused=len(old_rows)))


def training_bundle(config, cohort, output):
    import numpy as np
    output=Path(output)
    refs=Path(config['tool_code'])/'refs'
    shutil.copytree(refs,output/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    generalize_worker(refs/'train_atlas222_grouped.py',output/'code/worker.py')
    shutil.copy2(Path(config['tool_code'])/'full_fit.py',output/'code/full_fit.py')
    for name in ('train_future_rmsd_rows.csv','outer_folds.csv'):
        shutil.copy2(Path(cohort)/name,output/name)
    assignment={r['system_id']:r for r in read_csv(Path(cohort)/'system_assignment.csv')}
    with np.load(Path(cohort)/'dataset/multisystem_dataset_raw.npz',allow_pickle=False) as z:
        decode=lambda a:np.asarray([s.decode() if isinstance(s,bytes) else str(s) for s in a])
        systems=decode(z['system_ids'])
        take=np.array([assignment[s]['split']=='train' for s in systems])
        values=np.full((int(take.sum()),5),np.nan)
        mask=np.zeros(values.shape,dtype=bool)
        ids=decode(z['sample_ids'])[take]
        targets={r['sample_id']:r for r in read_csv(output/'train_future_rmsd_rows.csv')}
        require(set(ids)==set(targets),'Training input and target identifiers differ')
        for i,sid in enumerate(ids):
            if targets[sid]['future_valid'].lower()=='true':
                values[i,0]=float(targets[sid]['future_target_A'])
                mask[i,0]=True
        np.savez_compressed(output/'dataset.npz',X_raw=z['X_raw'][take],
             system_ids=systems[take],sample_ids=ids,time_ns=z['time_ns'],
             feature_names=z['feature_names'],target_names=z['target_names'],
             target_values=values,target_mask=mask)
    folds=read_csv(output/'outer_folds.csv')
    counts=dict(n_systems=len(folds),n_groups=len({r['group_id'] for r in folds}),
                n_replicas=int(mask.sum()),n_samples=len(ids),
                fold_systems={str(f):sum(int(r['outer_fold'])==f for r in folds) for f in range(5)})
    atomic_json(output/'counts.json',counts)
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    atomic_json(output/'manifest.json',{'files':files,'base_seed':20261007,
                'hidden_size':32,'max_epochs':150,'validation_test_present':False})


def audit_training(directory):
    import numpy as np
    directory=Path(directory)
    counts=load_json(directory/'counts.json')
    folds={r['system_id']:r for r in read_csv(directory/'outer_folds.csv')}
    rows=read_csv(directory/'system_predictions.csv')
    valid_targets={r['sample_id']:r for r in read_csv(directory/'train_future_rmsd_rows.csv')
                   if r['future_valid'].lower()=='true'}
    grouped={m:{} for m in MODELS}
    for r in rows:
        m,s=r['model'],r['system_id']
        require(m in grouped and s in folds and s not in grouped[m],'Duplicate/unknown OOF prediction')
        require(r['group_id']==folds[s]['group_id'] and r['fold']==folds[s]['outer_fold'],'OOF group/fold differs')
        grouped[m][s]=r
    require(all(set(r)==set(folds) for r in grouped.values()),'Missing OOF predictions')
    for f in range(5):
        metadata=load_json(directory/f'fold_{f}/metadata.json')
        protected=set(metadata['outer_eval_groups'])
        for inner in metadata['ridge_inner_eval_groups']:
            require(protected.isdisjoint(inner),'Inner/outer group leak')
        require(protected.isdisjoint(metadata['selection']['gru_residual_rmsd_pilot']['inner_eval_groups']),
                'GRU inner/outer group leak')
        rr=read_csv(directory/f'fold_{f}/replica_predictions.csv')
        seen=set()
        for r in rr:
            sid=r['sample_id']
            require(sid in valid_targets and (r['model'],sid) not in seen,'Unknown/duplicate OOF replica')
            seen.add((r['model'],sid))
            target=valid_targets[sid]
            require(r['system_id']==target['system_id'] and r['group_id']==target['group_id'],
                    'OOF replica identity differs from target')
            require(int(r['fold'])==f==int(folds[r['system_id']]['outer_fold']),'OOF replica fold differs')
            require(np.isclose(float(r['true']),float(target['future_target_A']),atol=1e-9,rtol=0),
                    'OOF truth differs from frozen future target')
        require(len(rr)==len(MODELS)*sum(int(r['n_valid_replicas']) for r in folds.values()
                 if int(r['outer_fold'])==f),'Unexpected OOF replica row count')
        for m in MODELS:
            for s in (s for s in folds if int(folds[s]['outer_fold'])==f):
                selected=[r for r in rr if r['model']==m and r['system_id']==s]
                require(len(selected)==int(folds[s]['n_valid_replicas']),'Replica count differs')
                for key in ('true','predicted'):
                    require(np.isclose(np.mean([float(r[key]) for r in selected]),
                            float(grouped[m][s][key]),atol=1e-9,rtol=0),'Replica aggregation differs')
    systems=sorted(folds)
    truth=np.array([float(grouped[MODELS[0]][s]['true']) for s in systems])
    errors={}
    for m in MODELS:
        require(np.allclose(truth,[float(grouped[m][s]['true']) for s in systems],atol=1e-9,rtol=0),'Model targets differ')
        errors[m]=np.abs(np.array([float(grouped[m][s]['predicted']) for s in systems])-truth)
        require(np.isfinite(errors[m]).all(),'Nonfinite OOF error')
    groups=sorted({folds[s]['group_id'] for s in systems})
    members=np.array([[folds[s]['group_id']==g for s in systems] for g in groups])
    sizes=members.sum(axis=1)
    group_errors={m:members@errors[m] for m in MODELS}
    rng=np.random.default_rng(20261016)
    boot={m:[] for m in MODELS}
    for _ in range(100):
        draw=rng.integers(0,len(groups),size=(100,len(groups)))
        denominator=sizes[draw].sum(axis=1)
        for m in MODELS:
            boot[m].extend((group_errors[m][draw].sum(axis=1)/denominator).tolist())
    comparisons=[]
    for a,b in [('gru_residual_rmsd_pilot','ridge_summary'),
                ('gru_residual_rmsd_pilot','last_value_persistence'),
                ('ridge_summary','last_value_persistence')]:
        comparisons.append(dict(contrast=a+' minus '+b,delta_mae=float((errors[a]-errors[b]).mean()),
            bootstrap_95=np.quantile(np.array(boot[a])-np.array(boot[b]),[.025,.975]).tolist()))
    report=dict(counts=counts,comparisons=comparisons,bootstrap_draws=10000,
                scope='Exploratory whole-group bootstrap conditional on fixed OOF predictions; no retraining',
                validation_test_evaluated=False,model_predictions_sha256=sha(directory/'system_predictions.csv'))
    atomic_json(directory/'paired_audit.json',report)
    return report
