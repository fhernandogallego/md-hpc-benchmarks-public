"""Optional post-training XTC cleanup, with complete archive recovery checks."""
from pathlib import Path
import getpass
import json
import os
import zipfile

from common import atomic_json, checked_members, digest, execute, load_json, require, sha, stamp


def cleanup(root, selection, state):
    root=Path(root)
    directory=root/'cleanup'
    directory.mkdir(exist_ok=True)
    if (directory/'summary.json').is_file():
        return load_json(directory/'summary.json')
    require((root/'training/paired_audit.json').is_file() and
            (root/'training/full_train/metadata.json').is_file(),'Training/audit incomplete; cleanup blocked')
    selected={r['system']:r for r in selection}
    require('3txs_C' not in selected,'Known excluded system must not be cleaned')
    queued=execute(['squeue','-h','-u',os.environ.get('USER') or getpass.getuser(),'-o','%i|%j']).splitlines()
    require(not any(line.split('|',1)[-1].strip().endswith('_'+s)
                    for line in queued for s in selected),'A pipeline for a selected system is still active')
    planfile=directory/'verified_plan.json'
    if planfile.is_file():
        evidence=load_json(planfile)
        require(sha(planfile)==(directory/'verified_plan.sha256').read_text().strip(),'Changed cleanup plan')
    else:
        files,archives=[],[]
        for system,item in sorted(selected.items()):
            require(state['systems'][system].get('passed'),f'Pipeline not validated: {system}')
            raw=Path(item['raw']).resolve(strict=True)
            found={}
            allowed={f'{system}_R{r}.xtc' for r in (1,2,3)}|{f'{system}_prod_R{r}_fit.xtc' for r in (1,2,3)}
            paths=sorted((raw/'unpacked').rglob('*.xtc'))
            require(len(paths)==6 and {p.name for p in paths}==allowed,f'Unexpected XTC layout: {system}')
            by_name={p.name:p for p in paths}
            for p in paths:
                require(not p.is_symlink() and p.resolve()==p,f'Unsafe XTC path: {p}')
            for kind in ('analysis','protein'):
                archive=raw/'archives'/f'{system}_{kind}.zip'
                original=state['systems'][system][kind]
                require(sha(archive)==original['sha256'],f'Changed archive: {archive}')
                before=stamp(archive)
                with zipfile.ZipFile(archive) as z:
                    for info in checked_members(z):
                        if info.is_dir():
                            continue
                        with z.open(info) as stream:
                            member_sha=digest(stream)
                        name=Path(info.filename).name
                        if name in by_name:
                            require(name not in found,'Ambiguous archived XTC')
                            p=by_name[name]
                            before_file=stamp(p)
                            require(member_sha==sha(p) and info.file_size==p.stat().st_size,f'Recovery differs: {p}')
                            require(stamp(p)==before_file,f'File changed during comparison: {p}')
                            found[name]=dict(system=system,path=str(p),sha256=member_sha,
                                bytes=info.file_size,stamp=before_file,archive=str(archive),member=info.filename)
                require(stamp(archive)==before,f'Archive changed during comparison: {archive}')
                archives.append(dict(path=str(archive),sha256=original['sha256'],stamp=before))
            require(set(found)==allowed,'Some XTC files cannot be recovered')
            files.extend(found.values())
            print('RECOVERABILITY_PASS '+system,flush=True)
        evidence=dict(files=files,archives=archives)
        atomic_json(planfile,evidence)
        (directory/'verified_plan.sha256').write_text(sha(planfile)+'\n')
    # Revalidate the complete plan before deleting anything, including after interruption.
    require(len(evidence['files'])==6*len(selection),'Unexpected cleanup count')
    seen=set()
    for row in evidence['archives']:
        p=Path(row['path'])
        require(not p.is_symlink() and stamp(p)==row['stamp'] and sha(p)==row['sha256'],f'Changed recovery archive: {p}')
    for row in evidence['files']:
        require(row['system'] in selected,'Cleanup plan names an unselected system')
        p=Path(row['path'])
        p.relative_to(Path(selected[row['system']]['raw']).resolve()/'unpacked')
        require(p.suffix=='.xtc' and str(p) not in seen,'Unsafe/duplicate deletion target')
        seen.add(str(p))
        if p.exists():
            require(not p.is_symlink() and stamp(p)==row['stamp'] and sha(p)==row['sha256'],f'Changed deletion target: {p}')
    removed=0
    print('ALL_PRE_DELETE_CHECKS=PASS',flush=True)
    with (directory/'deletion.jsonl').open('a') as log:
        for row in evidence['files']:
            p=Path(row['path'])
            if not p.exists():
                continue  # Already removed in a prior interrupted attempt.
            require(not p.is_symlink() and stamp(p)==row['stamp'],f'Changed deletion target: {p}')
            log.write(json.dumps({'event':'before_unlink',**row})+'\n'); log.flush(); os.fsync(log.fileno())
            p.unlink(); removed+=1
            log.write(json.dumps({'event':'deleted','path':str(p)})+'\n'); log.flush(); os.fsync(log.fileno())
    summary=dict(selected_systems=len(selection),planned_xtc_files=len(evidence['files']),
        deleted_this_attempt=removed,logical_GiB=sum(r['bytes'] for r in evidence['files'])/2**30,
        archives_retained=len(evidence['archives']))
    atomic_json(directory/'summary.json',summary)
    print('XTC_CLEANUP=PASS',flush=True)
    return summary
