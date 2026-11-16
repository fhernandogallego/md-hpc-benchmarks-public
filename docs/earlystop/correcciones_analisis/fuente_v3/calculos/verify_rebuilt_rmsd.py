#!/usr/bin/env python3
"""Fail before fitting if rebuilding changes the archived RMSD training reference."""
import argparse
from pathlib import Path
import numpy as np
from common import read_csv, load_folds, write_json, sha

def verify(run, rebuilt_targets):
    folds=load_folds(run)
    original=read_csv(run/'train_future_rmsd_rows.csv')
    rebuilt=[r for r in read_csv(rebuilt_targets) if r['target']=='rmsd']
    def eligible(rows):
        return {r['sample_id']:r for r in rows if r['system_id'] in folds
                and r['future_valid'].lower() in ('true','1')}
    x,y=eligible(original),eligible(rebuilt)
    if set(x)!=set(y):
        raise ValueError(f'RMSD valid-mask mismatch: removed={sorted(set(x)-set(y))[:10]}, added={sorted(set(y)-set(x))[:10]}')
    if not x:raise ValueError('No valid training labels to verify')
    delta=max(abs(float(x[s]['future_target_A'])-float(y[s]['future_target'])) for s in x)
    if not np.isfinite(delta) or delta>1e-8:
        raise ValueError(f'Rebuilt RMSD labels differ by up to {delta} A: inspect before continuing')
    return {'valid_mask_identical':True,'maximum_absolute_label_difference_A':delta,
        'n_valid_replicas':len(x),'original_sha256':sha(run/'train_future_rmsd_rows.csv'),
        'rebuilt_sha256':sha(rebuilt_targets)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--rebuilt-targets',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    write_json(a.out,verify(a.run,a.rebuilt_targets))
    print('RMSD_REFERENCE_CHECK=PASS')

if __name__=='__main__':main()
