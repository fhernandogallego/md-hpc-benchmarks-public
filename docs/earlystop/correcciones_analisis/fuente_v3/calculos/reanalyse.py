#!/usr/bin/env python3
"""Ridge sensitivities and externally evaluated split-conformal diagnostics.

Only training-cohort rows in an existing outer_folds.csv are eligible. No validation
or test predictions are read or generated. Existing point estimates are not replaced.
Requires numpy; uses common.py copied unchanged from the archived calculations.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common import (ALPHAS_EXTENDED, SEED, load_prefix, load_folds, read_csv,
                    write_csv, write_json, sha, summarize, ridge_cv_predict,
                    ordered, metrics, group_bootstrap_delta)

CHANNEL = {'rmsd': 0, 'rg': 1, 'core_rmsd_q30': 2, 'sasa': 5, 'helix': 6, 'strand': 7}
SAFE_CHANNELS = (0, 1, 5, 6, 7)

def partition(groups, folds, outer_fold):
    """One outer evaluation fold; 20% of remaining groups for calibration only."""
    ev = np.flatnonzero(folds == outer_fold)
    pool = np.flatnonzero(folds != outer_fold)
    names = ordered(groups[pool], SEED + outer_fold)
    if len(names) < 10:
        raise ValueError('Need at least ten non-evaluation groups for inner CV and calibration')
    cal_names = names[:max(2, round(len(names) * .2))]
    ca = pool[np.isin(groups[pool], cal_names)]
    tr = pool[~np.isin(groups[pool], cal_names)]
    sets = [set(groups[a]) for a in (tr, ca, ev)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Training/calibration/evaluation groups overlap')
    return tr, ca, ev

def quantile(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError('Miscoverage must lie strictly between zero and one')
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Calibration scores must be nonempty and finite')
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float('inf') if k > len(scores) else float(np.sort(scores)[k - 1])

def aggregate(idx, systems, y, pred, sigma):
    names = sorted(set(systems[idx]))
    bins = [idx[systems[idx] == s] for s in names]
    return (names, np.array([y[b].mean() for b in bins]),
            np.array([pred[b].mean() for b in bins]),
            np.array([sigma[b].mean() for b in bins]))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--targets', type=Path, required=True)
    ap.add_argument('--only', choices=CHANNEL, default='rmsd')
    ap.add_argument('--mode', choices=['prefix', 'intervals'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--draws', type=int, default=10000)
    a = ap.parse_args()
    if a.mode == 'prefix' and a.only == 'core_rmsd_q30':
        ap.error('Core target changes with core definition: use independently reconstructed per-horizon targets')
    if a.out.exists():
        raise FileExistsError(a.out)
    folds = load_folds(a.run)
    ids, sys, x_all = load_prefix(a.run)
    lookup = {s:i for i,s in enumerate(ids)}
    rows = [r for r in read_csv(a.targets) if r['target'] == a.only
            and r['future_valid'].lower() in ('true', '1')
            and r['system_id'] in folds and r['sample_id'] in lookup]
    rows.sort(key=lambda r:r['sample_id'])
    if not rows:
        raise ValueError('No eligible training rows')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate target rows')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prefix sample IDs')
    for r in rows:
        if sys[lookup[r['sample_id']]] != r['system_id']:
            raise ValueError('Prefix/target system identity mismatch')
    ix = np.array([lookup[r['sample_id']] for r in rows])
    systems = np.array([r['system_id'] for r in rows])
    groups = np.array([folds[s]['group_id'] for s in systems])
    f = np.array([folds[s]['outer_fold'] for s in systems])
    y = np.array([float(r['future_target']) for r in rows])
    if not np.isfinite(y).all() or not np.isfinite(x_all[ix]).all():
        raise ValueError('Nonfinite valid labels or prefix inputs')
    ch = CHANNEL[a.only]
    predictions, summary, manifests, selections, intervals, strata = [], [], [], [], [], []
    for ns in ([5, 10, 20] if a.mode == 'prefix' else [20]):
        chans = SAFE_CHANNELS if a.mode == 'prefix' else tuple(range(8))
        x = x_all[ix, :int(ns*10)+1, :][:, :, chans]
        z = summarize(x)
        p = np.full(len(y), np.nan)
        baseline = {'persistence': x_all[ix, int(ns*10), ch],
                    'prefix_mean': x_all[ix, :int(ns*10)+1, ch].mean(1)}
        # Scale is target RMSD variability at 15--20 ns; no target labels used.
        sig = x_all[ix, 150:201, 0].std(1)
        for k in sorted(set(f)):
            ev = np.flatnonzero(f == k)
            tr = np.flatnonzero(f != k)
            if a.mode == 'intervals':
                tr, ca, ev = partition(groups, f, k)
                # Only proper-training rows enter inner selection or normalisation.
                both = np.concatenate([ca, ev])
                qpred, alpha = ridge_cv_predict(z, y, systems, groups, tr, both, ALPHAS_EXTENDED, SEED+100*k)
                p[both] = qpred
                for role, idx in [('training',tr),('calibration',ca),('evaluation',ev)]:
                    for g in sorted(set(groups[idx])):
                        manifests.append({'outer_fold':int(k), 'group_id':g, 'role':role})
                # Compute scales independently of labels and temporary predictions.
                trsig = np.array([sig[tr[systems[tr] == s]].mean() for s in sorted(set(systems[tr]))])
                floor = max(.1 * float(np.median(trsig)), 1e-12)
                cn, cy, cp, cs = aggregate(ca, systems, y, p, sig)
                en, ey, ep, es = aggregate(ev, systems, y, p, sig)
                # Quartile edges and scale floor are frozen on proper training inputs.
                edges = np.quantile(trsig, [.25,.5,.75])
                variants = ['absolute', 'group_max'] + (['scaled_group_max'] if a.only == 'rmsd' else [])
                for variant in variants:
                    scaled = variant == 'scaled_group_max'
                    cscale = np.maximum(cs, floor) if scaled else np.ones(len(cn))
                    escale = np.maximum(es, floor) if scaled else np.ones(len(en))
                    scores = abs(cy-cp)/cscale
                    if variant != 'absolute':
                        cg = np.array([folds[s]['group_id'] for s in cn])
                        scores = np.array([max(scores[cg==g]) for g in sorted(set(cg))])
                    for miscoverage in [.10,.05]:
                        q = quantile(scores, miscoverage)
                        for j,s in enumerate(en):
                            width = q*escale[j]
                            intervals.append({'system_id':s,'group_id':folds[s]['group_id'],
                                'fold':int(k),'target':a.only,'variant':variant,'nominal':1-miscoverage,
                                'true':float(ey[j]),'predicted':float(ep[j]),
                                'lower':float(ep[j]-width),'upper':float(ep[j]+width),
                                'covered':int(abs(ey[j]-ep[j])<=width), 'width':float(2*width),
                                'prefix_variability_stratum':int(np.searchsorted(edges,es[j],side='right')+1),
                                'scale_floor':floor,'calibration_scores':len(scores)})
                # p[calibration] is only temporary; later folds must not reuse it.
                p[ca] = np.nan
            else:
                p[ev], alpha = ridge_cv_predict(z,y,systems,groups,tr,ev,ALPHAS_EXTENDED,SEED+100*k)
            selections.append({'prefix_ns':ns,'fold':int(k),'alpha':alpha,'features':z.shape[1]})
            for idx in ev:
                predictions.append({'sample_id':rows[idx]['sample_id'],'system_id':systems[idx],
                    'group_id':groups[idx],'fold':int(k),'prefix_ns':ns,'target':a.only,
                    'true':float(y[idx]),'predicted':float(p[idx]),
                    'persistence':float(baseline['persistence'][idx]),
                    'prefix_mean':float(baseline['prefix_mean'][idx])})
        # Rebuild out-of-fold point predictions from evaluation records only.
        # The training of fold k may overwrite earlier folds' temporary p entries.
        pr={r['sample_id']:r['predicted'] for r in predictions if r['prefix_ns']==ns}
        p=np.array([pr[r['sample_id']] for r in rows])
        sysnames = sorted(set(systems))
        sg = np.array([folds[s]['group_id'] for s in sysnames])
        errs={}
        for model, pred in {'ridge':p, **baseline}.items():
            sy=np.array([y[systems==s].mean() for s in sysnames])
            sp=np.array([pred[systems==s].mean() for s in sysnames])
            errs[model]=sp-sy
            summary.append({'prefix_ns':ns,'model':model,'unit':'aggregate_system',**metrics(sy,sp)})
            mae=np.array([abs(pred[systems==s]-y[systems==s]).mean() for s in sysnames])
            summary.append({'prefix_ns':ns,'model':model,'unit':'replica_errors_mean_per_system','mae':float(mae.mean())})
            for count in (1,2,3):
                m=np.array([sum(systems==s)==count for s in sysnames])
                if m.any():
                    summary.append({'prefix_ns':ns,'model':model,'unit':f'aggregate_{count}_valid_replicas',
                                    'n_systems':int(m.sum()),**metrics(sy[m],sp[m])})
        for control in baseline:
            d,lo,hi=group_bootstrap_delta(errs['ridge'],errs[control],sg,draws=a.draws)
            summary.append({'prefix_ns':ns,'model':f'ridge - {control}','unit':'aggregate_system',
                            'delta_mae':d,'ci_low':lo,'ci_high':hi})
    if intervals:
        for variant in sorted({r['variant'] for r in intervals}):
            for nominal in (.90,.95):
                rr=[r for r in intervals if r['variant']==variant and r['nominal']==nominal]
                for st in (0,1,2,3,4):
                    sub=rr if st==0 else [r for r in rr if r['prefix_variability_stratum']==st]
                    if sub:
                        group_coverage = ''
                        if st == 0:
                            group_coverage = float(np.mean([
                                all(r['covered'] for r in sub if r['group_id'] == g)
                                for g in sorted({r['group_id'] for r in sub})]))
                        strata.append({'variant':variant,'nominal':nominal,'stratum':st,
                            'n_systems':len(sub),'coverage':float(np.mean([r['covered'] for r in sub])),
                            'coverage_all_systems_in_group':group_coverage,
                            'mean_width':float(np.mean([r['width'] for r in sub]))})
    a.out.mkdir(parents=True)
    # Union keys allow different metric records to share a CSV schema.
    for name,data in [('replica_predictions',predictions),('metrics',summary),('partition_manifest',manifests),
                      ('selection',selections),('intervals',intervals),('coverage',strata)]:
        if data:
            keys=list(dict.fromkeys(k for r in data for k in r))
            write_csv(a.out/(name+'.csv'),[{k:r.get(k,'') for k in keys} for r in data])
    write_json(a.out/'provenance.json',{'mode':a.mode,'target':a.only,'run':str(a.run),
        'targets_sha256':sha(a.targets),'dataset_sha256':sha(a.run/'dataset.npz'),
        'folds_sha256':sha(a.run/'outer_folds.csv'),'n_systems':len(set(systems)),
        'n_replicas':len(y),'seed':SEED,'bootstrap_draws':a.draws,
        'selection_scope':'training cohort only','prefix_channels':list(SAFE_CHANNELS) if a.mode=='prefix' else list(range(8)),
        'coverage_note':'absolute scores are descriptive with dependent systems; group-max variants target new exchangeable groups, not conditional coverage for each protein; intervals predict valid-replica aggregate labels',
        'bootstrap_note':'paired point-error intervals condition on OOF fits; calibration and fit uncertainty not resampled'})
    print('REANALYSE=PASS',a.out)

if __name__=='__main__':
    main()
