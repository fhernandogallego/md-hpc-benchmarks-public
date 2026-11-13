#!/usr/bin/env python3
"""Prespecified recent-prefix controls and ridge ablations on training-cohort CV.

See PROTOCOLO_BASELINES_ABLACION.txt. All model/window choices are internal to
the outer training groups; fixed candidates and selected procedures are distinct.
"""
from __future__ import annotations
import argparse
import platform
from pathlib import Path
import numpy as np
from common import (ALPHAS_EXTENDED, SEED, FEATURES, load_folds, load_prefix,
                    read_csv, write_csv, write_json, sha, summarize, inner_pairs,
                    system_errors, ridge_cv_predict, metrics, group_bootstrap_delta)
from reanalyse import CHANNEL, SAFE_CHANNELS
from verify_rebuilt_rmsd import verify

WINDOWS_NS = (.5, 1., 2., 5., 10.)
HALFLIVES_NS = (.25, .5, 1., 2., 5.)
SUMMARY_BLOCKS = {'last': (4,), 'mean_std': (0, 1),
                  'mean_std_last': (0, 1, 4),
                  'mean_std_last_slope': (0, 1, 4, 5),
                  'full': (0, 1, 2, 3, 4, 5)}

def baseline_candidates(values, time_ns):
    """values contains only the current prefix; endpoints of a tail window are included."""
    if values.ndim != 2 or values.shape[1] != len(time_ns):
        raise ValueError('Expected one prefix time series per sample')
    if not np.isfinite(values).all() or not np.isfinite(time_ns).all() or np.any(np.diff(time_ns) <= 0):
        raise ValueError('Invalid prefix values or time grid')
    duration = time_ns[-1] - time_ns[0]
    candidates = {'persistence': values[:, -1].copy(), 'prefix_mean': values.mean(1)}
    for window in WINDOWS_NS:
        if window <= duration + 1e-9:
            mask = time_ns >= time_ns[-1] - window - 1e-9
            candidates[f'trailing_mean_{window:g}ns'] = values[:, mask].mean(1)
    for half in HALFLIVES_NS:
        weights = np.exp2((time_ns - time_ns[-1]) / half)
        candidates[f'ewma_half_{half:g}ns'] = values @ (weights / weights.sum())
    return candidates

def select_candidate(candidates, y, systems, groups, fit_idx, seed):
    """Only inner held-out labels from fit_idx enter the score or tie-breaking."""
    if not candidates:
        raise ValueError('No baseline candidates')
    scores = {}
    pairs = inner_pairs(fit_idx, groups, seed)
    for name, pred in candidates.items():
        errors = []
        for _, ev in pairs:
            if not len(ev):
                raise ValueError('An inner baseline fold is empty')
            errors.extend(system_errors(y[ev], pred[ev], systems[ev]))
        scores[name] = float(np.mean(errors))
    # Dict insertion order resolves exact ties according to the written protocol.
    return min(scores, key=scores.get), scores

def ablation_features(prefix, target_channel, horizon_ns):
    """Summarise only channels that can be constructed at the requested horizon."""
    if horizon_ns not in (5, 10, 20):
        raise ValueError('Unsupported horizon')
    if horizon_ns < 20 and target_channel in (2, 3, 4):
        raise ValueError('A core target defined at 20 ns is unavailable at shorter horizons')
    if prefix.shape[1] != horizon_ns * 10 + 1 or prefix.shape[2] != 8:
        raise ValueError('Expected original eight-channel prefix at 0.1 ns cadence')
    channels = tuple(range(8)) if horizon_ns == 20 else SAFE_CHANNELS
    result, descriptions = {}, []
    for scope, selected in [('target', (target_channel,)), ('all', channels)]:
        if scope == 'target':
            scoped = prefix[:, :, target_channel:target_channel + 1]
        elif selected == tuple(range(8)):
            scoped = prefix
        else:
            scoped = prefix[:, :, selected]
        summary = summarize(scoped)
        p = len(selected)
        for family, blocks in SUMMARY_BLOCKS.items():
            name = f'ridge_{scope}_{family}'
            columns = [b * p + c for b in blocks for c in range(p)]
            result[name] = summary if family == 'full' else summary[:, columns]
            descriptions.append({'model': name, 'features': len(columns),
                'channels': '|'.join(FEATURES[c] for c in selected), 'summaries': family})
    # At short horizons this duplicates ridge_all_full, so do not create a second model.
    if horizon_ns == 20:
        name = 'ridge_without_core_full'
        result[name] = summarize(prefix[:, :, SAFE_CHANNELS])
        descriptions.append({'model': name, 'features': 30,
            'channels': '|'.join(FEATURES[c] for c in SAFE_CHANNELS), 'summaries': 'full'})
    return result, descriptions

def contrasts_for(models):
    pairs = [('ridge_all_full', m) for m in
             ('simple_selected', 'trailing_mean_selected', 'ewma_selected', 'persistence', 'prefix_mean')]
    pairs += [('ridge_all_full', 'ridge_target_full'), ('ridge_all_full', 'ridge_without_core_full')]
    for scope in ('target', 'all'):
        pairs += [(f'ridge_{scope}_mean_std_last', f'ridge_{scope}_mean_std'),
                  (f'ridge_{scope}_mean_std_last_slope', f'ridge_{scope}_mean_std_last'),
                  (f'ridge_{scope}_full', f'ridge_{scope}_mean_std_last_slope')]
    return [(x, y) for x, y in pairs if x in models and y in models]

def write_records(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        write_csv(path, [{k: r.get(k, '') for k in keys} for r in rows])

def check_reference(reference_path, names, sy, sp, folds):
    rows = read_csv(reference_path)
    reference = {r['system_id']: r for r in rows}
    if len(reference) != len(rows) or set(reference) != set(names):
        raise ValueError('The RMSD reference and new full model evaluate different systems')
    for s in names:
        if (reference[s]['group_id'], int(reference[s]['fold'])) != (folds[s]['group_id'], folds[s]['outer_fold']):
            raise ValueError('The original and new RMSD prediction partitions differ')
    label_error = float(np.max(abs(sy - np.array([float(reference[s]['true']) for s in names]))))
    prediction_error = float(np.max(abs(sp - np.array([float(reference[s]['predicted']) for s in names]))))
    if max(label_error, prediction_error) > 1e-8:
        raise ValueError(f'Full 48-feature RMSD model failed reference check: labels={label_error}, predictions={prediction_error}')
    return {'reference_sha256': sha(reference_path), 'n_systems': len(names),
            'max_absolute_label_difference': label_error, 'max_absolute_prediction_difference': prediction_error}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--targets', type=Path, required=True)
    ap.add_argument('--only', choices=CHANNEL, default='rmsd')
    ap.add_argument('--prefix-ns', type=int, nargs='+', choices=(5, 10, 20), default=[20])
    ap.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1] / 'PROTOCOLO_BASELINES_ABLACION.txt')
    ap.add_argument('--reference-predictions', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'sources' / 'reference_rmsd20.csv')
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if not a.protocol.is_file():
        ap.error('The written baseline/ablation protocol must accompany the package')
    if a.draws <= 0 or len(set(a.prefix_ns)) != len(a.prefix_ns):
        ap.error('Positive bootstrap draws and unique horizons are required')
    if a.only == 'core_rmsd_q30' and any(t < 20 for t in a.prefix_ns):
        ap.error('Core RMSD is only supported at 20 ns, after target reconstruction')
    if a.only == 'rmsd' and 20 in a.prefix_ns and not a.reference_predictions.is_file():
        ap.error('The original 20 ns RMSD OOF prediction reference is required')
    reference = verify(a.run, a.targets) if a.only == 'rmsd' else None
    folds = load_folds(a.run)
    ids, prefix_systems, raw = load_prefix(a.run)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs in prefix input')
    lookup = {s: i for i, s in enumerate(ids)}
    eligible = [r for r in read_csv(a.targets) if r['target'] == a.only
                and r['system_id'] in folds and r['future_valid'].lower() in ('true', '1')]
    rows = sorted(eligible, key=lambda r: r['sample_id'])
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('No valid training rows, or duplicate target rows')
    for r in rows:
        if r['sample_id'] not in lookup:
            raise ValueError(f"Valid training label lacks prefix input: {r['sample_id']}")
        if prefix_systems[lookup[r['sample_id']]] != r['system_id']:
            raise ValueError('Prefix/target system identity mismatch')
    idx = np.array([lookup[r['sample_id']] for r in rows])
    raw = raw[idx]
    systems = np.array([r['system_id'] for r in rows])
    groups = np.array([folds[s]['group_id'] for s in systems])
    outer = np.array([folds[s]['outer_fold'] for s in systems])
    y = np.array([float(r['future_target']) for r in rows])
    if not np.isfinite(y).all() or not np.isfinite(raw).all():
        raise ValueError('Nonfinite valid labels or prefix data')
    if set(outer) != set(range(5)):
        raise ValueError('Expected all five original outer folds')
    names = sorted(set(systems))
    system_idx = [np.flatnonzero(systems == s) for s in names]
    sy = np.array([y[b].mean() for b in system_idx])
    sg = np.array([folds[s]['group_id'] for s in names])
    nrep = np.array([len(b) for b in system_idx])
    records = {k: [] for k in ('replica_predictions', 'system_predictions', 'metrics',
               'contrasts', 'selection', 'baseline_inner_scores', 'feature_manifest', 'outer_fold_manifest')}
    prediction_reference = None
    for s, count in zip(names, nrep):
        records['outer_fold_manifest'].append({'system_id': s, 'group_id': folds[s]['group_id'],
            'fold': folds[s]['outer_fold'], 'valid_replicas': int(count)})
    for horizon in a.prefix_ns:
        x = raw[:, :horizon * 10 + 1]
        candidates = baseline_candidates(x[:, :, CHANNEL[a.only]], np.arange(x.shape[1]) / 10.)
        features, descriptions = ablation_features(x, CHANNEL[a.only], horizon)
        records['feature_manifest'].extend({'prefix_ns': horizon, **r} for r in descriptions)
        pred = {**{m: p.copy() for m, p in candidates.items()},
                **{m: np.full(len(y), np.nan) for m in
                   (*features, 'simple_selected', 'trailing_mean_selected', 'ewma_selected')}}
        for k in range(5):
            tr, ev = np.flatnonzero(outer != k), np.flatnonzero(outer == k)
            if set(groups[tr]) & set(groups[ev]):
                raise ValueError('Outer sequence groups overlap')
            seed = SEED + 100 * k
            for family, subset in (
                ('simple_selected', candidates),
                ('trailing_mean_selected', {m: p for m, p in candidates.items() if m.startswith('trailing_mean_')}),
                ('ewma_selected', {m: p for m, p in candidates.items() if m.startswith('ewma_half_')})):
                chosen, scores = select_candidate(subset, y, systems, groups, tr, seed)
                pred[family][ev] = candidates[chosen][ev]
                records['selection'].append({'prefix_ns': horizon, 'fold': k, 'model': family,
                    'selected_candidate': chosen, 'inner_mae': scores[chosen]})
                for candidate, score in scores.items():
                    records['baseline_inner_scores'].append({'prefix_ns': horizon, 'fold': k,
                        'family': family, 'candidate': candidate, 'inner_mae': score})
            for model, z in features.items():
                pred[model][ev], alpha = ridge_cv_predict(z, y, systems, groups, tr, ev, ALPHAS_EXTENDED, seed)
                records['selection'].append({'prefix_ns': horizon, 'fold': k, 'model': model,
                    'alpha': alpha, 'features': z.shape[1]})
        errors, replica_mae = {}, {}
        for model, p in pred.items():
            if not np.isfinite(p).all():
                raise ValueError(f'Incomplete predictions for {model}')
            sp = np.array([p[b].mean() for b in system_idx])
            if a.only == 'rmsd' and horizon == 20 and model == 'ridge_all_full':
                prediction_reference = check_reference(a.reference_predictions, names, sy, sp, folds)
                print('FULL_RMSD_PREDICTION_CHECK=PASS', flush=True)
            errors[model] = sp - sy
            replica_mae[model] = np.array([abs(p[b] - y[b]).mean() for b in system_idx])
            records['metrics'].append({'prefix_ns': horizon, 'model': model,
                'unit': 'aggregate_system', 'n_systems': len(names), **metrics(sy, sp)})
            records['metrics'].append({'prefix_ns': horizon, 'model': model,
                'unit': 'mean_replica_absolute_error_per_system', 'n_systems': len(names),
                'mae': float(replica_mae[model].mean())})
            for count in (1, 2, 3):
                m = nrep == count
                if m.any():
                    records['metrics'].append({'prefix_ns': horizon, 'model': model,
                        'unit': f'aggregate_{count}_valid_replicas', 'n_systems': int(m.sum()), **metrics(sy[m], sp[m])})
            for s, truth, forecast, count in zip(names, sy, sp, nrep):
                records['system_predictions'].append({'prefix_ns': horizon, 'model': model,
                    'system_id': s, 'group_id': folds[s]['group_id'], 'fold': folds[s]['outer_fold'],
                    'valid_replicas': int(count), 'true': float(truth), 'predicted': float(forecast)})
            for j, row in enumerate(rows):
                records['replica_predictions'].append({'prefix_ns': horizon, 'model': model,
                    'sample_id': row['sample_id'], 'system_id': systems[j], 'group_id': groups[j],
                    'fold': int(outer[j]), 'true': float(y[j]), 'predicted': float(p[j])})
        for first, second in contrasts_for(pred):
            for unit, array in [('aggregate_system', errors),
                                ('mean_replica_absolute_error_per_system', replica_mae)]:
                d, lo, hi = group_bootstrap_delta(array[first], array[second], sg, draws=a.draws)
                records['contrasts'].append({'prefix_ns': horizon, 'contrast': f'{first} - {second}',
                    'unit': unit, 'primary': first == 'ridge_all_full' and second == 'simple_selected' and unit == 'aggregate_system',
                    'delta_mae': d, 'ci_low': lo, 'ci_high': hi,
                    'relative_change': d / float(abs(array[second]).mean()) if np.any(array[second]) else ''})
        print(f'Temporal controls: {a.only}, {horizon} ns, {len(pred)} procedures, {len(names)} systems', flush=True)
    a.out.mkdir(parents=True)
    for name, data in records.items():
        write_records(a.out / f'{name}.csv', data)
    write_json(a.out / 'provenance.json', {'target': a.only, 'horizons_ns': a.prefix_ns,
        'protocol_sha256': sha(a.protocol), 'script_sha256': sha(Path(__file__)),
        'dataset_sha256': sha(a.run / 'dataset.npz'), 'targets_sha256': sha(a.targets),
        'folds_sha256': sha(a.run / 'outer_folds.csv'), 'rmsd_reference_check': reference,
        'full_rmsd_prediction_check': prediction_reference,
        'windows_ns': WINDOWS_NS, 'halflives_ns': HALFLIVES_NS, 'summary_blocks': SUMMARY_BLOCKS,
        'alpha_grid': ALPHAS_EXTENDED, 'n_systems': len(names), 'n_replicas': len(y),
        'bootstrap_draws': a.draws, 'bootstrap_seed': 20261016, 'python': platform.python_version(),
        'numpy': np.__version__, 'scope': 'Training-cohort grouped nested CV only; descriptive paired intervals condition on OOF fits',
        'caution': 'Short horizons exclude all 20 ns core-derived channels; valid-label selection is retrospective; no confirmatory multiple-comparison claims'})
    print('TEMPORAL_CONTROLS=PASS', a.out, flush=True)

if __name__ == '__main__':
    main()
