#!/usr/bin/env python3
"""Comprobaciones de estructura sobre las salidas de la fase 2 de CILS v3 (solo lectura; sin recalcular modelos).
Uso: python3 validar_fase2.py <carpeta con intervals_rmsd/ prefix_rmsd/ temporal_rmsd/ ...>"""
import csv, json, sys, math
from collections import defaultdict
from pathlib import Path

def rd(p): return list(csv.DictReader(open(p, newline='', encoding='utf-8')))
bad = []
def chk(c, m):
    if not c: bad.append(m)
    return c

def intervals(d, target):
    prov = json.loads((d / 'provenance.json').read_text())
    chk(prov['selection_scope'] == 'training cohort only', 'ámbito distinto de solo entrenamiento')
    pm = rd(d / 'partition_manifest.csv'); roles = defaultdict(lambda: defaultdict(set))
    for r in pm: roles[r['outer_fold']][r['role']].add(r['group_id'])
    seen_eval = defaultdict(int)
    for f, rr in sorted(roles.items()):
        names = set(rr)
        chk(names == {'training', 'calibration', 'evaluation'}, 'fold %s: roles %s' % (f, sorted(names)))
        chk(not (rr['training'] & rr['calibration']) and not (rr['training'] & rr['evaluation']) and not (rr['calibration'] & rr['evaluation']),
            'fold %s: grupos solapados entre roles' % f)
        for g in rr['evaluation']: seen_eval[g] += 1
    chk(all(v == 1 for v in seen_eval.values()), 'algún grupo se evalúa más de una vez')
    # cada grupo evaluado en un solo fold, y nunca como training/calibration en ese fold ya comprobado
    iv = rd(d / 'intervals.csv'); cov = rd(d / 'coverage.csv')
    for r in iv:  # el grupo de cada sistema se evalúa en su fold
        chk(r['group_id'] in roles[r['fold']]['evaluation'], 'sistema %s fuera del rol de evaluación de su fold' % r['system_id'])
    bad_rows = sum(1 for r in iv if (float(r['lower']) <= float(r['true']) <= float(r['upper'])) != (r['covered'] == '1'))
    chk(bad_rows == 0, '%d filas con "covered" incoherente con los límites' % bad_rows)
    # recomputar cobertura global y por estrato
    agg = defaultdict(lambda: [0, 0, 0.0])
    for r in iv:
        for key in ((r['variant'], r['nominal'], '0'), (r['variant'], r['nominal'], r['prefix_variability_stratum'])):
            a = agg[key]; a[0] += 1; a[1] += int(r['covered']); a[2] += float(r['width'])
    worst = 0.0
    for r in cov:
        key = (r['variant'], r['nominal'], r['stratum']) if (r['variant'], r['nominal'], r['stratum']) in agg else (r['variant'], str(float(r['nominal'])), r['stratum'])
        n, c, w = agg[key]
        chk(n == int(r['n_systems']), 'recuento distinto en %s' % (key,))
        worst = max(worst, abs(c / n - float(r['coverage'])), abs(w / n - float(r['mean_width'])))
    chk(worst < 1e-9, 'cobertura/anchura recomputadas difieren (%.3g)' % worst)
    levels = sorted({r['nominal'] for r in cov}); strata = sorted({r['stratum'] for r in cov})
    chk(levels == ['0.9', '0.95'] and strata == ['0', '1', '2', '3', '4'], 'niveles/estratos inesperados %s %s' % (levels, strata))
    return {'sistemas': len({r['system_id'] for r in iv}), 'filas_intervalos': len(iv), 'folds': len(roles), 'max_dif_cobertura_recomputada': worst,
            'target': prov['target']}

def prefix(d):
    prov = json.loads((d / 'provenance.json').read_text())
    chk(prov['prefix_channels'] == [0, 1, 5, 6, 7], 'canales de prefijo inesperados %s' % prov['prefix_channels'])
    horizons = sorted({r['prefix_ns'] for r in rd(d / 'metrics.csv')})
    chk(horizons == ['10', '20', '5'] or horizons == ['5', '10', '20'], 'horizontes %s' % horizons)
    return {'canales': prov['prefix_channels'], 'horizontes': horizons}

def temporal(d):
    prov = json.loads((d / 'provenance.json').read_text())
    chk(prov['rmsd_reference_check']['valid_mask_identical'] is True and prov['rmsd_reference_check']['maximum_absolute_label_difference_A'] == 0.0, 'referencia RMSD no coincide')
    chk(prov['full_rmsd_prediction_check']['max_absolute_prediction_difference'] < 1e-8, 'predicción del ridge completo no reproduce la referencia')
    chk(prov['horizons_ns'] == [5, 10, 20], 'horizontes %s' % prov['horizons_ns'])
    return {'dif_pred_ridge_completo': prov['full_rmsd_prediction_check']['max_absolute_prediction_difference'], 'modelos_por_horizonte': {h: len({r['model'] for r in rd(d / 'metrics.csv') if r['prefix_ns'] == str(h)}) for h in (5, 10, 20)}}

if __name__ == '__main__':
    root = Path(sys.argv[1]); out = {}
    if (root / 'intervals_rmsd').is_dir(): out['intervals_rmsd'] = intervals(root / 'intervals_rmsd', 'rmsd')
    if (root / 'intervals_rg').is_dir(): out['intervals_rg'] = intervals(root / 'intervals_rg', 'rg')
    if (root / 'prefix_rmsd').is_dir(): out['prefix_rmsd'] = prefix(root / 'prefix_rmsd')
    if (root / 'temporal_rmsd').is_dir(): out['temporal_rmsd'] = temporal(root / 'temporal_rmsd')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print('PROBLEMAS=%d' % len(bad)); [print(' ', b) for b in bad[:20]]
    print('VALIDACION_FASE2=%s' % ('PASS' if not bad else 'REVISAR'))
