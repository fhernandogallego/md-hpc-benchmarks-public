#!/usr/bin/env python3
"""Resumen de recuentos y exclusiones de los objetivos consolidados (bloque A de CILS v3).
Solo lectura de targets_{include,exclude}_constants.csv y sus resúmenes; no recalcula objetivos ni abre validation/test.
Criterio de validez del constructor (build_targets.py): duración de la ventana futura >= 20 ns y neff >= 10.
Uso: python3 resumen_objetivos_consolidados.py <carpeta merged> <carpeta de salida (no debe existir)>"""
import csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

NAMES = ('core_rmsd_q30', 'helix', 'rg', 'rmsd', 'sasa', 'strand')
EPS = 1e-8

def truth(x):
    return str(x).strip().lower() in ('true', '1')

def reason(r):
    if truth(r['constant_window']):
        return 'ventana constante'
    dur, neff = float(r['future_duration_ns']), float(r['future_neff'])
    if dur == 0.0 and neff == 0.0:
        return 'sin ventana futura'
    short, low = dur < 20.0 - EPS, neff < 10.0 - EPS
    return 'duración<20 ns y neff<10' if short and low else 'duración<20 ns' if short else 'neff<10' if low else 'otro'

def main(merged, out):
    merged, out = Path(merged), Path(out)
    if out.exists():
        raise SystemExit('La salida ya existe: %s' % out)
    out.mkdir(parents=True)
    rows_out, problems = [], []
    for policy in ('include', 'exclude'):
        data = list(csv.DictReader(open(merged / ('targets_%s_constants.csv' % policy), newline='', encoding='utf-8')))
        summary = json.loads((merged / ('targets_%s_constants.summary.json' % policy)).read_text())
        for t in NAMES:
            sel = [r for r in data if r['target'] == t]
            valid = [r for r in sel if truth(r['future_valid'])]
            invalid = [r for r in sel if not truth(r['future_valid'])]
            systems = {r['system_id'] for r in sel}
            vsys = {r['system_id'] for r in valid}
            why = Counter(reason(r) for r in invalid)
            for r in valid:
                if not (math.isfinite(float(r['future_target'])) and float(r['future_duration_ns']) >= 20.0 - EPS and float(r['future_neff']) >= 10.0 - EPS):
                    problems.append('%s %s: réplica "válida" que incumple el criterio: %s' % (policy, t, r['sample_id']))
            ref = summary['per_target'][t]
            if (ref['replicas'], ref['valid_replicas'], ref['valid_systems']) != (len(sel), len(valid), len(vsys)):
                problems.append('%s %s: el resumen del consolidado no coincide con las filas' % (policy, t))
            reps_by_sys = Counter(r['system_id'] for r in valid)
            rows_out.append({'politica': policy, 'objetivo': t, 'sistemas': len(systems), 'replicas': len(sel),
                             'replicas_validas': len(valid), 'replicas_invalidas': len(invalid), 'sistemas_con_valida': len(vsys),
                             'sistemas_con_1_valida': sum(1 for v in reps_by_sys.values() if v == 1),
                             'sistemas_con_2_validas': sum(1 for v in reps_by_sys.values() if v == 2),
                             'sistemas_con_3_validas': sum(1 for v in reps_by_sys.values() if v == 3),
                             **{'inval_' + k: why.get(k, 0) for k in ('ventana constante', 'sin ventana futura', 'duración<20 ns', 'neff<10', 'duración<20 ns y neff<10', 'otro')}})
    with open(out / 'recuentos_objetivos_consolidados.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    json.dump({'filas': rows_out, 'problemas': problems}, open(out / 'recuentos_objetivos_consolidados.json', 'w'), ensure_ascii=False, indent=1)
    for r in rows_out:
        print('%-7s %-14s sist=%d réplicas=%d válidas=%d inválidas=%d (%s)' % (r['politica'], r['objetivo'], r['sistemas'], r['replicas'], r['replicas_validas'], r['replicas_invalidas'],
              ', '.join('%s=%d' % (k[6:], v) for k, v in r.items() if k.startswith('inval_') and v)))
    print('PROBLEMAS=%d' % len(problems)); [print(' ', p) for p in problems[:20]]
    print('RESUMEN_OBJETIVOS=%s' % ('PASS' if not problems else 'REVISAR'))

if __name__ == '__main__':
    main(*sys.argv[1:3])
