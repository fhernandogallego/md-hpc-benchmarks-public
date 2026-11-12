#!/usr/bin/env python3
"""Auditoria EARLYSTOP: codigo, validez de train y manifiestos. Sin test."""

import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

W = Path('/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks')
H = Path('/home/uva_dma_2/uva_dma_2_1')
RUN = W / 'results/paper2_automation/auto_20261030_091302'
PILOT = W / 'results/pid_atlas_poc/full_v2_3262025'

OUT = W / 'maintenance/earlystop' / (
    'auditoria_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S')
)

REPORT = []


def log(s=''):
    print(s, flush=True)
    REPORT.append(str(s))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csvread(p, required=()):
    with p.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        if not set(required).issubset(columns):
            raise RuntimeError(
                'Columnas ausentes en ' + str(p) + ': ' + str(required)
            )
        return list(reader)


def save(name, rows, columns):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction='ignore'
        )
        writer.writeheader()
        writer.writerows(rows)


def truth(value):
    value = str(value).strip().lower()
    if value in ('true', '1'):
        return True
    if value in ('false', '0'):
        return False
    return None


def git(repo, args):
    if not (repo / '.git').exists():
        return 'NO_REPO_GIT'
    try:
        result = subprocess.run(
            ['git', '-C', str(repo)] + args,
            capture_output=True,
            text=True,
            timeout=20
        )
        return (
            result.stdout.strip()
            if result.returncode == 0 else 'NO_CONFIRMADO'
        )
    except (OSError, subprocess.TimeoutExpired):
        return 'NO_CONFIRMADO'


def methods(root, label):
    log('\n[1] ' + label + ' - DEFINICIONES RMSD/Rg')

    if not root.is_dir():
        log('SIN ACCESO: ' + str(root))
        return

    folders = [
        root / 'pipeline_code',
        root / 'tool',
        root / 'code',
        root / 'src'
    ]

    if label == 'PILOTO_500':
        folders.append(root)

    seen = set()
    manifest_file = root / 'pipeline_manifest.json'
    recorded = (
        json.loads(manifest_file.read_text())
        if manifest_file.is_file() else {}
    )

    for folder in folders:
        if not folder.is_dir():
            continue

        for base, dirs, files in os.walk(folder):
            dirs[:] = [
                x for x in dirs
                if x not in (
                    '.git', 'raw', 'unpacked', 'results',
                    'test', 'tests', 'training', 'venv',
                    '.venv', 'data', 'models', 'runs', '__pycache__'
                )
            ]

            depth = len(Path(base).relative_to(folder).parts)
            if depth >= (3 if label == 'PILOTO_500' else 6):
                dirs[:] = []

            for name in files:
                p = Path(base) / name

                if p in seen or not name.endswith('.py'):
                    continue

                if p.stat().st_size > 1000000:
                    continue

                seen.add(p)
                lines = p.read_text(errors='replace').splitlines()

                hits = [
                    i for i, line in enumerate(lines)
                    if re.search(
                        r'rms\.RMSD|radius_of_gyration\('
                        r'|select_atoms\(|ref_frame\s*=|weights\s*=',
                        line
                    )
                ]

                if not hits:
                    continue

                log('CODIGO %s SHA256=%s' % (p, sha(p)))

                indices = sorted(set(
                    j
                    for k in hits[:12]
                    for j in range(
                        max(0, k - 2),
                        min(len(lines), k + 4)
                    )
                ))

                for i in indices[:70]:
                    log('  %04d: %s' % (i + 1, lines[i][:200]))

                try:
                    key = str(p.relative_to(root / 'pipeline_code'))
                    if key in recorded:
                        state = (
                            'SHA256_OK'
                            if recorded[key] == sha(p)
                            else 'SHA256_DISTINTO'
                        )
                        log('  pipeline_manifest: ' + state)
                except ValueError:
                    pass

    for name in ('config.json', 'controller_job.json'):
        p = root / name

        if p.is_file():
            obj = json.loads(p.read_text())
            allowed = (
                'work', 'project', 'run_id', 'tool_code',
                'account', 'partition', 'python', 'id'
            )
            safe = {k: obj[k] for k in allowed if k in obj}
            log('%s SHA256=%s campos=%s' % (p, sha(p), safe))


def counts():
    log('\n[2] RECUENTOS POR OBSERVABLE - SOLO TRAIN')

    cohort = RUN / 'cohort'
    assignment_file = cohort / 'system_assignment.csv'
    inventory_file = cohort / 'source_inventory.csv'
    future_file = cohort / 'train_future_rmsd_rows.csv'

    for p in (assignment_file, inventory_file, future_file):
        if not p.is_file():
            log('FALTA: ' + str(p))
            return
        log('FUENTE %s SHA256=%s' % (p, sha(p)))

    assignments = csvread(
        assignment_file,
        ['system_id', 'split', 'group_id']
    )

    train = {
        r['system_id'] for r in assignments
        if r['split'].strip().lower() == 'train'
    }

    groupmap = {
        r['system_id']: r['group_id']
        for r in assignments
    }

    inventory = {
        r['system']: r
        for r in csvread(
            inventory_file,
            ['system', 'master_targets_csv']
        )
    }

    future_rows = csvread(
        future_file,
        ['system_id', 'replica', 'future_valid']
    )

    valid_future = {
        (r['system_id'], r['replica'])
        for r in future_rows
        if (
            r['system_id'] in train
            and truth(r['future_valid']) is True
        )
    }

    labeled = {system for system, replica in valid_future}

    log(
        'TOTAL: %d sistemas / %d grupos'
        % (len(groupmap), len(set(groupmap.values())))
    )

    log(
        'TRAIN: %d sistemas / %d grupos / %d replicas posibles'
        % (
            len(train),
            len({groupmap[s] for s in train}),
            3 * len(train)
        )
    )

    log(
        'RMSD FUTURO: %d sistemas etiquetados / %d replicas validas'
        % (len(labeled), len(valid_future))
    )

    flags = Counter(
        (
            str(truth(r.get('original_valid'))),
            str(truth(r.get('future_valid'))),
            str(truth(r.get('common_valid')))
        )
        for r in future_rows
        if r['system_id'] in train
    )

    save(
        'estados_rmsd_futuro.csv',
        [
            {
                'original_valid': k[0],
                'future_valid': k[1],
                'common_valid': k[2],
                'n_replicas': n
            }
            for k, n in sorted(flags.items())
        ],
        [
            'original_valid', 'future_valid',
            'common_valid', 'n_replicas'
        ]
    )

    if (len(train), len(labeled), len(valid_future)) != (
        924, 757, 1243
    ):
        log(
            'ADVERTENCIA: recuentos diferentes a la '
            'instantanea ATLAS942 publicada.'
        )

    # Exclusivamente targets de sistemas del split train.
    # No leer master_targets de validation ni test.

    observations = {}
    faults = []
    masters_read = 0

    for system in sorted(train):
        path_text = inventory.get(
            system, {}
        ).get('master_targets_csv', '')

        p = Path(path_text) if path_text else None

        if not p or not p.is_file():
            faults.append({
                'system': system,
                'problem': 'SIN_MASTER_CSV'
            })
            continue

        try:
            rows = csvread(
                p,
                [
                    'system', 'replica', 'target_name',
                    'target_valid', 'target_status'
                ]
            )

            for r in rows:
                if r['system'] != system:
                    faults.append({
                        'system': system,
                        'problem': 'ID_NO_COINCIDE'
                    })
                    continue

                key = (
                    system,
                    r['replica'].strip(),
                    r['target_name'].lower().strip()
                )

                if key in observations:
                    faults.append({
                        'system': system,
                        'problem': 'FILA_DUPLICADA'
                    })
                    continue

                observations[key] = (
                    truth(r['target_valid']),
                    r['target_status'].strip() or 'SIN_STATUS',
                    str(p)
                )

            masters_read += 1

        except (OSError, ValueError, csv.Error, RuntimeError) as exc:
            faults.append({
                'system': system,
                'problem': 'ERROR_' + str(exc)[:150]
            })

    save(
        'incidencias_master.csv',
        faults,
        ['system', 'problem']
    )

    observables = sorted({
        key[2] for key in observations
    })

    log(
        'MASTER DISPONIBLES: %d/%d | OBSERVABLES: %s'
        % (masters_read, len(train), ', '.join(observables))
    )

    if not any('core' in x for x in observables):
        log(
            'CORE RMSD q30: no figura en master. '
            'Requiere inventario/tabla de validez independiente.'
        )

    summaries = []
    reasons = Counter()
    details = []
    unverified_systems = []

    scopes = (
        ('train_924', train),
        ('rmsd_labeled_757', labeled)
    )

    for scope, universe in scopes:
        for observable in observables:
            n = Counter()

            for system in sorted(universe):
                states = []

                for replica in ('1', '2', '3'):
                    valid, status, source = observations.get(
                        (system, replica, observable),
                        (None, 'SIN_FILA_O_ARCHIVO', '')
                    )

                    states.append(valid)

                    if valid is True:
                        n['valid'] += 1
                    elif valid is False:
                        n['invalid'] += 1
                    else:
                        n['missing'] += 1

                    if valid is not True:
                        reasons[(scope, observable, status)] += 1

                        details.append({
                            'scope': scope,
                            'system': system,
                            'replica': replica,
                            'observable': observable,
                            'valid': str(valid),
                            'reason': status,
                            'source': source
                        })

                if any(v is True for v in states):
                    n['systems_valid'] += 1

                elif all(v is False for v in states):
                    n['systems_excluded_verified'] += 1

                else:
                    n['systems_unverifiable'] += 1
                    unverified_systems.append({
                        'scope': scope,
                        'system': system,
                        'observable': observable
                    })

            summaries.append({
                'scope': scope,
                'observable': observable,
                'systems_input': len(universe),
                'replicas_input_slots': 3 * len(universe),
                'systems_retained': n['systems_valid'],
                'replicas_retained': n['valid'],
                'replicas_excluded_verified': n['invalid'],
                'replicas_unverifiable': n['missing'],
                'systems_excluded_verified':
                    n['systems_excluded_verified'],
                'systems_unverifiable': n['systems_unverifiable']
            })

    columns = [
        'scope', 'observable',
        'systems_input', 'replicas_input_slots',
        'systems_retained', 'replicas_retained',
        'replicas_excluded_verified', 'replicas_unverifiable',
        'systems_excluded_verified', 'systems_unverifiable'
    ]

    save('recuentos_por_observable.csv', summaries, columns)

    save(
        'motivos_por_observable.csv',
        [
            {
                'scope': k[0],
                'observable': k[1],
                'reason': k[2],
                'n_replicas': value
            }
            for k, value in sorted(reasons.items())
        ],
        ['scope', 'observable', 'reason', 'n_replicas']
    )

    save(
        'detalle_no_retenidos.csv',
        details,
        [
            'scope', 'system', 'replica',
            'observable', 'valid', 'reason', 'source'
        ]
    )

    save(
        'sistemas_sin_datos_completos.csv',
        unverified_systems,
        ['scope', 'system', 'observable']
    )

    for r in summaries:
        log(
            '%s / %s: entrada=%s sistemas, %s replicas; '
            'retenidos=%s sistemas, %s replicas; '
            'excluidos confirmados=%s sistemas, %s replicas; '
            'no verificables=%s sistemas, %s replicas'
            % (
                r['scope'], r['observable'],
                r['systems_input'], r['replicas_input_slots'],
                r['systems_retained'], r['replicas_retained'],
                r['systems_excluded_verified'],
                r['replicas_excluded_verified'],
                r['systems_unverifiable'],
                r['replicas_unverifiable']
            )
        )

    # Metadatos de una replica de train.
    # No se abre ninguna serie ni etiqueta de test.

    for system in sorted(train):
        basic = inventory.get(system, {}).get(
            'basic_R1_csv', ''
        )

        p = Path(basic).with_suffix('.json') if basic else None

        if p and p.is_file():
            obj = json.loads(p.read_text())

            log(
                'METADATA BASICA %s SHA256=%s '
                'canonical_rmsd=%s canonical_rg=%s'
                % (
                    p, sha(p),
                    obj.get('canonical_rmsd'),
                    obj.get('canonical_rg')
                )
            )
            break


def splits():
    log('\n[3] MANIFIESTOS - SOLO ASIGNACIONES DE PARTICION')

    roots = [
        RUN / 'cohort',
        W / 'results/paper2_hito0',
        W / 'results/earlystop',
        W / 'releases',
        W / 'results/pid_atlas_poc',
        H / 'paper2-auto-v3',
        H / 'paper2-atlas942-publication'
    ]

    assignment = RUN / 'cohort/system_assignment.csv'
    atlas_groups = {}

    if assignment.is_file():
        atlas_groups = {
            r['system_id']: r['group_id']
            for r in csvread(
                assignment, ['system_id', 'group_id']
            )
        }

    found = []

    for root in roots:
        if not root.is_dir():
            continue

        for base, dirs, files in os.walk(root):
            depth = len(Path(base).relative_to(root).parts)

            dirs[:] = [
                x for x in dirs
                if x not in (
                    '.git', 'raw', 'unpacked', 'results',
                    'test', 'tests', 'training', 'venv',
                    '.venv', 'data', 'models', 'full_train',
                    'predictions'
                )
            ]

            if depth >= 5:
                dirs[:] = []

            for name in files:
                if not re.search(
                    r'split|partition|assignment|calibration|development',
                    name, re.I
                ):
                    continue

                if re.search(
                    r'prediction|metric|score|eval|result|label',
                    name, re.I
                ):
                    continue

                p = Path(base) / name

                if p.is_symlink():
                    continue

                if not name.lower().endswith(
                    ('.csv', '.tsv', '.json')
                ):
                    continue

                if p.stat().st_size > 20 * 1024 * 1024:
                    continue

                # JSON: solo extraer metadatos de particion
                # con nombres explicitamente autorizados.

                if p.suffix.lower() == '.json':
                    state = 'JSON_CANDIDATO_SIN_LEER'
                    counts = ''

                    allowed_names = (
                        'split_metadata.json',
                        'partition_manifest.json',
                        'split_manifest.json',
                        'development_calibration_test_manifest.json'
                    )

                    if name.lower() in allowed_names:
                        try:
                            metadata = json.loads(p.read_text())
                            keys = (
                                'unit_of_split',
                                'proteins_per_split',
                                'groups_per_split',
                                'systems_per_split',
                                'replicas_per_split',
                                'split_sha256',
                                'seed'
                            )
                            counts = json.dumps({
                                k: metadata[k]
                                for k in keys if k in metadata
                            }, ensure_ascii=False)[:800]
                            state = 'METADATOS_SPLIT_SIN_RESULTADOS'
                        except (OSError, ValueError, TypeError) as exc:
                            state = 'JSON_NO_INTERPRETABLE_' + str(exc)[:100]

                    found.append({
                        'path': str(p),
                        'sha256': sha(p),
                        'state': state,
                        'counts': counts,
                        'commit': 'NO_ACREDITADO'
                    })
                    continue

                delimiter = (
                    '\t' if p.suffix.lower() == '.tsv' else ','
                )

                try:
                    with p.open(
                        newline='', encoding='utf-8-sig'
                    ) as handle:

                        reader = csv.DictReader(
                            handle, delimiter=delimiter
                        )

                        cols = set(reader.fieldnames or [])

                        split_col = next(
                            (
                                x for x in (
                                    'split', 'partition',
                                    'subset', 'set', 'role'
                                ) if x in cols
                            ),
                            None
                        )

                        id_col = next(
                            (
                                x for x in (
                                    'system_id', 'system',
                                    'protein_id'
                                ) if x in cols
                            ),
                            None
                        )

                        group_col = next(
                            (
                                x for x in (
                                    'group_id', 'sequence_group',
                                    'group', 'cluster_id'
                                ) if x in cols
                            ),
                            None
                        )

                        if not split_col or not id_col:
                            continue

                        data = defaultdict(
                            lambda: {
                                'systems': set(),
                                'groups': set(),
                                'replicas': set()
                            }
                        )

                        memberships = defaultdict(set)

                        for row in reader:
                            split = row[split_col].strip().lower()
                            system = row[id_col].strip()

                            if not split or not system:
                                continue

                            data[split]['systems'].add(system)

                            if group_col and row[group_col].strip():
                                group = row[group_col].strip()
                                data[split]['groups'].add(group)
                                memberships[group].add(split)

                            if 'replica' in cols:
                                data[split]['replicas'].add(
                                    (system, row['replica'])
                                )

                    counts = '; '.join(
                        '%s: sistemas=%d grupos=%s replicas=%s'
                        % (
                            label,
                            len(value['systems']),
                            str(len(value['groups']))
                                if group_col else 'NO_DISPONIBLE',
                            str(len(value['replicas']))
                                if 'replica' in cols
                                else 'NO_DISPONIBLE'
                        )
                        for label, value in sorted(data.items())
                    )

                    crossing = sum(
                        len(parts) > 1
                        for parts in memberships.values()
                    )

                    # Si no incluye grupos, se pueden derivar
                    # usando el mapa de ATLAS942, solo si
                    # todos los sistemas estan identificados.

                    if (
                        not group_col
                        and atlas_groups
                        and all(
                            s in atlas_groups
                            for v in data.values()
                            for s in v['systems']
                        )
                    ):
                        derived = defaultdict(set)

                        for label, value in data.items():
                            value['groups'] = {
                                atlas_groups[s]
                                for s in value['systems']
                            }

                            for group in value['groups']:
                                derived[group].add(label)

                        crossing = sum(
                            len(parts) > 1
                            for parts in derived.values()
                        )

                        counts += (
                            ' | grupos derivados de ATLAS942: '
                            + ', '.join(
                                '%s=%d' % (
                                    label, len(value['groups'])
                                )
                                for label, value in sorted(data.items())
                            )
                        )

                    commit = 'NO_ACREDITADO'

                    for repo in (
                        H / 'paper2-auto-v3',
                        H / 'paper2-atlas942-publication'
                    ):
                        try:
                            relative = p.relative_to(repo)

                            tracked = git(
                                repo,
                                [
                                    'ls-files',
                                    '--error-unmatch',
                                    str(relative)
                                ]
                            )

                            if tracked not in (
                                'NO_CONFIRMADO', 'NO_REPO_GIT'
                            ):
                                candidate = git(
                                    repo,
                                    [
                                        'log', '-1',
                                        '--format=%H',
                                        '--', str(relative)
                                    ]
                                )

                                dirty = git(
                                    repo,
                                    [
                                        'status', '--porcelain',
                                        '--', str(relative)
                                    ]
                                )

                                commit = (
                                    candidate if not dirty
                                    else 'MODIFICADO_DESDE_' + candidate
                                )

                        except ValueError:
                            pass

                    found.append({
                        'path': str(p),
                        'sha256': sha(p),
                        'state': (
                            'GRUPOS_CRUZADOS_' + str(crossing)
                            if crossing else 'ASIGNACION'
                        ),
                        'counts': counts,
                        'commit': commit
                    })

                    log(
                        '%s\n  SHA256=%s\n  commit=%s\n'
                        '  %s\n  grupos_cruzados=%d'
                        % (p, sha(p), commit, counts, crossing)
                    )

                except (OSError, ValueError, csv.Error) as exc:
                    found.append({
                        'path': str(p),
                        'sha256': 'NO_DISPONIBLE',
                        'state': 'ERROR: ' + str(exc),
                        'counts': '',
                        'commit': 'NO_ACREDITADO'
                    })

    save(
        'manifiestos_particiones.csv',
        found,
        ['path', 'sha256', 'state', 'counts', 'commit']
    )

    for repo in (
        H / 'paper2-auto-v3',
        H / 'paper2-atlas942-publication'
    ):
        log(
            'CHECKOUT %s HEAD=%s'
            % (repo, git(repo, ['rev-parse', 'HEAD']))
        )

    log(
        'AVISO: train/validation/test no equivale '
        'automaticamente a development/calibration/test.'
    )


def main():
    if not RUN.is_dir():
        raise SystemExit(
            'FALTA EJECUCION ATLAS942: ' + str(RUN)
        )

    OUT.mkdir(parents=True, exist_ok=False)

    log('CARPETA DE AUDITORIA: ' + str(OUT))
    log('SOLO LECTURA: sin simulaciones y sin resultados de test.')

    if shutil.which('sacct'):
        try:
            result = subprocess.run(
                [
                    'sacct', '-j', '3262490',
                    '-n', '-P',
                    '--format=JobID,JobName,Account,Partition,State,ExitCode'
                ],
                capture_output=True,
                text=True,
                timeout=25
            )

            log(
                'SACCT 3262490: '
                + (
                    result.stdout.strip()[:1500]
                    if result.returncode == 0
                    else 'NO_DISPONIBLE'
                )
            )

        except (OSError, subprocess.TimeoutExpired):
            log('SACCT 3262490: NO_DISPONIBLE')

    methods(RUN, 'ATLAS942')

    if PILOT.is_dir():
        methods(PILOT, 'PILOTO_500')
    else:
        log('PILOTO_500 NO LOCALIZADO: ' + str(PILOT))

        for p in (W / 'results').iterdir():
            if (
                p.is_dir()
                and re.search(
                    r'poc|pilot|earlystop|500',
                    p.name, re.I
                )
            ):
                log(
                    'CANDIDATO PILOTO SIN CONFIRMAR: ' + str(p)
                )

    counts()
    splits()

    log(
        '\nFIN DE AUDITORIA. '
        'Ausencias y discrepancias quedan PENDIENTES.'
    )

    (OUT / 'INFORME.txt').write_text(
        '\n'.join(REPORT) + '\n',
        encoding='utf-8'
    )

    log('CARPETA_RESULTADOS=' + str(OUT))


if __name__ == '__main__':
    main()
