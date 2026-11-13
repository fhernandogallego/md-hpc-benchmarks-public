#!/usr/bin/env bash
# Launch ONLY the first ten-system pilot after synthetic verification.
# Execute with: bash /path/instalar_piloto_cils_v3.sh (never source it).
set -euo pipefail
WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks
BASE="$WORK/maintenance/earlystop"
STATE="$WORK/maintenance/paper2_cils_v3_relaunch_v1"
PY=/home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/python3
mkdir -p "$BASE"
test -x "$PY"

"$PY" - "$BASE" <<'PYTHON'
import base64, hashlib, json, os, pathlib, sys, tempfile, urllib.request
base=pathlib.Path(sys.argv[1])
repo='fhernandogallego/md-hpc-benchmarks-public'
revision='178e42599aa0e65ca4b8a0da3b5b18732d13c27c'
root='docs/earlystop/correcciones_analisis/'
files={
 'cils_v3_shards.py':'42a88171b3a2a3144652e294e390476d4264099f',
 'cils_v3_pilot.sbatch':'1f8d5ff8c6a18d19fc38467ab47cd9bcf26b6488',
}
def gitsha(data):
 return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
for name,expected in files.items():
 dest=base/name
 if dest.exists():
  data=dest.read_bytes()
  if gitsha(data)!=expected:raise SystemExit('REFUSO_SOBRESCRIBIR_ARCHIVO_DIFERENTE: '+str(dest))
  print('INSTALACION_REUTILIZADA',name,expected,flush=True)
  continue
 url=f'https://api.github.com/repos/{repo}/contents/{root}{name}?ref={revision}'
 req=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'earlystop-scayle-audit','X-GitHub-Api-Version':'2022-11-28'})
 with urllib.request.urlopen(req,timeout=120) as response: item=json.load(response)
 data=base64.b64decode(item['content'])
 if item['sha']!=expected or gitsha(data)!=expected:
  raise SystemExit('HASH_ARCHIVO_REMOTO_DIFERENTE: '+name)
 fd,tmp=tempfile.mkstemp(dir=base,prefix='.download_'+name+'_')
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.replace(tmp,dest)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
 print('INSTALADO_VERIFICADO',name,'bytes',len(data),'gitsha',expected,flush=True)
PYTHON

"$PY" -m py_compile "$BASE/cils_v3_shards.py"
bash -n "$BASE/cils_v3_pilot.sbatch"
TEST_LOG="$BASE/SELFTEST_CILS_V3_SCAYLE.log"
"$PY" "$BASE/cils_v3_shards.py" selftest \
 --report "$BASE/SELFTEST_CILS_V3_SCAYLE.json" 2>&1 | tee "$TEST_LOG"
grep -q '^SELFTEST=PASS ' "$TEST_LOG"

# Strict real development-only freeze, no trajectory analysis here.
"$PY" "$BASE/cils_v3_shards.py" init
mkdir -p "$STATE/logs"

cat > "$BASE/verificar_piloto_cils_v3.sh" <<'VERIFY'
#!/usr/bin/env bash
set -euo pipefail
WORK=/scratch/uva_dma_2/uva_dma_2_1/md-hpc-benchmarks
BASE="$WORK/maintenance/earlystop"
STATE="$WORK/maintenance/paper2_cils_v3_relaunch_v1"
PY=/home/uva_dma_2/uva_dma_2_1/venvs/md-poc-py311/bin/python3
JOB=$(cat "$STATE/pilot_job_id.txt")
printf 'PILOT_JOB_ID=%s\n' "$JOB"
sacct -j "$JOB" -P --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS
for ext in out err; do
 p="$STATE/logs/pilot_$JOB.$ext"
 if [ -f "$p" ]; then
  echo "===== $p ====="
  tail -n 70 "$p"
 fi
done
"$PY" - "$BASE" "$JOB" <<'PYTHON'
import json, pathlib, subprocess, sys
sys.path.insert(0,sys.argv[1])
import cils_v3_shards as c
job=sys.argv[2]
proc=subprocess.run(['sacct','-j',job,'-P','--format=JobID,State,ExitCode'],capture_output=True,text=True,check=True)
records=[v.split('|') for v in proc.stdout.strip().splitlines()[1:] if v]
status={r[0]:(r[1],r[2]) for r in records if len(r)>=3}
if status.get(job)!=('COMPLETED','0:0') or status.get(job+'.batch')!=('COMPLETED','0:0'):
 raise SystemExit('PILOT_SLURM_NO_COMPLETADO: '+repr(status))
plan=c.checked_plan(c.STATE)
for policy in ('include','exclude'):
 cp,data=c.check_part(c.STATE,plan,0,policy)
 print('PILOT_VERIFIED',policy,'systems',cp['n_systems'],'rows',len(data),
       'wall_seconds',cp['elapsed_seconds'],'cpu_seconds',cp['child_cpu_seconds'],
       'maxrss_kib',cp['child_maxrss_kib'], 'sha256',cp['files']['targets.csv'])
assert len(plan['ids'])==757 and len(plan['shards'])==76
print('PILOT_VALIDATED=PASS; REST_OF_ARRAY_NOT_SUBMITTED')
PYTHON
VERIFY
chmod 700 "$BASE/verificar_piloto_cils_v3.sh"

# Safe resume: do not create a second job if the first is still queued.
JOBFILE="$STATE/pilot_job_id.txt"
if [ -s "$JOBFILE" ]; then
 OLD=$(cat "$JOBFILE")
 if squeue -h -j "$OLD" -o '%i' 2>/dev/null | grep -Eq '^[0-9]+'; then
  echo "PILOTO_YA_EN_COLA=$OLD"
  exit 0
 fi
fi

if [ -f "$STATE/parts/include_000/complete.json" ] && \
   [ -f "$STATE/parts/exclude_000/complete.json" ]; then
 "$PY" - "$BASE" <<'PYTHON'
import sys
sys.path.insert(0,sys.argv[1]);import cils_v3_shards as c
p=c.checked_plan(c.STATE)
for mode in ('include','exclude'):c.check_part(c.STATE,p,0,mode)
print('PILOT_CHECKPOINTS_ALREADY_COMPLETE=PASS; NO_NEW_JOB')
PYTHON
 exit 0
fi

RAW=$(sbatch --parsable "$BASE/cils_v3_pilot.sbatch")
JOB=$(printf '%s' "$RAW" | cut -d';' -f1)
if [[ ! "$JOB" =~ ^[0-9]+$ ]]; then
 echo "ERROR_JOB_ID=$RAW" >&2
 exit 1
fi
printf '%s\n' "$JOB" > "$JOBFILE"
printf '%s job=%s sha_driver=%s\n' "$(date -Is)" "$JOB" \
 '42a88171b3a2a3144652e294e390476d4264099f' >> "$STATE/pilot_jobs.txt"
echo "PILOT_SUBMITTED_JOB=$JOB"
echo "PILOT_STDOUT=$STATE/logs/pilot_$JOB.out"
echo "PILOT_STDERR=$STATE/logs/pilot_$JOB.err"
echo "AFTER_JOB=bash $BASE/verificar_piloto_cils_v3.sh"
squeue -j "$JOB" || true
