#!/bin/bash
#SBATCH --job-name=md_poc_check
#SBATCH --account=uva_dma_2
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "========================================"
echo "PRUEBA SLURM"
echo "========================================"
echo "Fecha:             $(date)"
echo "Usuario:           $(whoami)"
echo "Job ID:            ${SLURM_JOB_ID:-no_disponible}"
echo "Job name:          ${SLURM_JOB_NAME:-no_disponible}"
echo "Nodo de cálculo:   $(hostname)"
echo "Partición:         ${SLURM_JOB_PARTITION:-no_disponible}"
echo "Cuenta:            ${SLURM_JOB_ACCOUNT:-no_disponible}"
echo "Directorio envío:  ${SLURM_SUBMIT_DIR:-no_disponible}"
echo "Nodos asignados:   ${SLURM_JOB_NODELIST:-no_disponible}"
echo "CPUs por tarea:    ${SLURM_CPUS_PER_TASK:-no_disponible}"
echo "Número de tareas:  ${SLURM_NTASKS:-no_disponible}"
echo "Memoria solicitada: 1G"

echo
echo "Información de CPU:"
lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)' || true

echo
echo "Comprobación mediante srun:"
srun --label hostname

echo
echo "Escritura en SCRATCH:"
TEST_FILE="$SCRATCH/md-hpc-benchmarks/tmp/slurm_test_${SLURM_JOB_ID}.txt"
echo "Trabajo ejecutado en $(hostname) el $(date)" > "$TEST_FILE"
ls -lh "$TEST_FILE"
cat "$TEST_FILE"

echo
echo "Prueba completada correctamente."
