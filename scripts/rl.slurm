#!/bin/bash
# Block E: runs a list of commands from $CMDS (one per line) on 1x A100-80GB. Aborts if the GPU is not an A100.
#SBATCH --account=<account>@a100
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
module purge
module load arch/a100
export HF_HOME=$WORK_ROOT/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true WANDB_MODE=disabled
export RLW=$WORK_ROOT/rlhead PY=$WORK_ROOT/venvs/rlhead/bin/python TOKENIZERS_PARALLELISM=false
cd $RLW/scripts; export JOB_T0=$(date +%s)
source $RLW/scripts/stage_env.sh
nvidia-smi
G=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU_NAME $G"
case "$G" in *A100*) ;; *) echo "ABORT: not an A100"; exit 1;; esac
mkdir -p $RLW/runs/$SLURM_JOB_ID; cp $CMDS $RLW/runs/$SLURM_JOB_ID/cmds.txt; echo "$G" > $RLW/runs/$SLURM_JOB_ID/gpu_name.txt
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 30 > $RLW/logs/gpu_$SLURM_JOB_ID.csv &
rc=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "=== $(date +%T) $line"
  eval "srun $line" < /dev/null || { echo "FAILED: $line"; rc=1; }
done < $CMDS
echo ALL_DONE rc=$rc
exit $rc
