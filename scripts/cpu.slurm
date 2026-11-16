#!/bin/bash
# Block E CPU jobs (<account>@cpu). Runs $CMDS lines.
#SBATCH --account=<account>@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
export HF_HOME=$WORK_ROOT/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true WANDB_MODE=disabled
export RLW=$WORK_ROOT/rlhead TOKENIZERS_PARALLELISM=false NPROC=38
cd $RLW/scripts
source $RLW/scripts/stage_env.sh
rc=0
while IFS= read -r line; do
  [ -z "$line" ] && continue; echo "=== $(date +%T) $line"
  eval "srun $line" < /dev/null || { echo "FAILED: $line"; rc=1; }
done < $CMDS
echo ALL_DONE rc=$rc; exit $rc
