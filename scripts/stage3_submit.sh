#!/bin/bash
# Submit Stage 3 once all Stage 2 runs are DONE/COLLAPSED (login node).
cd $WORK/rlhead/scripts
sb(){ sbatch --parsable --export=ALL --chdir=$WORK/rlhead "$@"; }
g1=$(CMDS=cmds/S3_gen_final.txt sb --qos=qos_gpu_a100-dev -t 01:00:00 -J E_S3gf rl.slurm)
g2=$(CMDS=cmds/S3_gen_curve.txt sb --qos=qos_gpu_a100-dev -t 01:00:00 -J E_S3gc rl.slurm)
c2=$(CMDS=cmds/C2_metrics_final_handlabel.txt sb --dependency=afterok:$g1 -J E_C2 -t 01:00:00 cpu.slurm)
s1=$(CMDS=cmds/S3_score_final.txt sb --dependency=afterok:$g1 --qos=qos_gpu_a100-dev -t 01:30:00 -J E_S3sf rl.slurm)
c3=$(CMDS=cmds/C3_metrics_curve.txt sb --dependency=afterok:$g2 -J E_C3 -t 01:00:00 cpu.slurm)
s2=$(CMDS=cmds/S3_score_curve.txt sb --dependency=afterok:$g2 --qos=qos_gpu_a100-dev -t 01:00:00 -J E_S3sc rl.slurm)
c4=$(CMDS=cmds/C4_analyze.txt sb --dependency=afterok:$c2:$s1:$c3:$s2 -J E_C4 -t 00:30:00 cpu.slurm)
echo "gen_final=$g1 gen_curve=$g2 metrics_final+handlabel=$c2 score_final=$s1 metrics_curve=$c3 score_curve=$s2 analyze=$c4" | tee -a ../logs/stage3_jobs.log
