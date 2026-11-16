#!/bin/bash
# Login node: rebuild usage.csv from sacct for all Block E jobs (names E_*). A100-h = elapsed x GPUs.
cd $WORK/rlhead
echo "job,name,start,end,state,elapsed_s,a100_h,cumulative_a100_h" > usage.csv
sacct -u $USER -S 2026-11-14 -X -n -P -o JobID,JobName,Start,End,State,ElapsedRaw,AllocTRES | grep "|E_" | sort -t'|' -k3 | \
awk -F'|' '{g=0; if (match($7,/gres\/gpu=[0-9]+/)) {g=substr($7,RSTART+9,RLENGTH-9)}; h=$6*g/3600; c+=h; printf "%s,%s,%s,%s,%s,%s,%.3f,%.3f\n",$1,$2,$3,$4,$5,$6,h,c}' >> usage.csv
tail -1 usage.csv
