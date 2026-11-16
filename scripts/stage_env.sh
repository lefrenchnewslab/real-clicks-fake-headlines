# Sourced by rl.slurm: extract the packed env to a node-local, exec-capable dir and point $PY at it.
# /dev/shm is noexec on Jean Zay (job 189389), so probe candidates by actually executing a copied binary.
t=$(date +%s); D=""
for base in "$TMPDIR" /tmp "$JOBSCRATCH" "$SCRATCH/envstage"; do
  [ -n "$base" ] || continue; mkdir -p "$base" 2>/dev/null || continue
  free=$(df -Pk "$base" | awk 'NR==2{print $4}'); [ "${free:-0}" -gt 12000000 ] || { echo "skip $base (free ${free}k)"; continue; }
  cp /bin/true "$base/.xt_$SLURM_JOB_ID" 2>/dev/null && "$base/.xt_$SLURM_JOB_ID" 2>/dev/null && { rm -f "$base/.xt_$SLURM_JOB_ID"; D=$base/rl_$SLURM_JOB_ID; break; }
  rm -f "$base/.xt_$SLURM_JOB_ID"; echo "skip $base (noexec)"
done
echo "STAGE_DIR $D ($(df -Ph $(dirname $D) | awk 'NR==2{print $1, $4}'))"
mkdir -p $D; tar -xf $SCRATCH/rlenv.tar -C $D
PYBIN=$(ls -d $D/uv-python/cpython-3.12*/bin | head -1)
ln -sfn $PYBIN/python3.12 $D/rlhead/bin/python; ln -sfn $PYBIN/python3.12 $D/rlhead/bin/python3
sed -i "s#^home = .*#home = $PYBIN#" $D/rlhead/pyvenv.cfg
export PY=$D/rlhead/bin/python
trap "rm -rf $D" EXIT
echo "ENV_STAGED $D in $(( $(date +%s) - t ))s"; $PY -c "import sys; print(sys.prefix)"
