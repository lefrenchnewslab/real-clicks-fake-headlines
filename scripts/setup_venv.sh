#!/bin/bash
# Block E Stage 0. Run on a Jean Zay LOGIN node (internet). 0 GPU h. Isolated uv env (module peft/TE break under new transformers).
set -euo pipefail
cd $WORK/rlhead; mkdir -p logs; LOG=logs/setup_venv_$(date +%F_%H%M).log
exec > >(tee -a "$LOG") 2>&1
module purge; module load arch/a100 pytorch-gpu/py3/2.8.0
export PYTHONUSERBASE=$WORK/.local PATH=$WORK/.local/bin:$PATH
export UV_CACHE_DIR=$WORK/.cache/uv UV_PYTHON_INSTALL_DIR=$WORK/.local/uv-python
command -v uv >/dev/null || pip install --user -q uv
V=$WORK/venvs/rlhead
[ -d "$V" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" "torch==2.8.0" "trl==1.16.0" peft transformers accelerate datasets spacy vaderSentiment \
   scikit-learn pandas pyarrow rapidfuzz word2number wordfreq matplotlib sentencepiece protobuf tiktoken scipy tqdm huggingface_hub
uv pip freeze --python "$V/bin/python" > logs/venv_freeze.txt
"$V/bin/python" - <<'PY'
import inspect, torch, trl, transformers, peft, accelerate, datasets, spacy
print("torch", torch.__version__, "cuda", torch.version.cuda, "trl", trl.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__, "accelerate", accelerate.__version__, "datasets", datasets.__version__, "spacy", spacy.__version__)
open("logs/grpo_signature.txt","w").write(str(inspect.signature(trl.GRPOConfig)).replace(", ", ",\n"))
print("GRPOConfig fields saved")
PY
echo SETUP_DONE
