#!/bin/bash
# Login node, 0 GPU h. Runs after setup_venv.sh.
cd $WORK/rlhead; export HF_HOME=$WORK/hf; P=$WORK/venvs/rlhead/bin/python
$P -u scripts/download_upworthy.py > logs/dl_upworthy.log 2>&1 &
$P -u scripts/download_ccnews.py > logs/dl_ccnews.log 2>&1 &
$P -u scripts/download_models.py > logs/dl_models.log 2>&1 &
wait; echo DOWNLOADS_DONE
