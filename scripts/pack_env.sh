#!/bin/bash
# Login node: pack venv + uv python into one tar on $SCRATCH (Lustre small-file imports took ~15 min per job on 14 Nov).
set -e
cd $WORK
tar -cf $SCRATCH/rlenv.tar.tmp -C $WORK/venvs rlhead -C $WORK/.local uv-python
mv $SCRATCH/rlenv.tar.tmp $SCRATCH/rlenv.tar; ls -l $SCRATCH/rlenv.tar; echo PACK_DONE
