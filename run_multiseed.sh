#!/bin/bash
# Three training seeds per configuration. Fold assignment is held fixed, so
# every run scores the identical clips and only training randomness changes.
PY=/Users/jshum/Desktop/code-folders/sicknote/.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
for SEED in 100 200 300; do
  echo "##### pool_time seed=$SEED #####"
  $PY -u -m model.crossval --head pool_time --select val_auc_smooth --seed "$SEED" 2>&1 | tail -6
  echo "##### pool_time+specaug seed=$SEED #####"
  $PY -u -m model.crossval --head pool_time --select val_auc_smooth --specaug --seed "$SEED" 2>&1 | tail -6
done
echo "ALL RUNS DONE"
