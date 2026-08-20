#!/bin/bash
# Flatten head at the same three seeds, so the headline pooling-vs-flatten
# comparison is paired on seed as well as on clips.
PY=/Users/jshum/Desktop/code-folders/sicknote/.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
for SEED in 100 200 300; do
  echo "##### flatten seed=$SEED #####"
  $PY -u -m model.crossval --head flatten --select val_auc_smooth --seed "$SEED" 2>&1 | tail -4
done
echo "ALL FLATTEN SEEDS DONE"
