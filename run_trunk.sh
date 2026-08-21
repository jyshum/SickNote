#!/bin/bash
# Does widening the conv trunk help? [8,16,32] -> [16,32,64], on top of the
# current best config. The loss curves argue no; this measures it.
PY=/Users/jshum/Desktop/code-folders/sicknote/.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
for SEED in 100 200 300; do
  echo "##### channels 16,32,64 seed=$SEED #####"
  $PY -u -m model.crossval --head pool_time --select val_auc_smooth \
      --specaug --channels 16,32,64 --seed "$SEED" 2>&1 | tail -3
done
echo "TRUNK RUNS DONE"
