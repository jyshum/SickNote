#!/bin/bash
# Asymmetric pooling: keep more frequency resolution, pool time just as hard.
# Run on top of the current best config (pool_time head + SpecAugment).
PY=/Users/jshum/Desktop/code-folders/sicknote/.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
for POOL in freq16 freq32; do
  for SEED in 100 200 300; do
    echo "##### $POOL seed=$SEED #####"
    $PY -u -m model.crossval --head pool_time --select val_auc_smooth \
        --specaug --pool "$POOL" --seed "$SEED" 2>&1 | tail -3
  done
done
echo "ALL POOL SHAPES DONE"
