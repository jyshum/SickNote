#!/usr/bin/env bash
# Assemble a Hugging Face Space repo from this repository.
#
# The Space is a separate git repo on huggingface.co, so this stages a clean
# copy rather than duplicating source in-tree. Run it again after any change to
# api/, model/, or requirements.txt — it is idempotent.
#
# Usage:
#   bash deploy/huggingface/build_space.sh <hf-username>
#
# Then:
#   cd build/hf-space && git add -A && git commit -m "..." && git push
set -euo pipefail

USER="${1:-}"
if [ -z "$USER" ]; then
  echo "usage: bash deploy/huggingface/build_space.sh <hf-username>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/build/hf-space"

rm -rf "$OUT"
mkdir -p "$OUT"

# The Space needs only what the Dockerfile copies, plus the checkpoints.
cp "$ROOT/deploy/huggingface/Dockerfile" "$OUT/Dockerfile"
cp "$ROOT/deploy/huggingface/README.md"  "$OUT/README.md"
cp "$ROOT/requirements.txt"              "$OUT/requirements.txt"

mkdir -p "$OUT/api" "$OUT/model/checkpoints"
cp "$ROOT"/api/*.py   "$OUT/api/"
cp "$ROOT"/model/*.py "$OUT/model/"

# Only the three shipped members plus the preprocessing params. Everything else
# under checkpoints/ is training scaffolding the API never reads.
cp "$ROOT"/model/checkpoints/model_final_[0-2].pt "$OUT/model/checkpoints/"
cp "$ROOT/model/checkpoints/preprocessing_params.pt" "$OUT/model/checkpoints/"

# train.py imports scikit-learn, which requirements.txt deliberately excludes to
# keep the image small. The API never imports train.py — see api/inference.py —
# but shipping it would let a future edit break the Space silently.
rm -f "$OUT/model/train.py" "$OUT/model/crossval.py" "$OUT/model/ensemble_cv.py" \
      "$OUT/model/evaluate.py" "$OUT/model/explore.py" "$OUT/model/preprocess.py" \
      "$OUT/model/audit.py" "$OUT/model/derive_threshold.py" \
      "$OUT/model/select_demos.py"

echo "assembled: $OUT"
echo
echo "  size:      $(du -sh "$OUT" | cut -f1)"
echo "  checkpoints: $(ls "$OUT/model/checkpoints" | tr '\n' ' ')"
echo "  model/:      $(ls "$OUT/model"/*.py | xargs -n1 basename | tr '\n' ' ')"
echo
echo "next:"
echo "  1. create the Space at https://huggingface.co/new-space (SDK: Docker)"
echo "  2. cd build/hf-space"
echo "  3. git init && git remote add origin https://huggingface.co/spaces/$USER/sicknote"
echo "  4. git add -A && git commit -m 'SickNote API' && git push -u origin main"
