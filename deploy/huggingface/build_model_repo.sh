#!/usr/bin/env bash
# Assemble a Hugging Face *model* repository from this repository.
#
# Model repos are free for public use on a free account; Docker Spaces require
# PRO ($9/mo) as of mid-2026, which is more than the Railway bill this project
# already pays. So the Hub gets the model and the card; Railway keeps serving
# the app.
#
# Usage:
#   bash deploy/huggingface/build_model_repo.sh <hf-username>
set -euo pipefail

USER="${1:-}"
if [ -z "$USER" ]; then
  echo "usage: bash deploy/huggingface/build_model_repo.sh <hf-username>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/build/hf-model"

rm -rf "$OUT"
mkdir -p "$OUT/checkpoints"

# The card is the point of this repo, so it becomes README.md.
cp "$ROOT/deploy/huggingface/MODEL_CARD.md" "$OUT/README.md"

# Weights: the three shipped members plus the normalisation stats needed to
# reproduce the exact input the model was trained on.
cp "$ROOT"/model/checkpoints/model_final_[0-2].pt "$OUT/checkpoints/"
cp "$ROOT/model/checkpoints/preprocessing_params.pt" "$OUT/checkpoints/"

# Enough code to actually load and run it, keeping the model/ package layout.
# architecture.py does `from model import config`, so flattening it breaks the
# import — verified, not assumed.
mkdir -p "$OUT/model"
cp "$ROOT/model/__init__.py"     "$OUT/model/__init__.py"
cp "$ROOT/model/architecture.py" "$OUT/model/architecture.py"
cp "$ROOT/model/config.py"       "$OUT/model/config.py"

cat > "$OUT/example.py" <<'PY'
"""
Minimal end-to-end use of the SickNote ensemble.

    pip install torch torchaudio huggingface_hub
    python example.py path/to/cough.wav
"""
import sys

import torch
import torchaudio
from huggingface_hub import snapshot_download

REPO = "jyShum/sicknote"
MEMBERS = 3


def load_ensemble():
    # snapshot_download pulls the model/ package too, which architecture.py needs
    local = snapshot_download(REPO)
    sys.path.insert(0, local)
    params = torch.load(
        f"{local}/checkpoints/preprocessing_params.pt",
        weights_only=True, map_location="cpu",
    )
    from model.architecture import SickNoteCNN

    models = []
    for i in range(MEMBERS):
        path = f"{local}/checkpoints/model_final_{i}.pt"
        m = SickNoteCNN(
            n_mels=params["n_mels"], time_frames=params["time_frames"],
            head="pool_time", pool="sym",
        )
        m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        m.eval()
        models.append(m)
    return models, params


def to_spectrogram(path, params):
    """Must match training exactly, or predictions drift."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != params["sample_rate"]:
        wav = torchaudio.functional.resample(wav, sr, params["sample_rate"])

    n = int(params["sample_rate"] * params["clip_length"])
    wav = torch.nn.functional.pad(wav, (0, max(0, n - wav.shape[1])))[:, :n]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=params["sample_rate"], n_mels=params["n_mels"],
        n_fft=params["n_fft"], hop_length=params["hop_length"],
    )(wav)
    db = torchaudio.transforms.AmplitudeToDB()(mel)
    return ((db - params["mean"]) / params["std"]).unsqueeze(0)


if __name__ == "__main__":
    models, params = load_ensemble()
    x = to_spectrogram(sys.argv[1], params)
    with torch.no_grad():
        prob = sum(torch.sigmoid(m(x)).item() for m in models) / len(models)

    # 0.46, selected on validation under an 80% sensitivity floor. Read the
    # limitations in README.md before acting on either branch of this.
    print(f"probability abnormal: {prob:.4f}")
    print(f"label: {'abnormal' if prob >= 0.46 else 'healthy'}")
PY

printf '__pycache__/\n*.pyc\n' > "$OUT/.gitignore"

echo "assembled: $OUT"
echo "  size: $(du -sh "$OUT" | cut -f1)"
find "$OUT" -type f | sed "s|$OUT|  .|"
echo
echo "next:"
echo "  1. create the model repo at https://huggingface.co/new (type: Model)"
echo "  2. cd build/hf-model"
echo "  3. git init -b main && git remote add origin https://huggingface.co/$USER/sicknote"
echo "  4. git add -A && git commit -m 'SickNote cough classifier' && git push -u origin main"
