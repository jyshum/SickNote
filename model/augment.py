"""
Spectrogram augmentation shared by the measurement harness and the trainer.

This lives in its own module for one reason: model/crossval.py measured a
configuration and model/train.py ships one, and if each owned a copy of this
code they would eventually disagree. That is the same class of bug as the
threshold leak in EXPERIMENTS.md section 3 — a number established one way and
applied another. Importing the same function from both makes it impossible.
"""

# SpecAugment masks, sized for a 64 x 345 spectrogram. Deliberately milder
# than the speech-recognition defaults — 1,542 training clips is not much to
# hide parts of.
FREQ_MASKS, FREQ_PARAM = 2, 12   # up to 2 bands of <=12 of 64 mel bins
TIME_MASKS, TIME_PARAM = 2, 40   # up to 2 bands of <=40 of 345 frames


def spec_augment(x, rng):
    """Black out random frequency and time bands. Training only.

    Applied AFTER normalisation, so a masked region is set to 0 — which is the
    dataset mean, i.e. "no information here" rather than "silence". Silence has
    its own meaning in a spectrogram and we do not want to fake it.

    Unlike augmenting the waveform (noise injection, time stretching, both
    tried and dropped earlier in this project), nothing here distorts the
    signal. Whatever survives the mask is untouched real audio; the model just
    cannot lean on any single frequency band or moment to make its decision.

    Args:
        x:   (batch, 1, n_mels, n_frames) normalised spectrograms
        rng: numpy Generator — passed in so the caller owns reproducibility

    Returns a new tensor; the input is not modified.
    """
    x = x.clone()
    n, _, n_mels, n_frames = x.shape
    for i in range(n):
        for _ in range(FREQ_MASKS):
            f = int(rng.integers(0, FREQ_PARAM + 1))
            if f:
                f0 = int(rng.integers(0, n_mels - f + 1))
                x[i, :, f0:f0 + f, :] = 0.0
        for _ in range(TIME_MASKS):
            t = int(rng.integers(0, TIME_PARAM + 1))
            if t:
                t0 = int(rng.integers(0, n_frames - t + 1))
                x[i, :, :, t0:t0 + t] = 0.0
    return x
