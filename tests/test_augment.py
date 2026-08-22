"""
SpecAugment — masking behaviour that both crossval.py and train.py depend on.
"""
import numpy as np
import torch

from model.augment import FREQ_MASKS, FREQ_PARAM, TIME_MASKS, TIME_PARAM, spec_augment


class TestSpecAugment:
    """spec_augment blacks out frequency and time bands, in place of nothing else."""

    def _batch(self, n=4, n_mels=64, n_frames=345):
        # all-ones so any zero in the output came from a mask
        return torch.ones(n, 1, n_mels, n_frames)

    def test_returns_same_shape(self):
        x = self._batch()
        out = spec_augment(x, np.random.default_rng(0))
        assert out.shape == x.shape

    def test_does_not_mutate_input(self):
        x = self._batch()
        spec_augment(x, np.random.default_rng(0))
        assert torch.all(x == 1.0), "input tensor was modified in place"

    def test_masks_are_zero_not_silence(self):
        # masked regions must be 0.0 (the post-normalisation dataset mean),
        # never a negative dB floor — see EXPERIMENTS.md section 4
        x = self._batch()
        out = spec_augment(x, np.random.default_rng(1))
        assert out.min() >= 0.0
        assert set(torch.unique(out).tolist()) <= {0.0, 1.0}

    def test_actually_masks_something(self):
        x = self._batch(n=16)
        out = spec_augment(x, np.random.default_rng(2))
        assert (out == 0.0).any(), "no masking occurred across 16 samples"

    def test_deterministic_for_a_given_rng_seed(self):
        x = self._batch()
        a = spec_augment(x, np.random.default_rng(7))
        b = spec_augment(x, np.random.default_rng(7))
        assert torch.equal(a, b)

    def test_mask_widths_stay_within_configured_bounds(self):
        # a fully-masked row would mean a band wider than FREQ_PARAM
        x = self._batch(n=32)
        out = spec_augment(x, np.random.default_rng(3))
        n_mels, n_frames = x.shape[2], x.shape[3]
        for i in range(x.shape[0]):
            zero_mel_rows = int((out[i, 0].sum(dim=1) == 0).sum())
            zero_time_cols = int((out[i, 0].sum(dim=0) == 0).sum())
            assert zero_mel_rows <= FREQ_MASKS * FREQ_PARAM
            assert zero_time_cols <= TIME_MASKS * TIME_PARAM
            assert zero_mel_rows < n_mels
            assert zero_time_cols < n_frames

    def test_masks_vary_across_samples_in_a_batch(self):
        # a batch-uniform bug (draw the mask once, apply to all) would still
        # satisfy every bound above — this is what catches it
        x = self._batch(n=24)
        out = spec_augment(x, np.random.default_rng(11))
        patterns = {tuple((out[i, 0].sum(dim=1) == 0).tolist()) for i in range(x.shape[0])}
        assert len(patterns) > 1, "every sample got an identical frequency mask"
