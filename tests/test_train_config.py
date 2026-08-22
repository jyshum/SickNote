"""
train.py must default to the configuration EXPERIMENTS.md section 4b measured,
and must ship three ensemble members rather than five.
"""
import inspect
import math
import os

import torch

import model.train as train_mod
from model.architecture import SickNoteCNN
from model.augment import spec_augment as real_spec_augment
from model.train import ENSEMBLE_SEEDS, train_single


class TestProductionDefaults:
    """The shipped defaults are the measured-best ones, not the historical ones."""

    def test_train_single_defaults_to_pool_time_and_specaug(self):
        sig = inspect.signature(train_single)
        assert sig.parameters["head"].default == "pool_time"
        assert sig.parameters["pool"].default == "sym"
        assert sig.parameters["specaug"].default is True

    def test_ensemble_is_three_members(self):
        # section 4b: members 4 and 5 are worth +0.0013 combined, inside noise
        assert len(ENSEMBLE_SEEDS) == 3

    def test_ensemble_seeds_are_a_prefix_of_the_old_five(self):
        # keeping the first three makes the new ensemble a strict subset
        assert ENSEMBLE_SEEDS == [42, 123, 456]


class TestArchitectureContract:
    """The head that train.py defaults to is the one inference.py must build."""

    def test_pool_time_model_is_small(self):
        model = SickNoteCNN(n_mels=64, time_frames=345, head="pool_time")
        assert sum(p.numel() for p in model.parameters()) == 71_793

    def test_flatten_model_is_the_old_large_one(self):
        model = SickNoteCNN(n_mels=64, time_frames=345, head="flatten")
        assert sum(p.numel() for p in model.parameters()) == 1_415_281

    def test_state_dicts_are_not_interchangeable(self):
        # this incompatibility is why inference.py must fail loudly (Task 4)
        pooled = SickNoteCNN(n_mels=64, time_frames=345, head="pool_time")
        flat = SickNoteCNN(n_mels=64, time_frames=345, head="flatten")
        try:
            pooled.load_state_dict(flat.state_dict())
        except RuntimeError:
            return
        raise AssertionError("expected a RuntimeError loading flatten weights into pool_time")


class TestAugmentationAppliesOnlyToNormalisedTrainingBatches:
    """Every other test in this file is static (signature/list/param-count

    checks). None of them actually run training, so none would notice if
    spec_augment leaked into the validation loop, or if CoughDataset stopped
    normalising before augmentation ran. This test spies on the real
    spec_augment call inside a real (tiny, truncated) train_single run and
    checks both properties directly.
    """

    def _small_splits(self):
        # Slice the real splits down to ~40 train / 16 val files so this stays
        # fast. 40 is deliberately not a multiple of BATCH_SIZE (32), so the
        # expected batch count below exercises the ceil(), not just a lucky
        # single-batch case. Both classes are present in each slice (checked
        # by hand against model/checkpoints/splits.pt) so pos_weight doesn't
        # divide by zero. 16 val files rather than 8: with only 8, one seed
        # produced a val set where the two negative-class samples happened to
        # get the highest predicted scores after a single epoch — a genuine
        # AUC of exactly 0.0 — which starves train_single's checkpoint step
        # (best_val_auc starts at 0.0 and requires strict improvement) and
        # makes it crash on the final shutil.copy2. 16 files avoids that.
        real_splits = torch.load(
            os.path.join(train_mod.CHECKPOINT_DIR, "splits.pt"), weights_only=False
        )
        return {
            "train_files": real_splits["train_files"][:40],
            "train_labels": real_splits["train_labels"][:40],
            "val_files": real_splits["val_files"][:16],
            "val_labels": real_splits["val_labels"][:16],
        }

    def _run_truncated(self, monkeypatch, small_splits, specaug):
        """Run train_single for one epoch against the truncated splits and
        return the list of (mean, std) pairs the spy observed."""
        real_torch_load = torch.load

        def fake_torch_load(path, *args, **kwargs):
            # Only intercept splits.pt. preprocessing_params.pt and the
            # per-file spectrogram loads inside CoughDataset must still hit
            # disk for real, so mean/std/n_mels/time_frames and the actual
            # tensors stay genuine.
            if str(path).endswith("splits.pt"):
                return small_splits
            return real_torch_load(path, *args, **kwargs)

        monkeypatch.setattr(torch, "load", fake_torch_load)

        # train.py does `from model.config import EPOCHS`, which binds EPOCHS
        # as a name inside model.train's own namespace at import time.
        # Patching model.config.EPOCHS here would be invisible to
        # model.train — it already has its own copy of the old value — so we
        # have to patch model.train.EPOCHS directly. This is a genuine trap:
        # it is exactly what the Task 2 smoke test tripped over.
        monkeypatch.setattr(train_mod, "EPOCHS", 1)

        calls = []

        def spy(x, rng):
            calls.append((x.mean().item(), x.std().item()))
            return real_spec_augment(x, rng)

        monkeypatch.setattr(train_mod, "spec_augment", spy)

        suffix = "_augtest"
        best_path = os.path.join(train_mod.CHECKPOINT_DIR, f"model_best{suffix}.pt")
        final_path = os.path.join(train_mod.CHECKPOINT_DIR, f"model_final{suffix}.pt")
        try:
            train_mod.train_single(seed=42, suffix=suffix, specaug=specaug)
        finally:
            # Never touch model_final_0.pt .. model_final_4.pt — only the
            # _augtest-suffixed files this run could have produced.
            for p in (best_path, final_path):
                if os.path.exists(p):
                    os.remove(p)

        return calls

    def test_specaug_called_once_per_training_batch_on_normalised_input(self, monkeypatch):
        small_splits = self._small_splits()
        n_train = len(small_splits["train_files"])
        expected_calls = math.ceil(n_train / train_mod.BATCH_SIZE)

        calls = self._run_truncated(monkeypatch, small_splits, specaug=True)

        # 1. Called once per TRAINING batch, not once per training+validation
        # batch. If augmentation leaked into the validation loop this count
        # would be higher than expected_calls.
        assert len(calls) == expected_calls

        # 2. Every tensor the spy saw was already normalised. Raw log-mel
        # spectrograms sit near -50 dB with std ~38 (see
        # preprocessing_params.pt: mean=-49.60, std=37.93). Normalised
        # batches should look like ~N(0, 1). If CoughDataset ever stopped
        # normalising, or normalise/augment got reordered, these tensors
        # would show that raw scale instead, and these bounds fail loudly.
        for mean, std in calls:
            assert abs(mean) < 1.0, f"unnormalised mean leaked into spec_augment: {mean}"
            assert 0.3 < std < 3.0, f"unnormalised std leaked into spec_augment: {std}"

    def test_specaug_false_produces_zero_calls(self, monkeypatch):
        small_splits = self._small_splits()
        calls = self._run_truncated(monkeypatch, small_splits, specaug=False)
        assert len(calls) == 0
