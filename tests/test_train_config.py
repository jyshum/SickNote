"""
train.py must default to the configuration EXPERIMENTS.md section 4b measured,
and must ship three ensemble members rather than five.
"""
import inspect

import torch

from model.architecture import SickNoteCNN
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
