"""
inference.py must build the architecture the checkpoints were trained with,
and must refuse to serve rather than quietly serve something else.
"""
import os

import pytest
import torch

import api.inference as inference
from model.architecture import SickNoteCNN


@pytest.fixture(autouse=True)
def _clear_module_cache():
    """_load_models memoises into module globals; reset between tests."""
    inference._models = None
    inference._params = None
    yield
    inference._models = None
    inference._params = None


def _write_params(tmp_path):
    torch.save(
        {"mean": -50.0, "std": 38.0, "n_mels": 64, "time_frames": 345,
         "sample_rate": 22050, "clip_length": 8, "n_fft": 1024, "hop_length": 512},
        tmp_path / "preprocessing_params.pt",
    )


class TestCheckpointCompatibility:

    def test_loads_a_pool_time_ensemble(self, tmp_path, monkeypatch):
        _write_params(tmp_path)
        for i in range(3):
            m = SickNoteCNN(n_mels=64, time_frames=345, head="pool_time")
            torch.save(m.state_dict(), tmp_path / f"model_final_{i}.pt")
        monkeypatch.setattr(inference, "_CHECKPOINT_DIR", str(tmp_path))

        models, params = inference._load_models()
        assert len(models) == 3
        assert params["n_mels"] == 64

    def test_rejects_a_flatten_checkpoint_loudly(self, tmp_path, monkeypatch):
        _write_params(tmp_path)
        stale = SickNoteCNN(n_mels=64, time_frames=345, head="flatten")
        torch.save(stale.state_dict(), tmp_path / "model_final_0.pt")
        monkeypatch.setattr(inference, "_CHECKPOINT_DIR", str(tmp_path))

        with pytest.raises(RuntimeError) as excinfo:
            inference._load_models()
        assert "model_final_0.pt" in str(excinfo.value)

    def test_raises_when_no_checkpoints_exist(self, tmp_path, monkeypatch):
        _write_params(tmp_path)
        monkeypatch.setattr(inference, "_CHECKPOINT_DIR", str(tmp_path))

        with pytest.raises(RuntimeError) as excinfo:
            inference._load_models()
        assert "no ensemble checkpoints" in str(excinfo.value).lower()

    def test_raises_on_a_partial_ensemble(self, tmp_path, monkeypatch):
        """Two of three members present must fail, not silently average two."""
        _write_params(tmp_path)
        for i in range(2):
            m = SickNoteCNN(n_mels=64, time_frames=345, head="pool_time")
            torch.save(m.state_dict(), tmp_path / f"model_final_{i}.pt")
        monkeypatch.setattr(inference, "_CHECKPOINT_DIR", str(tmp_path))

        with pytest.raises(RuntimeError) as excinfo:
            inference._load_models()
        assert "2" in str(excinfo.value) and "3" in str(excinfo.value)


class TestServingPathStaysLight:
    """The API container does not install scikit-learn.

    Commit da1ac74 dropped pandas and scikit-learn from requirements.txt to cut
    ~150MB off the deploy image; they are training-only dependencies. But
    model/train.py imports sklearn at module level, so any `from model.train
    import ...` in the serving path passes locally — sklearn is in the dev venv
    — and raises ModuleNotFoundError on every prediction in production.

    This runs in a subprocess so the check is not fooled by sklearn already
    being imported by another test in the same session.
    """

    def test_loading_the_ensemble_does_not_import_sklearn_or_model_train(self, tmp_path):
        import subprocess
        import sys

        from model.config import ENSEMBLE_SEEDS, HEAD, POOL

        _write_params(tmp_path)
        for i in range(len(ENSEMBLE_SEEDS)):
            m = SickNoteCNN(n_mels=64, time_frames=345, head=HEAD, pool=POOL)
            torch.save(m.state_dict(), tmp_path / f"model_final_{i}.pt")

        script = (
            "import sys\n"
            "import api.inference as inf\n"
            f"inf._CHECKPOINT_DIR = {str(tmp_path)!r}\n"
            "inf._load_models()\n"
            "bad = [m for m in sys.modules if m.startswith('sklearn') or m == 'model.train']\n"
            "print(','.join(sorted(bad)))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(inference.__file__))),
            env={**os.environ, "PYTHONPATH": "."},
        )
        assert out.returncode == 0, f"subprocess failed:\n{out.stderr}"
        leaked = out.stdout.strip()
        assert leaked == "", (
            f"the serving path pulled in training-only modules: {leaked}. "
            "These are not in requirements.txt and will fail in production."
        )
