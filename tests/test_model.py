"""
P1 — model and dataset tests.
"""
import pandas as pd
import pytest

from model.explore import majority_label, has_good_quality


# ── majority_label tests ──────────────────────────────────────────────


class TestMajorityLabel:
    """Test majority_label: expert diagnosis_1-4 → 'healthy' | 'abnormal' | None."""

    def _row(self, d1=None, d2=None, d3=None, d4=None):
        """Helper to build a pandas Series with diagnosis columns."""
        return pd.Series({
            "diagnosis_1": d1,
            "diagnosis_2": d2,
            "diagnosis_3": d3,
            "diagnosis_4": d4,
        })

    def test_all_healthy(self):
        row = self._row("healthy_cough", "healthy_cough", "healthy_cough")
        assert majority_label(row) == "healthy"

    def test_all_abnormal(self):
        row = self._row("COVID-19", "upper_infection", "lower_infection")
        assert majority_label(row) == "abnormal"

    def test_majority_abnormal(self):
        row = self._row("healthy_cough", "COVID-19", "upper_infection", "lower_infection")
        assert majority_label(row) == "abnormal"

    def test_tie_defaults_to_abnormal(self):
        row = self._row("healthy_cough", "COVID-19")
        assert majority_label(row) == "abnormal"

    def test_no_labels_returns_none(self):
        row = self._row()
        assert majority_label(row) is None

    def test_single_expert_healthy(self):
        row = self._row("healthy_cough")
        assert majority_label(row) == "healthy"

    def test_single_expert_abnormal(self):
        row = self._row("COVID-19")
        assert majority_label(row) == "abnormal"

    def test_obstructive_disease_is_abnormal(self):
        row = self._row("obstructive_disease", "obstructive_disease")
        assert majority_label(row) == "abnormal"

    def test_majority_healthy(self):
        row = self._row("healthy_cough", "healthy_cough", "COVID-19")
        assert majority_label(row) == "healthy"

    def test_three_way_tie_abnormal(self):
        # 1 healthy, 1 COVID, 1 upper_infection → 1 healthy vs 2 abnormal → abnormal
        row = self._row("healthy_cough", "COVID-19", "upper_infection")
        assert majority_label(row) == "abnormal"


# ── has_good_quality tests ─────────────────────────────────────────────


class TestHasGoodQuality:
    """Test has_good_quality: quality_1-4 → True unless majority poor/no_cough."""

    def _row(self, q1=None, q2=None, q3=None, q4=None):
        """Helper to build a pandas Series with quality columns."""
        return pd.Series({
            "quality_1": q1,
            "quality_2": q2,
            "quality_3": q3,
            "quality_4": q4,
        })

    def test_all_good(self):
        row = self._row("good", "good", "good")
        assert has_good_quality(row) is True

    def test_all_ok(self):
        row = self._row("ok", "ok")
        assert has_good_quality(row) is True

    def test_majority_poor(self):
        row = self._row("poor", "poor", "good")
        assert has_good_quality(row) is False

    def test_majority_no_cough(self):
        row = self._row("no_cough", "no_cough", "ok")
        assert has_good_quality(row) is False

    def test_no_quality_info_returns_true(self):
        row = self._row()
        assert has_good_quality(row) is True

    def test_mixed_bad_types(self):
        # poor + no_cough = 2 bad vs 1 good → bad majority
        row = self._row("poor", "no_cough", "good")
        assert has_good_quality(row) is False

    def test_single_ok(self):
        row = self._row("ok")
        assert has_good_quality(row) is True

    def test_single_poor(self):
        row = self._row("poor")
        assert has_good_quality(row) is False

    def test_tie_good_and_bad(self):
        # 1 poor, 1 good → not a majority bad → True
        row = self._row("poor", "good")
        assert has_good_quality(row) is True

    def test_all_poor(self):
        row = self._row("poor", "poor", "poor", "poor")
        assert has_good_quality(row) is False


# ── load_and_filter_metadata tests ───────────────────────────────────


class TestLoadAndFilterMetadata:
    """Test load_and_filter_metadata: CSV → filtered DataFrame with label_int."""

    def _make_csv(self, tmp_path, rows):
        """Helper: write rows to a CSV and return the path."""
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "metadata_compiled.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def _base_row(self, uuid, cough_detected=0.9, d1="healthy_cough", q1="good"):
        """Create a minimal row that passes all filters."""
        return {
            "uuid": uuid,
            "cough_detected": cough_detected,
            "diagnosis_1": d1,
            "diagnosis_2": None,
            "diagnosis_3": None,
            "diagnosis_4": None,
            "quality_1": q1,
            "quality_2": None,
            "quality_3": None,
            "quality_4": None,
        }

    def test_filters_low_cough_detected(self, tmp_path):
        from model.preprocess import load_and_filter_metadata

        rows = [
            self._base_row("aaa", cough_detected=0.5),  # below threshold
            self._base_row("bbb", cough_detected=0.9),  # above threshold
        ]
        csv_path = self._make_csv(tmp_path, rows)
        result = load_and_filter_metadata(csv_path)
        assert len(result) == 1
        assert result.iloc[0]["uuid"] == "bbb"

    def test_filters_no_expert_label(self, tmp_path):
        from model.preprocess import load_and_filter_metadata

        rows = [
            {
                "uuid": "no_expert",
                "cough_detected": 0.9,
                "diagnosis_1": None,
                "diagnosis_2": None,
                "diagnosis_3": None,
                "diagnosis_4": None,
                "quality_1": "good",
                "quality_2": None,
                "quality_3": None,
                "quality_4": None,
            },
            self._base_row("with_expert"),
        ]
        csv_path = self._make_csv(tmp_path, rows)
        result = load_and_filter_metadata(csv_path)
        assert len(result) == 1
        assert result.iloc[0]["uuid"] == "with_expert"

    def test_filters_poor_quality(self, tmp_path):
        from model.preprocess import load_and_filter_metadata

        rows = [
            self._base_row("poor_qual", q1="poor"),
            self._base_row("good_qual", q1="good"),
        ]
        csv_path = self._make_csv(tmp_path, rows)
        result = load_and_filter_metadata(csv_path)
        assert len(result) == 1
        assert result.iloc[0]["uuid"] == "good_qual"

    def test_label_int_mapping(self, tmp_path):
        from model.preprocess import load_and_filter_metadata

        rows = [
            self._base_row("healthy_one", d1="healthy_cough"),
            self._base_row("abnormal_one", d1="COVID-19"),
        ]
        csv_path = self._make_csv(tmp_path, rows)
        result = load_and_filter_metadata(csv_path)
        assert len(result) == 2
        healthy = result[result["uuid"] == "healthy_one"]
        abnormal = result[result["uuid"] == "abnormal_one"]
        assert healthy.iloc[0]["label_int"] == 0
        assert abnormal.iloc[0]["label_int"] == 1

    def test_label_column_present(self, tmp_path):
        from model.preprocess import load_and_filter_metadata

        rows = [self._base_row("x")]
        csv_path = self._make_csv(tmp_path, rows)
        result = load_and_filter_metadata(csv_path)
        assert "label" in result.columns
        assert "label_int" in result.columns
        assert "uuid" in result.columns


# ── audio_to_spectrogram tests ───────────────────────────────────────


class TestAudioToSpectrogram:
    """Test audio_to_spectrogram: audio file → (1, N_MELS, time_frames) tensor."""

    def _make_wav(self, tmp_path, filename="test.wav", duration_s=2.0, sr=22050):
        """Generate a synthetic WAV file with a sine wave."""
        import numpy as np
        import wave
        import struct

        n_samples = int(sr * duration_s)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)
        # 440 Hz sine wave
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        path = tmp_path / filename
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{n_samples}h", *audio))
        return str(path)

    def test_output_shape(self, tmp_path):
        from model.preprocess import audio_to_spectrogram
        from model.config import N_MELS, SAMPLE_RATE, CLIP_LENGTH_S, HOP_LENGTH

        wav_path = self._make_wav(tmp_path, duration_s=4.0, sr=SAMPLE_RATE)
        tensor = audio_to_spectrogram(wav_path)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 1        # channels
        assert tensor.shape[1] == N_MELS   # mel bins

        expected_time = int(SAMPLE_RATE * CLIP_LENGTH_S) // HOP_LENGTH + 1
        assert tensor.shape[2] == expected_time

    def test_short_clip_is_padded(self, tmp_path):
        """A clip shorter than CLIP_LENGTH_S should be zero-padded to same shape."""
        from model.preprocess import audio_to_spectrogram
        from model.config import N_MELS, SAMPLE_RATE, CLIP_LENGTH_S, HOP_LENGTH

        wav_path = self._make_wav(tmp_path, filename="short.wav", duration_s=1.0)
        tensor = audio_to_spectrogram(wav_path)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == N_MELS
        expected_time = int(SAMPLE_RATE * CLIP_LENGTH_S) // HOP_LENGTH + 1
        assert tensor.shape[2] == expected_time

    def test_long_clip_is_trimmed(self, tmp_path):
        """A clip longer than CLIP_LENGTH_S should be trimmed."""
        from model.preprocess import audio_to_spectrogram
        from model.config import N_MELS, SAMPLE_RATE, CLIP_LENGTH_S, HOP_LENGTH

        wav_path = self._make_wav(tmp_path, filename="long.wav", duration_s=10.0)
        tensor = audio_to_spectrogram(wav_path)

        expected_time = int(SAMPLE_RATE * CLIP_LENGTH_S) // HOP_LENGTH + 1
        assert tensor.shape == (1, N_MELS, expected_time)

    def test_resamples_different_sr(self, tmp_path):
        """Audio at a different sample rate should be resampled to SAMPLE_RATE."""
        from model.preprocess import audio_to_spectrogram
        from model.config import N_MELS, SAMPLE_RATE, CLIP_LENGTH_S, HOP_LENGTH

        # Create a wav at 44100 Hz instead of 22050
        wav_path = self._make_wav(
            tmp_path, filename="resample.wav", duration_s=4.0, sr=44100
        )
        tensor = audio_to_spectrogram(wav_path)

        expected_time = int(SAMPLE_RATE * CLIP_LENGTH_S) // HOP_LENGTH + 1
        assert tensor.shape == (1, N_MELS, expected_time)

    def test_output_is_float_tensor(self, tmp_path):
        import torch
        from model.preprocess import audio_to_spectrogram

        wav_path = self._make_wav(tmp_path, duration_s=4.0)
        tensor = audio_to_spectrogram(wav_path)
        assert tensor.dtype == torch.float32


# ── CoughDataset tests ──────────────────────────────────────────────


class TestCoughDataset:
    """Test CoughDataset: loads .pt tensors, applies normalization, returns (tensor, label)."""

    def _make_pt_files(self, tmp_path, n=5, shape=(1, 64, 173)):
        """Create n dummy .pt spectrogram files and return (file_list, label_list)."""
        import torch

        file_list = []
        label_list = []
        for i in range(n):
            tensor = torch.randn(*shape)
            path = str(tmp_path / f"sample_{i}.pt")
            torch.save(tensor, path)
            file_list.append(path)
            label_list.append(i % 2)  # alternate 0/1
        return file_list, label_list

    def test_len_returns_correct_count(self, tmp_path):
        from model.dataset import CoughDataset

        file_list, label_list = self._make_pt_files(tmp_path, n=7)
        ds = CoughDataset(file_list, label_list)
        assert len(ds) == 7

    def test_getitem_returns_tensor_and_float32_label(self, tmp_path):
        import torch
        from model.dataset import CoughDataset

        file_list, label_list = self._make_pt_files(tmp_path, n=3)
        ds = CoughDataset(file_list, label_list)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert label.dtype == torch.float32
        assert label.item() == float(label_list[0])

    def test_works_with_dataloader(self, tmp_path):
        import torch
        from torch.utils.data import DataLoader
        from model.dataset import CoughDataset

        shape = (1, 64, 173)
        file_list, label_list = self._make_pt_files(tmp_path, n=6, shape=shape)
        ds = CoughDataset(file_list, label_list)
        loader = DataLoader(ds, batch_size=3, shuffle=False)
        batch_tensor, batch_label = next(iter(loader))
        assert batch_tensor.shape == (3, *shape)
        assert batch_label.shape == (3,)
        assert batch_label.dtype == torch.float32

    def test_applies_normalization(self, tmp_path):
        import torch
        from model.dataset import CoughDataset

        # Create a known tensor
        known = torch.ones(1, 64, 173) * 10.0
        path = str(tmp_path / "known.pt")
        torch.save(known, path)

        mean = 10.0
        std = 2.0
        ds = CoughDataset([path], [1], mean=mean, std=std)
        tensor, label = ds[0]
        # (10 - 10) / 2 = 0.0
        assert torch.allclose(tensor, torch.zeros(1, 64, 173), atol=1e-6)

    def test_no_normalization_without_mean_std(self, tmp_path):
        import torch
        from model.dataset import CoughDataset

        known = torch.ones(1, 64, 173) * 5.0
        path = str(tmp_path / "raw.pt")
        torch.save(known, path)

        ds = CoughDataset([path], [0])
        tensor, label = ds[0]
        assert torch.allclose(tensor, torch.ones(1, 64, 173) * 5.0)


# ── SickNoteResNet tests ─────────────────────────────────────────────


class TestSickNoteResNet:
    """Test SickNoteResNet: output shape, various input sizes, Grad-CAM target."""

    def test_output_shape_batch_4(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(4, 1, 64, 173)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_batch_1(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(1, 1, 64, 173)
        out = model(x)
        assert out.shape == (1, 1)

    def test_output_shape_batch_16(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(16, 1, 64, 173)
        out = model(x)
        assert out.shape == (16, 1)

    def test_different_input_dimensions(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(2, 1, 128, 87)
        out = model(x)
        assert out.shape == (2, 1)

    def test_outputs_raw_logits(self):
        """Model should output raw logits, not probabilities (no sigmoid)."""
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        model.eval()
        x = torch.randn(4, 1, 64, 173)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_has_layer4_for_gradcam(self):
        """model.layer4 must exist for Grad-CAM hooks."""
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        assert hasattr(model, "layer4")

    def test_backbone_and_head_params(self):
        """backbone_params() and head_params() must return non-empty iterators."""
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        backbone = list(model.backbone_params())
        head = list(model.head_params())
        assert len(backbone) > 0
        assert len(head) > 0
        assert len(head) < len(backbone)

    def test_conv1_is_single_channel(self):
        """conv1 must accept 1-channel input, not 3-channel."""
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        assert model.conv1.in_channels == 1
