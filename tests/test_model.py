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
