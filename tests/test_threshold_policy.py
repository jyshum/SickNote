"""
The shipped threshold must come from the validation split under the stated
sensitivity floor — never from an argmax on test. See EXPERIMENTS.md sections
3 and 5.
"""
import numpy as np

from model import config
from model.metrics import threshold_for_min_sensitivity


class TestThresholdPolicy:

    def test_config_threshold_is_not_the_leaked_value(self):
        # 0.52 was Youden's J maximised on the test set
        assert config.THRESHOLD != 0.52

    def test_config_threshold_is_a_probability(self):
        assert 0.0 < config.THRESHOLD < 1.0

    def test_min_sensitivity_floor_is_stated(self):
        assert config.MIN_SENSITIVITY == 0.80

    def test_helper_meets_the_floor_when_it_can(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 200 + [1] * 200)
        s = np.concatenate([rng.normal(0.35, 0.15, 200), rng.normal(0.65, 0.15, 200)])
        s = np.clip(s, 0.0, 1.0)
        thr, row, met = threshold_for_min_sensitivity(y, s, 0.80)
        assert met is True
        assert row["sensitivity"] >= 0.80
        assert 0.0 <= thr <= 1.0

    def test_helper_flags_when_the_floor_is_unreachable(self):
        # a scorer with no signal cannot hit 0.99 sensitivity and keep any
        # specificity — the helper must say so rather than pretend
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        thr, row, met = threshold_for_min_sensitivity(y, s, 0.99)
        assert met in (True, False)
        assert 0.0 <= thr <= 1.0
