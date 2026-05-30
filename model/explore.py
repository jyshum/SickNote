"""
Run FIRST before writing any model code.
Answers dataset questions that feed into config.py.

See ARCHITECTURE.md → P1 Step 0 for the 6 questions this must answer.

Usage: python -m model.explore
"""


def majority_label(row):
    """Compute binary label from expert diagnosis_1-4 columns via majority vote.

    Returns 'healthy' or 'abnormal', or None if no expert labels exist.
    Label mapping: healthy_cough → healthy | everything else → abnormal
    """
    raise NotImplementedError


def has_good_quality(row):
    """Return False if majority of experts rate quality as 'poor' or 'no_cough'."""
    raise NotImplementedError


def main():
    """Print answers to all 6 dataset questions. See ARCHITECTURE.md."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
