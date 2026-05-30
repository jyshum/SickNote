"""
Evaluate trained model on held-out test set.

Reports: AUC-ROC, accuracy, sensitivity, specificity, confusion matrix.
See ARCHITECTURE.md → P1 Step 4 for target thresholds.

Usage: python -m model.evaluate
"""


def main():
    """Load model_final.pt, run on test split, print metrics.

    Must call model.eval() and use torch.no_grad().
    Apply torch.sigmoid() to raw logits before computing metrics.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
