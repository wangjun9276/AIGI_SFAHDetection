import math
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def _nan():
    return float("nan")


def compute_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    real_mask, fake_mask = labels == 0, labels == 1
    real_acc = accuracy_score(labels[real_mask], predictions[real_mask]) if real_mask.any() else _nan()
    fake_acc = accuracy_score(labels[fake_mask], predictions[fake_mask]) if fake_mask.any() else _nan()
    balanced = np.nanmean([real_acc, fake_acc])
    both_classes = np.unique(labels).size == 2
    auc = roc_auc_score(labels, probabilities) if both_classes else _nan()
    ap = average_precision_score(labels, probabilities) if fake_mask.any() else _nan()
    eer = _nan()
    if both_classes:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2)
    return {
        "num_samples": int(labels.size), "num_real": int(real_mask.sum()), "num_fake": int(fake_mask.sum()),
        "accuracy": float(accuracy_score(labels, predictions)), "balanced_accuracy": float(balanced),
        "ap": float(ap), "auc": float(auc), "eer": float(eer),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "real_accuracy": float(real_acc), "fake_accuracy": float(fake_acc), "threshold": float(threshold),
    }
