"""
Evaluation metrics for deepfake detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        y_true: True labels (0 or 1)
        y_scores: Prediction scores (probabilities)
        
    Returns:
        EER value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    eer = compute_eer(y_true, y_scores)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'eer': float(eer),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Print metrics in a formatted way."""
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"EER:       {metrics['eer']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
    print("="*50 + "\n")

