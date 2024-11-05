import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def roc_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0

def evaluate_metrics(y_true, y_pred, y_scores):
    eval_metrics = {}
    eval_metrics['precision'] = precision(y_true, y_pred)
    eval_metrics['recall'] = recall(y_true, y_pred)
    eval_metrics['f1_score'] = f1_score(y_true, y_pred)
    eval_metrics['roc_auc'] = roc_auc(y_true, y_scores)
    eval_metrics['false_positive_rate'] = false_positive_rate(y_true, y_pred)
    eval_metrics['false_negative_rate'] = false_negative_rate(y_true, y_pred)
    return eval_metrics