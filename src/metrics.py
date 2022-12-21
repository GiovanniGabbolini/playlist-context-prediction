"""
Created on Thu May 1 2022

@author Name Redacted Surname Redacted
"""
from sklearn.metrics import top_k_accuracy_score
import numpy as np


def compute_metrics(logits, label):
    """
    Logits: predicted probability of all labels;
    Label: Actual true label.
    """
    assert all(label < logits.shape[1])

    logits_np = logits.cpu().detach().numpy()
    label_np = label.cpu().detach().numpy()

    return {
        "FH@1": float(_fh(logits_np, label_np, 1)),
        "FH@5": float(_fh(logits_np, label_np, 5)),
        "MRR": float(_mrr(logits_np, label_np)),
        "MAP@5": float(_map(logits_np, label_np, 5)),
    }


def _fh(logits, label, k):
    """
    Flat hits @ k, or top-k accuracy [1].

    [1]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html#sklearn.metrics.top_k_accuracy_score
    """
    labels = list(range(logits.shape[1]))
    return top_k_accuracy_score(label, logits, k=k, labels=labels)


def _mrr(logits, label, return_avg=True):
    """
    In this multi-label classification example, MRR is the position in which we put the right label, on average.
    """
    # Position on logit's ranking of the true label.
    rank = np.argsort(np.argsort(-logits, axis=1), axis=1)[np.arange(logits.shape[0]), label]
    rank = rank+1

    # Computing MRR
    rr = 1/rank

    return np.mean(rr) if return_avg else rr


def _map(logits, label, k, return_avg=True):
    """
    In this multi-label classification example, 
    MAP@k is the avg the following vector:
    - position in which we put the right label, if < k, and 0 otherwise.
    """
    # Position on logit's ranking of the true label.
    rank = np.argsort(np.argsort(-logits, axis=1), axis=1)[np.arange(logits.shape[0]), label]
    rank = rank+1

    # Computing MAP@5
    # In multi-label classification, MAP@l is equal to MRR, but dropping all ranking > k.
    rr = 1/rank
    rr[rank > k] = 0

    return np.mean(rr) if return_avg else rr
