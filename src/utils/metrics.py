"""Evaluation metrics implemented from scratch."""

import numpy as np


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = list(labels)
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm, labels


def silhouette_score(X, labels):
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    n = X.shape[0]
    unique = [l for l in set(labels.tolist()) if l != -1]
    if len(unique) < 2:
        return 0.0
    # pairwise euclidean
    # compute a(i): mean intra-cluster distance
    # b(i): min mean inter-cluster distance
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    s = 0.0
    count = 0
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a = dists[i, same].mean()
        b = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.sum() == 0:
                continue
            b = min(b, dists[i, mask].mean())
        if b == np.inf:
            continue
        s += (b - a) / max(a, b)
        count += 1
    return float(s / count) if count > 0 else 0.0


def ndcg_at_k(y_true_relevance, y_pred_scores, k=10):
    # y_true_relevance: array of relevance per item (ground truth)
    # y_pred_scores: predicted scores; sort descending
    order = np.argsort(y_pred_scores)[::-1][:k]
    rels = np.asarray(y_true_relevance)[order]
    gains = (2 ** rels - 1)
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    dcg = np.sum(gains * discounts)
    # ideal
    ideal_order = np.argsort(y_true_relevance)[::-1][:k]
    ideal_rels = np.asarray(y_true_relevance)[ideal_order]
    ideal_gains = (2 ** ideal_rels - 1)
    idcg = np.sum(ideal_gains * discounts[:len(ideal_rels)])
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def auc_roc(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, float)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U
    count = 0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(count / (len(pos) * len(neg)))
