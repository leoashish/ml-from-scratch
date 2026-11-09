"""Synthetic dataset generators for demos and tests."""

import numpy as np


def make_regression(n_samples=200, n_features=3, noise=10.0, random_state=None):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    true_coef = rng.standard_normal(n_features)
    y = X @ true_coef + 5.0 + rng.normal(0, noise, n_samples)
    return X, y


def make_classification(n_samples=200, n_features=2, n_classes=2, random_state=None):
    rng = np.random.default_rng(random_state)
    if n_classes == 2:
        # two gaussian clusters
        X0 = rng.standard_normal((n_samples // 2, n_features)) - 1.5
        X1 = rng.standard_normal((n_samples - n_samples // 2, n_features)) + 1.5
        X = np.vstack([X0, X1])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    else:
        centers = rng.standard_normal((n_classes, n_features)) * 3
        X = []
        y = []
        for c in range(n_classes):
            n = n_samples // n_classes
            X.append(rng.standard_normal((n, n_features)) + centers[c])
            y += [c] * n
        X = np.vstack(X)
        y = np.array(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.8, random_state=None):
    rng = np.random.default_rng(random_state)
    if isinstance(centers, int):
        centers = rng.standard_normal((centers, n_features)) * 5
    centers = np.asarray(centers, float)
    k = len(centers)
    sizes = [n_samples // k + (1 if i < n_samples % k else 0) for i in range(k)]
    X = []
    y = []
    for i, c in enumerate(centers):
        X.append(rng.standard_normal((sizes[i], n_features)) * cluster_std + c)
        y += [i] * sizes[i]
    X = np.vstack(X)
    y = np.array(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_moons(n_samples=200, noise=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.linspace(0, np.pi, n)
    X0 = np.column_stack([np.cos(theta), np.sin(theta)])
    X1 = np.column_stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5])
    X = np.vstack([X0, X1]) + rng.normal(0, noise, (n_samples, 2))
    y = np.array([0] * n + [1] * (n_samples - n))
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_time_series(n=200, trend=0.05, seasonality=10, noise=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    t = np.arange(n)
    y = trend * t + 5 * np.sin(2 * np.pi * t / seasonality) + rng.normal(0, noise, n)
    return y


def make_ranking(n_queries=50, n_items_per_query=10, n_features=5, random_state=None):
    rng = np.random.default_rng(random_state)
    X_list = []
    y_list = []
    qid_list = []
    for q in range(n_queries):
        X = rng.standard_normal((n_items_per_query, n_features))
        # relevance = function of features
        score = X @ rng.standard_normal(n_features)
        # discretize into 0..4 relevance
        ranks = np.argsort(np.argsort(score))
        rel = np.clip((ranks // (n_items_per_query // 5)), 0, 4).astype(int)
        X_list.append(X)
        y_list.append(rel)
        qid_list.append(np.full(n_items_per_query, q))
    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(qid_list)
