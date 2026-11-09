"""Preprocessing helpers."""

import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X, float) * self.std_ + self.mean_


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, float)
        self.min_ = X.min(axis=0)
        data_range = X.max(axis=0) - self.min_
        data_range[data_range == 0] = 1.0
        lo, hi = self.feature_range
        self.scale_ = (hi - lo) / data_range
        self.offset_ = lo - self.min_ * self.scale_
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return X * self.scale_ + self.offset_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    split = int(n * (1 - test_size))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], y[tr], y[te]


def one_hot(y, num_classes=None):
    y = np.asarray(y)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y.astype(int)] = 1
    return out


def add_bias(X):
    X = np.asarray(X, float)
    return np.hstack([np.ones((X.shape[0], 1)), X])
