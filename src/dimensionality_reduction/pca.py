"""Principal Component Analysis from first principles."""

import numpy as np


class PCA:
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # covariance
        # use SVD for stability
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        # eigenvalues of covariance
        n = X.shape[0]
        explained = (s ** 2) / (n - 1)
        total = explained.sum()
        ratio = explained / total
        if self.n_components is None:
            self.n_components = X.shape[1]
        k = min(self.n_components, X.shape[1])
        self.components_ = Vt[:k]
        self.explained_variance_ = explained[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.singular_values_ = s[:k]
        # store std for whiten
        self._std = np.sqrt(self.explained_variance_)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Xc = X - self.mean_
        out = Xc @ self.components_.T
        if self.whiten:
            out = out / self._std
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, float)
        out = X @ self.components_
        if self.whiten:
            out = out * self._std
        return out + self.mean_
