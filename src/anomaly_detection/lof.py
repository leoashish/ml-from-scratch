"""Local Outlier Factor from first principles."""

import numpy as np


class LocalOutlierFactor:
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.negative_outlier_factor_ = None

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        k = min(self.n_neighbors, n - 1)
        sq = np.sum(X ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
        knn_dist = np.zeros(n)
        knn_idx = []
        for i in range(n):
            order = np.argsort(D[i])
            order = order[order != i]
            nbrs = order[:k]
            knn_idx.append(nbrs)
            knn_dist[i] = D[i, nbrs[-1]] if len(nbrs) else 0.0
        lrd = np.zeros(n)
        for i in range(n):
            rd_sum = 0.0
            for j in knn_idx[i]:
                rd = max(knn_dist[j], D[i, j])
                rd_sum += rd
            lrd[i] = k / rd_sum if rd_sum > 0 else 0.0
        lof = np.zeros(n)
        for i in range(n):
            if lrd[i] == 0:
                lof[i] = 1.0
                continue
            ratios = [lrd[j] / lrd[i] for j in knn_idx[i]]
            lof[i] = float(np.mean(ratios))
        self._X = X
        self._D = D
        self._k = k
        self._knn_dist = knn_dist
        self._knn_idx = knn_idx
        self._lrd = lrd
        self._lof = lof
        self.negative_outlier_factor_ = -lof
        self.threshold_ = float(np.percentile(lof, 100 * (1 - self.contamination)))
        return self

    def fit_predict(self, X):
        self.fit(X)
        return np.where(self._lof >= self.threshold_, -1, 1)
