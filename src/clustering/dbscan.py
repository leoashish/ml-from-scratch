"""DBSCAN clustering from first principles."""

import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None

    def _dist(self, X):
        if self.metric == "euclidean":
            sq = np.sum(X ** 2, axis=1)
            D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
            return D
        raise ValueError("metric not supported")

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        D = self._dist(X)
        labels = np.full(n, -2)  # -2 unvisited, -1 noise
        core = []
        cluster_id = 0
        for i in range(n):
            if labels[i] != -2:
                continue
            neighbors = np.where(D[i] <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue
            # core point
            core.append(i)
            labels[i] = cluster_id
            # expand
            seeds = list(neighbors)
            seeds = [s for s in seeds if s != i]
            while seeds:
                j = seeds.pop()
                if labels[j] == -1:
                    labels[j] = cluster_id
                if labels[j] != -2:
                    continue
                labels[j] = cluster_id
                nbrs = np.where(D[j] <= self.eps)[0]
                if len(nbrs) >= self.min_samples:
                    core.append(j)
                    for nb in nbrs:
                        if labels[nb] in (-2, -1):
                            seeds.append(nb)
            cluster_id += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core)
        # remap noise stays -1
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
