"""K-Means clustering from first principles."""

import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4,
                 init="kmeans++", n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centers(self, X, rng):
        n, d = X.shape
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()
        # kmeans++
        centers = [X[rng.integers(0, n)]]
        for _ in range(1, self.n_clusters):
            # distances to nearest center
            D = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = D / D.sum()
            i = rng.choice(n, p=probs)
            centers.append(X[i])
        return np.array(centers)

    def _run_once(self, X, rng):
        centers = self._init_centers(X, rng)
        for it in range(self.max_iters):
            # assign
            dists = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(dists, axis=1)
            # update
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                pts = X[labels == k]
                if len(pts) > 0:
                    new_centers[k] = pts.mean(axis=0)
            shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
            centers = new_centers
            if shift < self.tol:
                break
        dists = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(dists, axis=1)
        inertia = sum(np.sum((X[labels == k] - centers[k]) ** 2) for k in range(self.n_clusters))
        return centers, labels, inertia

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            centers, labels, inertia = self._run_once(X, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        dists = np.sqrt(((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2))
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
