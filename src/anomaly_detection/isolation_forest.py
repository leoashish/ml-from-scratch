"""Isolation Forest from first principles.

Anomalies are easier to isolate (shorter path length in random trees).
"""

import numpy as np


class _IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, depth=0):
        X = np.asarray(X, float)
        self.root = self._build(X, depth)
        return self

    def _build(self, X, depth):
        n, d = X.shape
        if depth >= self.max_depth or n <= 1:
            return {"type": "leaf", "size": n, "depth": depth}
        feat = np.random.randint(0, d)
        vals = X[:, feat]
        mn, mx = vals.min(), vals.max()
        if mn == mx:
            return {"type": "leaf", "size": n, "depth": depth}
        thr = np.random.uniform(mn, mx)
        left_mask = vals < thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"type": "leaf", "size": n, "depth": depth}
        return {
            "type": "split",
            "feature": int(feat),
            "threshold": float(thr),
            "left": self._build(X[left_mask], depth + 1),
            "right": self._build(X[right_mask], depth + 1),
        }

    def path_length(self, x, node=None, depth=0):
        if node is None:
            node = self.root
        if node["type"] == "leaf":
            # adjustment for leaf size
            size = node["size"]
            if size <= 1:
                return depth
            # average path length of unsuccessful search in BST
            H = np.log(size - 1) + 0.5772156649 if size > 1 else 0
            return depth + H  # approximate; standard uses c(size)
            # Actually standard: path length contribution = c(size)
            # We'll handle c in forest. Return depth + c(size)
        if x[node["feature"]] < node["threshold"]:
            return self.path_length(x, node["left"], depth + 1)
        return self.path_length(x, node["right"], depth + 1)


def _c(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    H = np.log(n - 1) + 0.5772156649
    return 2 * H - 2 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees_ = []
        self.threshold_ = None

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        max_samples = min(self.max_samples, n)
        self.trees_ = []
        for t in range(self.n_estimators):
            idx = rng.choice(n, size=max_samples, replace=False)
            tree = _IsolationTree(max_depth=int(np.ceil(np.log2(max_samples))))
            tree.fit(X[idx])
            self.trees_.append((tree, max_samples))
        # compute scores on training to set threshold
        scores = self.decision_function(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def _path_length(self, x):
        lengths = []
        for tree, ms in self.trees_:
            # standard c(ms)
            pl = tree.path_length(x)
            lengths.append(pl)
        return np.mean(lengths)

    def decision_function(self, X):
        X = np.asarray(X, float)
        c_ms = _c(self.trees_[0][1])
        out = []
        for x in X:
            pl = self._path_length(x)
            score = 2 ** (-pl / c_ms)
            out.append(score)
        return np.array(out)

    def predict(self, X):
        scores = self.decision_function(X)
        # anomaly = 1 (or -1). Use sklearn convention: -1 anomaly, 1 normal
        return np.where(scores >= self.threshold_, -1, 1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
