"""Gradient Boosting for regression (and classification via log-loss).

Idea: build an additive model F(x) = sum of weak learners (shallow trees).
Each new tree fits the negative gradient (residuals) of the loss.
"""

import numpy as np


class DecisionTreeRegressor:
    """A simple regression tree (CART) with squared-error splits."""

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        node = {"n": len(y), "value": float(np.mean(y)) if len(y) else 0.0}
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or np.all(y == y[0])):
            return node
        best = self._best_split(X, y)
        if best is None:
            return node
        feat, thr = best
        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return node
        node["feature"] = int(feat)
        node["threshold"] = float(thr)
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        n, d = X.shape
        best_gain = 0.0  # require positive gain
        best = None
        parent_var = np.var(y) * n
        for feat in range(d):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[1:] + vals[:-1]) / 2
            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                gain = parent_var - (np.var(left) * len(left) + np.var(right) * len(right))
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (feat, thr)
        return best

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.zeros(len(X))
        for i in range(len(X)):
            out[i] = self._predict_one(X[i], self.tree_)
        return out

    def _predict_one(self, x, node):
        if "feature" not in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees_ = []
        self.init_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        rng = np.random.default_rng(self.random_state)
        self.init_ = float(np.mean(y))
        F = np.full_like(y, self.init_, dtype=float)
        for m in range(self.n_estimators):
            residual = y - F  # negative gradient of MSE
            # subsample
            if self.subsample < 1.0:
                idx = rng.choice(len(y), size=int(self.subsample * len(y)), replace=False)
            else:
                idx = np.arange(len(y))
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[idx], residual[idx])
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)
            self.loss_history_.append(float(np.mean((y - F) ** 2)))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        F = np.full(len(X), self.init_, dtype=float)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F


class GradientBoostingClassifier:
    """Binary classification via gradient boosting on log-loss."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.init_ = None

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        # convert to 0/1
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("binary only")
        y01 = (y == classes[1]).astype(float)
        # initial log odds
        p = np.clip(y01.mean(), 1e-6, 1 - 1e-6)
        self.init_ = float(np.log(p / (1 - p)))
        self.classes_ = classes
        F = np.full(len(y), self.init_, dtype=float)
        rng = np.random.default_rng(self.random_state)
        for m in range(self.n_estimators):
            p = self._sigmoid(F)
            residual = y01 - p  # negative gradient
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        F = np.full(len(X), self.init_, dtype=float)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        p = self._sigmoid(F)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
