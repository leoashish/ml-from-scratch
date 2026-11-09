"""Support Vector Machine from first principles.

- LinearSVM: hinge loss minimized with subgradient descent.
- KernelSVM: dual formulation solved with a simplified SMO (RBF/poly/linear).
"""

import numpy as np


class KernelSVM:
    """Kernel SVM solved with simplified SMO. Supports 'linear', 'rbf', 'poly'."""

    def __init__(self, C=1.0, kernel="rbf", gamma=None, degree=3, coef0=0.0,
                 tol=1e-3, max_passes=5, random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state
        self.alphas_ = None
        self.b_ = 0.0
        self.X_ = None
        self.y_ = None
        self.classes_ = None

    def _kernel_func(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "rbf":
            if self.gamma is None:
                self.gamma = 1.0 / X1.shape[1]
            # squared dist
            sq1 = np.sum(X1 ** 2, axis=1)[:, None]
            sq2 = np.sum(X2 ** 2, axis=1)[None, :]
            return np.exp(-self.gamma * (sq1 + sq2 - 2 * X1 @ X2.T))
        if self.kernel == "poly":
            return (X1 @ X2.T + self.coef0) ** self.degree
        raise ValueError("unknown kernel")

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("binary only")
        self.classes_ = classes
        yb = np.where(y == classes[1], 1.0, -1.0)
        n = len(y)
        self.X_ = X
        self.y_ = yb
        if self.gamma is None and self.kernel == "rbf":
            self.gamma = 1.0 / X.shape[1]
        K = self._kernel_func(X, X)
        alphas = np.zeros(n)
        b = 0.0
        passes = 0
        rng = np.random.default_rng(self.random_state)
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                # error
                Ei = (alphas * yb) @ K[:, i] + b - yb[i]
                if (yb[i] * Ei < -self.tol and alphas[i] < self.C) or (yb[i] * Ei > self.tol and alphas[i] > 0):
                    # pick j
                    j = i
                    while j == i:
                        j = int(rng.integers(0, n))
                    Ej = (alphas * yb) @ K[:, j] + b - yb[j]
                    ai_old, aj_old = alphas[i], alphas[j]
                    if yb[i] != yb[j]:
                        L = max(0.0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0.0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    if L >= H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    alphas[j] -= yb[j] * (Ei - Ej) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    if abs(alphas[j] - aj_old) < 1e-5:
                        continue
                    alphas[i] += yb[i] * yb[j] * (aj_old - alphas[j])
                    # update b
                    b1 = b - Ei - yb[i] * (alphas[i] - ai_old) * K[i, i] - yb[j] * (alphas[j] - aj_old) * K[i, j]
                    b2 = b - Ej - yb[i] * (alphas[i] - ai_old) * K[i, j] - yb[j] * (alphas[j] - aj_old) * K[j, j]
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0
        self.alphas_ = alphas
        self.b_ = float(b)
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        K = self._kernel_func(X, self.X_)
        return K @ (self.alphas_ * self.y_) + self.b_

    def predict(self, X):
        df = self.decision_function(X)
        return np.where(df >= 0, self.classes_[1], self.classes_[0])
