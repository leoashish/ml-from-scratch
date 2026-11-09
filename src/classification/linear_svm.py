"""Support Vector Machine from first principles.

- LinearSVM: hinge loss minimized with subgradient descent.
- KernelSVM: dual formulation solved with a simplified SMO (RBF/poly/linear).
"""

import numpy as np


class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, n_iters=1000, fit_intercept=True):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.w_ = None
        self.b_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        # map labels to +-1
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("binary only")
        yb = np.where(y == classes[1], 1.0, -1.0)
        self.classes_ = classes
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        for it in range(self.n_iters):
            for i in range(n):
                margin = yb[i] * (X[i] @ w + b)
                if margin < 1:
                    w -= self.lr * (w / n - self.C * yb[i] * X[i])  # approx
                    b += self.lr * self.C * yb[i]
                else:
                    w -= self.lr * (w / n)
            # Actually use vectorized update below is more stable; redo properly:
        # Redo with vectorized hinge subgradient for stability
        w = np.zeros(d)
        b = 0.0
        for it in range(self.n_iters):
            margins = yb * (X @ w + b)
            mask = margins < 1
            dw = w / n - self.C * (X[mask].T @ yb[mask]) / n if mask.any() else w / n
            # careful: gradient of (1/n)sum hinge + (1/(2nC?))... use standard:
            dw = w.copy()  # reg term (lambda)
            # We'll use: minimize 0.5||w||^2 + C sum hinge, grad = w - C sum_{viol} y_i x_i
            dw = w - self.C * (X[mask].T @ yb[mask]) if mask.any() else w.copy()
            db = -self.C * np.sum(yb[mask]) if mask.any() else 0.0
            w -= self.lr * dw
            b -= self.lr * db
        self.w_ = w
        self.b_ = float(b)
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        return X @ self.w_ + self.b_

    def predict(self, X):
        df = self.decision_function(X)
        return np.where(df >= 0, self.classes_[1], self.classes_[0])


