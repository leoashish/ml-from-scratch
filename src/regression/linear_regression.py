"""Linear regression from first principles.

Model: y = X w + b. We solve via (a) closed-form normal equation and
(b) gradient descent. Supports optional L2 (ridge) regularization.
"""

import numpy as np
from ..utils.preprocessing import add_bias


class LinearRegression:
    def __init__(self, method="normal", lr=0.01, n_iters=1000, l2=0.0, fit_intercept=True):
        self.method = method          # "normal" or "gd"
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_history_ = []

    def _prepare(self, X):
        X = np.asarray(X, float)
        if self.fit_intercept:
            Xb = add_bias(X)
        else:
            Xb = X.copy()
        return Xb

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        Xb = self._prepare(X)
        n, d = Xb.shape

        if self.method == "normal":
            # (X^T X + lambda I)^-1 X^T y  (ridge if l2>0)
            A = Xb.T @ Xb + self.l2 * np.eye(d)
            # don't regularize bias term
            if self.fit_intercept:
                A[0, 0] -= self.l2
            self.w_ = np.linalg.solve(A, Xb.T @ y)
        elif self.method == "gd":
            self.w_ = np.zeros(d)
            for it in range(self.n_iters):
                pred = Xb @ self.w_
                grad = (2.0 / n) * (Xb.T @ (pred - y))
                if self.l2 > 0:
                    reg = (2 * self.l2 / n) * self.w_
                    if self.fit_intercept:
                        reg[0] = 0.0
                    grad += reg
                self.w_ -= self.lr * grad
                self.loss_history_.append(float(np.mean((pred - y) ** 2)))
        else:
            raise ValueError("method must be 'normal' or 'gd'")

        if self.fit_intercept:
            self.intercept_ = float(self.w_[0])
            self.coef_ = self.w_[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.w_.copy()
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        from ..utils.metrics import r2_score
        return r2_score(y, self.predict(X))
