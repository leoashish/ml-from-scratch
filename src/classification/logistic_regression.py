"""Logistic regression from first principles.

Uses gradient descent on binary cross-entropy. Supports multiclass via
softmax (multinomial) or one-vs-rest.
"""

import numpy as np
from ..utils.preprocessing import add_bias


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000, l2=0.0, multi="softmax",
                 fit_intercept=True, random_state=None):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.multi = multi  # "softmax" or "ovr"
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        if K == 2:
            # binary
            y_bin = (y == self.classes_[1]).astype(float)
            w = np.zeros(d + (1 if self.fit_intercept else 0))
            # gradient descent
            Xb = add_bias(X) if self.fit_intercept else X.copy()
            for it in range(self.n_iters):
                z = Xb @ w
                p = self._sigmoid(z)
                grad = Xb.T @ (p - y_bin) / n
                if self.l2 > 0:
                    reg = (self.l2 / n) * w.copy()
                    if self.fit_intercept:
                        reg[0] = 0.0
                    grad += reg
                w -= self.lr * grad
                # loss
                eps = 1e-12
                loss = -np.mean(y_bin * np.log(p + eps) + (1 - y_bin) * np.log(1 - p + eps))
                self.loss_history_.append(float(loss))
            if self.fit_intercept:
                self.intercept_ = float(w[0])
                self.coef_ = w[1:].copy()
            else:
                self.intercept_ = 0.0
                self.coef_ = w.copy()
            self._binary = True
            self._w = w
            return self

        # multiclass
        self._binary = False
        if self.multi == "softmax":
            # multinomial logistic regression
            Y = np.zeros((n, K))
            for k, c in enumerate(self.classes_):
                Y[:, k] = (y == c).astype(float)
            # params: (d+1) x K
            W = np.zeros((d + (1 if self.fit_intercept else 0), K))
            Xb = add_bias(X) if self.fit_intercept else X.copy()
            for it in range(self.n_iters):
                Z = Xb @ W
                # stable softmax
                Z -= Z.max(axis=1, keepdims=True)
                P = np.exp(Z)
                P /= P.sum(axis=1, keepdims=True)
                grad = Xb.T @ (P - Y) / n
                if self.l2 > 0:
                    reg = (self.l2 / n) * W.copy()
                    if self.fit_intercept:
                        reg[0, :] = 0.0
                    grad += reg
                W -= self.lr * grad
                eps = 1e-12
                loss = -np.mean(np.sum(Y * np.log(P + eps), axis=1))
                self.loss_history_.append(float(loss))
            if self.fit_intercept:
                self.intercept_ = W[0].copy()
                self.coef_ = W[1:].copy()  # shape (d, K)
            else:
                self.intercept_ = np.zeros(K)
                self.coef_ = W.copy()
            self._W = W
            return self
        else:
            # one-vs-rest: train K binary classifiers
            self._ovr = []
            for k, c in enumerate(self.classes_):
                yb = (y == c).astype(float)
                lr = LogisticRegression(lr=self.lr, n_iters=self.n_iters, l2=self.l2,
                                        fit_intercept=self.fit_intercept, multi="softmax")
                # reuse binary path by forcing 2 classes? Simpler: fit a binary model manually
                w = self._fit_binary(X, yb)
                self._ovr.append(w)
            return self

    def _fit_binary(self, X, y_bin):
        X = np.asarray(X, float)
        y_bin = np.asarray(y_bin, float).ravel()
        n, d = X.shape
        Xb = add_bias(X) if self.fit_intercept else X.copy()
        w = np.zeros(Xb.shape[1])
        for it in range(self.n_iters):
            z = Xb @ w
            p = self._sigmoid(z)
            grad = Xb.T @ (p - y_bin) / n
            if self.l2 > 0:
                reg = (self.l2 / n) * w.copy()
                if self.fit_intercept:
                    reg[0] = 0.0
                grad += reg
            w -= self.lr * grad
        return w

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if getattr(self, "_binary", False):
            Xb = add_bias(X) if self.fit_intercept else X
            z = Xb @ self._w
            p = self._sigmoid(z)
            return np.column_stack([1 - p, p])
        if hasattr(self, "_ovr") and self._ovr:
            # ovr
            scores = np.zeros((len(X), len(self.classes_)))
            for k, w in enumerate(self._ovr):
                Xb = add_bias(X) if self.fit_intercept else X
                scores[:, k] = Xb @ w
            # softmax over scores
            scores -= scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            return exp / exp.sum(axis=1, keepdims=True)
        # softmax
        Xb = add_bias(X) if self.fit_intercept else X
        Z = Xb @ self._W
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        from ..utils.metrics import accuracy
        return accuracy(y, self.predict(X))
