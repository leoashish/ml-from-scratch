"""Denoising Diffusion Probabilistic Model (DDPM) from first principles.

Educational implementation on small 2D data. A tiny MLP predicts the noise
added at a given timestep; sampling reverses the diffusion process.
"""

import numpy as np


def _relu(x):
    return np.maximum(0, x)


class DiffusionModel:
    def __init__(self, data_dim=2, hidden=64, T=50, beta_start=1e-4,
                 beta_end=0.02, lr=1e-3, n_iters=2000, batch_size=64,
                 random_state=None):
        self.data_dim = data_dim
        self.hidden = hidden
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.random_state = random_state
        # beta schedule
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)
        # model params: MLP input (x_dim + 1 time) -> hidden -> hidden -> x_dim
        rng = np.random.default_rng(random_state)
        in_dim = data_dim + 1
        self.W1 = rng.standard_normal((in_dim, hidden)) * 0.01
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * 0.01
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((hidden, data_dim)) * 0.01
        self.b3 = np.zeros(data_dim)

    def _forward(self, x, t):
        # t: array (batch,) or scalar; normalize to [0,1]
        tnorm = (t / self.T).reshape(-1, 1) if np.ndim(t) > 0 else np.array([[t / self.T]])
        if np.ndim(t) == 0:
            tnorm = np.full((x.shape[0], 1), t / self.T)
        inp = np.column_stack([x, tnorm])
        h1 = _relu(inp @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        rng = np.random.default_rng(self.random_state + 7 if self.random_state is not None else 1)
        for it in range(self.n_iters):
            idx = rng.integers(0, n, size=self.batch_size)
            x0 = X[idx]
            t = rng.integers(1, self.T, size=self.batch_size)
            eps = rng.standard_normal(x0.shape)
            # x_t
            ab = self.alpha_bar[t - 1]
            x_t = np.sqrt(ab)[:, None] * x0 + np.sqrt(1 - ab)[:, None] * eps
            # predict eps
            pred = self._forward(x_t, t)
            # grad
            diff = (pred - eps) * 2 / self.batch_size
            # backprop
            # recompute intermediates
            tnorm = (t / self.T).reshape(-1, 1)
            inp = np.column_stack([x_t, tnorm])
            h1_pre = inp @ self.W1 + self.b1
            h1 = _relu(h1_pre)
            h2_pre = h1 @ self.W2 + self.b2
            h2 = _relu(h2_pre)
            out_pre = h2 @ self.W3 + self.b3
            # grads
            dW3 = h2.T @ diff
            db3 = diff.sum(axis=0)
            dh2 = diff @ self.W3.T
            dh2 = dh2 * (h2_pre > 0)
            dW2 = h1.T @ dh2
            db2 = dh2.sum(axis=0)
            dh1 = dh2 @ self.W2.T
            dh1 = dh1 * (h1_pre > 0)
            dW1 = inp.T @ dh1
            db1 = dh1.sum(axis=0)
            # update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
        return self

    def sample(self, n_samples=200, seed=None):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_samples, self.data_dim))
        for t in range(self.T, 0, -1):
            z = rng.standard_normal(x.shape) if t > 1 else np.zeros_like(x)
            ab = self.alpha_bar[t - 1]
            ab_prev = self.alpha_bar[t - 2] if t > 1 else 1.0
            beta_t = self.betas[t - 1]
            alpha_t = self.alphas[t - 1]
            # predict noise
            eps_pred = self._forward(x, np.full(n_samples, t))
            # mean
            coef = (1 - alpha_t) / np.sqrt(1 - ab)
            mean = (x - coef * eps_pred) / np.sqrt(alpha_t)
            # variance
            var = beta_t * (1 - ab_prev) / (1 - ab)
            x = mean + np.sqrt(var) * z
        return x
