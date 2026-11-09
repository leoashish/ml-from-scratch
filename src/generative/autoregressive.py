"""Autoregressive model from first principles.

Learns P(x_t | x_{<t}) and generates new sequences one step at a time.
Implemented as a character-level model: context window -> next char distribution.
"""

import numpy as np


class AutoregressiveCharModel:
    def __init__(self, context=4, hidden=32, lr=0.1, n_iters=500,
                 batch_size=64, random_state=None):
        self.context = context
        self.hidden = hidden
        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.random_state = random_state
        self.chars_ = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.V = None

    def _build_vocab(self, text):
        self.chars_ = sorted(set(text))
        self.V = len(self.chars_)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars_)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars_)}

    def _one_hot(self, idx):
        v = np.zeros(self.V)
        v[idx] = 1.0
        return v

    def _make_data(self, text):
        ids = [self.char_to_idx[c] for c in text]
        X = []
        y = []
        for i in range(self.context, len(ids)):
            ctx = ids[i - self.context:i]
            # input: concatenated one-hots
            vec = np.concatenate([self._one_hot(j) for j in ctx])
            X.append(vec)
            y.append(ids[i])
        return np.array(X), np.array(y)

    def fit(self, text):
        self._build_vocab(text)
        X, y = self._make_data(text)
        n = len(X)
        in_dim = self.context * self.V
        rng = np.random.default_rng(self.random_state)
        self.W1 = rng.standard_normal((in_dim, self.hidden)) * 0.01
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.standard_normal((self.hidden, self.V)) * 0.01
        self.b2 = np.zeros(self.V)
        for it in range(self.n_iters):
            idx = rng.integers(0, n, size=min(self.batch_size, n))
            Xb = X[idx]
            yb = y[idx]
            # forward
            h_pre = Xb @ self.W1 + self.b1
            h = np.tanh(h_pre)
            logits = h @ self.W2 + self.b2
            # stable softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            # loss grad
            # one-hot targets
            Yoh = np.zeros_like(probs)
            Yoh[np.arange(len(yb)), yb] = 1.0
            diff = (probs - Yoh) / len(yb)
            # backprop
            dW2 = h.T @ diff
            db2 = diff.sum(axis=0)
            dh = diff @ self.W2.T
            dh = dh * (1 - np.tanh(h_pre) ** 2)
            dW1 = Xb.T @ dh
            db1 = dh.sum(axis=0)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
        return self

    def _predict_next(self, context_ids, rng):
        vec = np.concatenate([self._one_hot(j) for j in context_ids])
        h = np.tanh(vec @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        return rng.choice(self.V, p=probs)

    def generate(self, seed_text, length=100, temperature=1.0, seed=None):
        rng = np.random.default_rng(seed)
        # pad seed if shorter
        ctx = [self.char_to_idx.get(c, 0) for c in seed_text]
        if len(ctx) < self.context:
            ctx = [0] * (self.context - len(ctx)) + ctx
        ctx = ctx[-self.context:]
        out = list(seed_text)
        for _ in range(length):
            # temperature sampling
            vec = np.concatenate([self._one_hot(j) for j in ctx])
            h = np.tanh(vec @ self.W1 + self.b1)
            logits = (h @ self.W2 + self.b2) / max(temperature, 1e-6)
            logits -= logits.max()
            exp = np.exp(logits)
            probs = exp / exp.sum()
            nxt = rng.choice(self.V, p=probs)
            out.append(self.idx_to_char[nxt])
            ctx = ctx[1:] + [nxt]
        return "".join(out)
