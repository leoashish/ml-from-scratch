"""Autoencoder (neural network) from first principles.

A symmetric MLP that compresses input to a latent code and reconstructs.
Trained with MSE loss via backpropagation (manual gradients).
"""

import numpy as np


def _relu(x):
    return np.maximum(0, x)


def _relu_grad(x):
    return (x > 0).astype(float)


class Autoencoder:
    def __init__(self, input_dim, hidden_dims=(64, 32), latent_dim=2,
                 lr=0.01, n_iters=200, batch_size=32, random_state=None):
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.random_state = random_state
        self._build()

    def _build(self):
        rng = np.random.default_rng(self.random_state)
        # encoder layers: input -> hidden... -> latent
        dims = [self.input_dim] + self.hidden_dims + [self.latent_dim]
        self.enc_W = []
        self.enc_b = []
        for i in range(len(dims) - 1):
            W = rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i + 1])
            self.enc_W.append(W)
            self.enc_b.append(b)
        # decoder: latent -> hidden (reversed) -> input
        dec_dims = [self.latent_dim] + list(reversed(self.hidden_dims)) + [self.input_dim]
        self.dec_W = []
        self.dec_b = []
        for i in range(len(dec_dims) - 1):
            W = rng.standard_normal((dec_dims[i], dec_dims[i + 1])) * np.sqrt(2.0 / dec_dims[i])
            b = np.zeros(dec_dims[i + 1])
            self.dec_W.append(W)
            self.dec_b.append(b)

    def _encode(self, X):
        # returns latent + intermediates for backprop
        acts = [X]
        zs = []
        a = X
        for W, b in zip(self.enc_W, self.enc_b):
            z = a @ W + b
            zs.append(z)
            a = _relu(z) if (W is not self.enc_W[-1]) else z  # linear latent
            acts.append(a)
        # last layer linear
        # fix: ensure last activation is z (linear)
        # recompute last properly
        return a, acts, zs

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        rng = np.random.default_rng(self.random_state + 1 if self.random_state is not None else 0)
        for it in range(self.n_iters):
            # mini-batch
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb = X[batch]
                # forward
                # encoder
                enc_zs = []
                enc_as = [Xb]
                a = Xb
                for li, (W, b) in enumerate(zip(self.enc_W, self.enc_b)):
                    z = a @ W + b
                    enc_zs.append(z)
                    if li == len(self.enc_W) - 1:
                        a = z  # linear
                    else:
                        a = _relu(z)
                    enc_as.append(a)
                latent = enc_as[-1]
                # decoder
                dec_zs = []
                dec_as = [latent]
                a = latent
                for li, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
                    z = a @ W + b
                    dec_zs.append(z)
                    if li == len(self.dec_W) - 1:
                        a = z  # linear output
                    else:
                        a = _relu(z)
                    dec_as.append(a)
                recon = dec_as[-1]
                # loss grad
                d_recon = (recon - Xb) * 2 / Xb.shape[0]
                # backprop decoder
                # gradient through layers reversed
                # We'll compute gradients lists
                grad_dec_W = [None] * len(self.dec_W)
                grad_dec_b = [None] * len(self.dec_b)
                # output layer
                # derivative of relu for hidden; output linear
                delta = d_recon  # (batch, out_dim)
                for li in range(len(self.dec_W) - 1, -1, -1):
                    a_prev = dec_as[li]
                    grad_dec_W[li] = a_prev.T @ delta
                    grad_dec_b[li] = delta.sum(axis=0)
                    if li > 0:
                        delta = delta @ self.dec_W[li].T
                        # relu grad of layer li-1 output (dec_as[li] is relu output of layer li-1? careful)
                        # dec_as[li] = activation of layer li-1's output (input to layer li)
                        # relu grad applies to layers except last; hidden layers use relu
                        if li - 1 < len(self.dec_W) - 1:
                            delta = delta * _relu_grad(dec_zs[li - 1])
                # gradient wrt latent
                d_latent = delta if False else None
                # compute d_latent from first decoder layer
                # delta currently after loop? Let's recompute latent grad cleanly:
                # latent grad = d_recon backprop through all decoder layers to latent
                delta = d_recon
                for li in range(len(self.dec_W) - 1, 0, -1):
                    delta = delta @ self.dec_W[li].T
                    delta = delta * _relu_grad(dec_zs[li - 1])
                d_latent = delta @ self.dec_W[0].T
                # backprop encoder
                grad_enc_W = [None] * len(self.enc_W)
                grad_enc_b = [None] * len(self.enc_b)
                delta = d_latent
                for li in range(len(self.enc_W) - 1, -1, -1):
                    a_prev = enc_as[li]
                    grad_enc_W[li] = a_prev.T @ delta
                    grad_enc_b[li] = delta.sum(axis=0)
                    if li > 0:
                        delta = delta @ self.enc_W[li].T
                        delta = delta * _relu_grad(enc_zs[li - 1])
                # gradient clipping for stability
                for g in (grad_enc_W + grad_enc_b + grad_dec_W + grad_dec_b):
                    nrm = np.linalg.norm(g)
                    if nrm > 5.0:
                        g *= 5.0 / nrm
                # update
                for li in range(len(self.enc_W)):
                    self.enc_W[li] -= self.lr * grad_enc_W[li]
                    self.enc_b[li] -= self.lr * grad_enc_b[li]
                for li in range(len(self.dec_W)):
                    self.dec_W[li] -= self.lr * grad_dec_W[li]
                    self.dec_b[li] -= self.lr * grad_dec_b[li]
        return self

    def encode(self, X):
        X = np.asarray(X, float)
        a = X
        for li, (W, b) in enumerate(zip(self.enc_W, self.enc_b)):
            z = a @ W + b
            if li == len(self.enc_W) - 1:
                a = z
            else:
                a = _relu(z)
        return a

    def decode(self, Z):
        Z = np.asarray(Z, float)
        a = Z
        for li, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
            z = a @ W + b
            if li == len(self.dec_W) - 1:
                a = z
            else:
                a = _relu(z)
        return a

    def predict(self, X):
        return self.decode(self.encode(X))

    def reconstruction_error(self, X):
        X = np.asarray(X, float)
        R = self.predict(X)
        return float(np.mean((X - R) ** 2))
