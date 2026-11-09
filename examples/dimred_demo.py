"""Demo: dimensionality reduction."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.data import make_blobs
from src.dimensionality_reduction.pca import PCA
from src.dimensionality_reduction.autoencoder import Autoencoder

X, y = make_blobs(n_samples=300, centers=3, n_features=8, random_state=0)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)
print("PCA explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
Xr = pca.inverse_transform(Z)
print("PCA reconstruction MSE:", round(float(np.mean((X - Xr) ** 2)), 4))

ae = Autoencoder(input_dim=8, hidden_dims=(16, 8), latent_dim=2,
                 lr=0.01, n_iters=100, batch_size=32, random_state=0)
ae.fit(X)
print("Autoencoder reconstruction MSE:", round(ae.reconstruction_error(X), 4))
print("latent shape:", ae.encode(X).shape)
