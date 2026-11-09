"""Demo: generative modelling."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.data import make_moons
from src.generative.diffusion import DiffusionModel
from src.generative.autoregressive import AutoregressiveCharModel

# Diffusion on 2D data
X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
dm = DiffusionModel(data_dim=2, T=20, n_iters=200, random_state=0)
dm.fit(X)
samples = dm.sample(100, seed=1)
print("Diffusion samples shape:", samples.shape, "mean:", np.round(samples.mean(0), 3))

# Autoregressive char model
text = "the quick brown fox jumps over the lazy dog. " * 10
m = AutoregressiveCharModel(context=4, hidden=32, lr=0.1, n_iters=300, random_state=0)
m.fit(text)
print("Generated:", m.generate("the ", length=50, seed=2))
