"""Applied Mathematics: Probability & Statistics, with visuals."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats if False else None
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "visuals")
os.makedirs(OUT, exist_ok=True)


def distributions_visual():
    x = np.linspace(-4, 4, 200)
    fig, ax = plt.subplots()
    ax.plot(x, _norm_pdf(x, 0, 1), label="N(0,1)")
    ax.plot(x, _norm_pdf(x, 1, 0.5), label="N(1,0.5)")
    # binomial
    n, p = 20, 0.3
    k = np.arange(0, n+1)
    ax2 = ax.twinx()
    ax2.bar(k, _binom_pmf(k, n, p), alpha=0.3, color="orange", label="Binomial")
    ax.set_title("Distributions")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.savefig(os.path.join(OUT, "distributions.png")); plt.close(fig)


def _norm_pdf(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)


def _binom_pmf(k, n, p):
    from math import comb
    return np.array([comb(n, int(kk)) * p**kk * (1-p)**(n-kk) for kk in k])


def bayes_visual():
    # prior, likelihood, posterior for coin bias
    thetas = np.linspace(0, 1, 100)
    prior = np.ones_like(thetas)
    # observed 7 heads in 10
    likelihood = thetas**7 * (1-thetas)**3
    posterior = prior * likelihood
    posterior /= np.trapz(posterior, thetas)
    fig, ax = plt.subplots()
    ax.plot(thetas, prior/prior.max(), label="prior")
    ax.plot(thetas, likelihood/likelihood.max(), label="likelihood")
    ax.plot(thetas, posterior/posterior.max(), label="posterior")
    ax.legend(); ax.grid(); ax.set_title("Bayes theorem")
    fig.savefig(os.path.join(OUT, "bayes.png")); plt.close(fig)


def clt_visual():
    rng = np.random.default_rng(0)
    means = []
    for _ in range(1000):
        sample = rng.exponential(scale=2, size=30)
        means.append(sample.mean())
    fig, ax = plt.subplots()
    ax.hist(means, bins=30, density=True, alpha=0.6)
    x = np.linspace(min(means), max(means), 100)
    ax.plot(x, _norm_pdf(x, 2, 2/np.sqrt(30)), "r")
    ax.set_title("Central Limit Theorem")
    fig.savefig(os.path.join(OUT, "clt.png")); plt.close(fig)


def run_all():
    distributions_visual()
    bayes_visual()
    clt_visual()
    print("probability visuals done")
