"""Applied Mathematics: Optimization, with visuals."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "visuals")
os.makedirs(OUT, exist_ok=True)


def convexity_visual():
    f = lambda x: x**2
    g = lambda x: -x**2  # concave
    x = np.linspace(-2, 2, 100)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(x, f(x)); ax[0].set_title("Convex")
    ax[1].plot(x, g(x)); ax[1].set_title("Concave (not convex)")
    for a in ax: a.grid()
    fig.savefig(os.path.join(OUT, "convexity.png")); plt.close(fig)


def newton_visual():
    f = lambda x: x**2 - 4
    df = lambda x: 2*x
    ddf = lambda x: 2
    x = 3.0
    xs = [x]
    for _ in range(5):
        x = x - df(x)/ddf(x)
        xs.append(x)
    xs = np.array(xs)
    fig, ax = plt.subplots()
    t = np.linspace(0, 4, 100)
    ax.plot(t, f(t), "k")
    ax.plot(xs, f(xs), "ro-")
    ax.axhline(0, color="gray")
    ax.set_title("Newton's method")
    fig.savefig(os.path.join(OUT, "newton.png")); plt.close(fig)


def lagrange_visual():
    # minimize x^2+y^2 s.t. x+y=1
    # solution (0.5,0.5)
    x = np.linspace(-1, 2, 100); y = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, levels=20)
    ax.plot(x, 1-x, "r", label="constraint x+y=1")
    ax.plot(0.5, 0.5, "go", label="optimum")
    ax.legend(); ax.grid(); ax.set_title("Lagrange multiplier")
    fig.savefig(os.path.join(OUT, "lagrange.png")); plt.close(fig)


def run_all():
    convexity_visual()
    newton_visual()
    lagrange_visual()
    print("optimization visuals done")
