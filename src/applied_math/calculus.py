"""Applied Mathematics: Calculus & gradients, with visuals."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "visuals")
os.makedirs(OUT, exist_ok=True)


def derivative_visual():
    f = lambda x: x**2
    df = lambda x: 2*x
    x = np.linspace(-3, 3, 100)
    fig, ax = plt.subplots()
    ax.plot(x, f(x), label="f(x)=x^2")
    ax.plot(x, df(x), label="f'(x)=2x")
    ax.grid(); ax.legend(); ax.set_title("Derivative")
    fig.savefig(os.path.join(OUT, "derivative.png")); plt.close(fig)


def gradient_descent_visual():
    f = lambda x: (x-2)**2 + 1
    x = 5.0
    xs = [x]; lr = 0.1
    for _ in range(20):
        g = 2*(x-2)
        x -= lr*g
        xs.append(x)
    xs = np.array(xs)
    fig, ax = plt.subplots()
    t = np.linspace(-1, 6, 100)
    ax.plot(t, f(t), "k-")
    ax.plot(xs, f(xs), "ro-")
    ax.set_title("Gradient descent on f(x)")
    fig.savefig(os.path.join(OUT, "gd.png")); plt.close(fig)


def chain_rule_visual():
    # compose f(g(x)) = (3x+1)^2
    x = np.linspace(-2, 2, 100)
    g = 3*x + 1
    f = g**2
    df = 2*g*3
    fig, ax = plt.subplots()
    ax.plot(x, f, label="f(g(x))")
    ax.plot(x, df, label="d/dx via chain rule")
    ax.grid(); ax.legend(); ax.set_title("Chain rule")
    fig.savefig(os.path.join(OUT, "chain_rule.png")); plt.close(fig)


def taylor_visual():
    f = np.exp
    x0 = 0
    x = np.linspace(-2, 2, 100)
    t1 = 1 + x
    t2 = 1 + x + x**2/2
    t3 = 1 + x + x**2/2 + x**3/6
    fig, ax = plt.subplots()
    ax.plot(x, f(x), "k", label="exp(x)")
    ax.plot(x, t1, "--", label="order1")
    ax.plot(x, t2, "--", label="order2")
    ax.plot(x, t3, "--", label="order3")
    ax.legend(); ax.grid(); ax.set_title("Taylor expansion")
    fig.savefig(os.path.join(OUT, "taylor.png")); plt.close(fig)


def run_all():
    derivative_visual()
    gradient_descent_visual()
    chain_rule_visual()
    taylor_visual()
    print("calculus visuals done")
