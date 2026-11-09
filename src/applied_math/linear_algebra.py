"""Applied Mathematics: Linear Algebra, with visual intuition."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "visuals")
os.makedirs(OUT, exist_ok=True)


def vector_add_visual():
    a = np.array([2, 1]); b = np.array([1, 2])
    fig, ax = plt.subplots()
    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="r", label="a")
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="b", label="b")
    ax.quiver(0, 0, (a+b)[0], (a+b)[1], angles="xy", scale_units="xy", scale=1, color="g", label="a+b")
    ax.set_xlim(-1, 4); ax.set_ylim(-1, 4); ax.grid(); ax.legend()
    ax.set_title("Vector addition")
    fig.savefig(os.path.join(OUT, "vec_add.png")); plt.close(fig)


def matrix_transform_visual():
    # linear map A rotates+scales
    A = np.array([[0, -1], [1, 0]])  # 90 deg rotation
    pts = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]]).T
    tpts = A @ pts
    fig, ax = plt.subplots()
    ax.plot(pts[0], pts[1], "b-o", label="original")
    ax.plot(tpts[0], tpts[1], "r-o", label="transformed")
    ax.set_aspect("equal"); ax.grid(); ax.legend()
    ax.set_title("Linear transformation (rotation)")
    fig.savefig(os.path.join(OUT, "mat_transform.png")); plt.close(fig)


def eigen_visual():
    A = np.array([[2, 1], [1, 2]])
    w, V = np.linalg.eig(A)
    fig, ax = plt.subplots()
    # unit circle
    th = np.linspace(0, 2*np.pi, 100)
    circ = np.vstack([np.cos(th), np.sin(th)])
    ax.plot(circ[0], circ[1], "k--", alpha=0.3)
    ell = A @ circ
    ax.plot(ell[0], ell[1], "b")
    for i in range(2):
        ax.quiver(0, 0, V[0, i], V[1, i], angles="xy", scale_units="xy", scale=1, color="r")
        ax.text(V[0, i]*1.1, V[1, i]*1.1, f"λ={w[i]:.2f}")
    ax.set_aspect("equal"); ax.grid()
    ax.set_title("Eigenvectors of symmetric A")
    fig.savefig(os.path.join(OUT, "eigen.png")); plt.close(fig)


def svd_visual():
    A = np.array([[3, 0], [0, 1], [0, 0]], dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # singular values
    ax[0].bar(range(len(s)), s); ax[0].set_title("Singular values")
    # reconstruction
    recon = (U * s) @ Vt
    ax[1].imshow(recon, cmap="viridis"); ax[1].set_title("Reconstructed A")
    fig.savefig(os.path.join(OUT, "svd.png")); plt.close(fig)


def run_all():
    vector_add_visual()
    matrix_transform_visual()
    eigen_visual()
    svd_visual()
    print("linear_algebra visuals done")
