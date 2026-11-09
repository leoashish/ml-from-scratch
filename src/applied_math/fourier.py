"""Applied Mathematics: Fourier & Signal, with visuals."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "visuals")
os.makedirs(OUT, exist_ok=True)


def fft_visual():
    fs = 100
    t = np.arange(0, 1, 1/fs)
    sig = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*12*t)
    F = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    ax[0].plot(t, sig); ax[0].set_title("Signal")
    ax[1].plot(freqs, np.abs(F)); ax[1].set_title("FFT magnitude")
    ax[1].set_xlim(0, 20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fft.png")); plt.close(fig)


def filtering_visual():
    fs = 200
    t = np.arange(0, 1, 1/fs)
    sig = np.sin(2*np.pi*5*t) + 0.5*np.random.randn(len(t))
    # moving average
    kernel = np.ones(5)/5
    filt = np.convolve(sig, kernel, mode="same")
    fig, ax = plt.subplots()
    ax.plot(t, sig, alpha=0.4, label="noisy")
    ax.plot(t, filt, label="filtered")
    ax.legend(); ax.grid(); ax.set_title("Moving average filter")
    fig.savefig(os.path.join(OUT, "filter.png")); plt.close(fig)


def run_all():
    fft_visual()
    filtering_visual()
    print("fourier visuals done")
