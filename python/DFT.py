import numpy as np


def get_W(N, inv=False):
    n = np.arange(N)
    k = n.reshape((N, 1))
    if inv:
        W = np.exp(2j * np.pi * k * n / N) / N
    else:
        W = np.exp(-2j * np.pi * k * n / N)

    return W


def dft(x, W):
    X = np.dot(W, x)
    return X


def fft_(x):
    x = x.astype(np.float32)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")

    X_even = fft(x[::2])
    X_odd = fft(x[1::2])
    terms = np.exp(-2j * np.pi * np.arange(N) / N)
    X = np.concatenate([X_even + terms[:int(N / 2)] * X_odd,
                        X_even + terms[int(N / 2):] * X_odd])
    return X


def fft(x):
    # x = x.astype(np.float32)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")

    N_min = min(N, 2)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                       / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])
    return X.ravel()
