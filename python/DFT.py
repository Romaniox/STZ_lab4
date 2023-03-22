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