import os
import cv2
import numpy as np
from pathlib import Path
from utils import show_images


def dft(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(W, x)
    return X


def idft(X):
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(2j * np.pi * k * n / N) / N
    x = np.dot(W, X)
    return x


ROOT = Path(os.getcwd())
img_path = ROOT.parent / 'cmake-build-debug' / 'images' / 'fourier.png'

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows, cols = img.shape
m = cv2.getOptimalDFTSize(rows)
n = cv2.getOptimalDFTSize(cols)

padded = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

rows_complex = []
for row in padded.astype(np.float32):
    row_complex = dft(row)
    rows_complex.append(row_complex)

rows_complex = np.array(rows_complex)

cols_complex = []
for col in rows_complex.T:
    col_complex = dft(col)
    cols_complex.append(col_complex)

img_complex = np.array(cols_complex)
img_complex = img_complex.T

planes = (img_complex.real, img_complex.imag)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

img_mag = cv2.magnitude(planes[0], planes[1])

img_mag = np.fft.fftshift(img_mag)

mat_ones = np.ones(img_mag.shape, dtype=img_mag.dtype)
img_mag = cv2.add(mat_ones, img_mag)  # switch to logarithmic scale
img_mag = cv2.log(img_mag)

cv2.normalize(img_mag, img_mag, 0, 1, cv2.NORM_MINMAX)  # Transform the matrix with float values into a

rows_complex = []
for row in img_complex:
    row_complex = idft(row)
    rows_complex.append(row_complex)
rows_complex = np.array(rows_complex)

cols_complex = []
for col in rows_complex.T:
    col_complex = idft(col)
    cols_complex.append(col_complex)

img_back = np.array(cols_complex)
img_back = img_back.T
img_back = img_back.real

# img_back = cv2.dft(img_complex, flags=(cv2.DFT_INVERSE | cv2.DFT_REAL_OUTPUT))
# cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
img_back = img_back.astype(np.uint8)

show_images(img, padded, img_mag, img_back)

pass
