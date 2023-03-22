import os
import cv2
import numpy as np
from pathlib import Path
from utils import show_images

from DFT import get_W, dft

ROOT = Path(os.getcwd())
img_path = ROOT.parent / 'cmake-build-debug' / 'images' / 'fourier.png'

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows, cols = img.shape
m = cv2.getOptimalDFTSize(rows)
n = cv2.getOptimalDFTSize(cols)

padded = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

img_complex = padded.astype(np.complex64)

W = get_W(img_complex.shape[1], inv=False)
for i, row in enumerate(img_complex):
    row_complex = dft(row, W)
    img_complex[i] = row_complex.copy()

W = get_W(img_complex.shape[0], inv=False)
for i, col in enumerate(img_complex.T):
    col_complex = dft(col, W)
    img_complex.T[i] = col_complex

planes = (img_complex.real, img_complex.imag)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
img_mag = cv2.magnitude(planes[0], planes[1])
img_mag = np.fft.fftshift(img_mag)

mat_ones = np.ones(img_mag.shape, dtype=img_mag.dtype)
img_mag = cv2.add(mat_ones, img_mag)  # switch to logarithmic scale
img_mag = cv2.log(img_mag)

cv2.normalize(img_mag, img_mag, 0, 1, cv2.NORM_MINMAX)  # Transform the matrix with float values into a

img_back = img_complex.copy()

W = get_W(img_back.shape[1], inv=True)
for i, row in enumerate(img_back):
    row_complex = dft(row, W)
    img_back[i] = row_complex.copy()

W = get_W(img_back.shape[0], inv=True)
for i, col in enumerate(img_back.T):
    col_complex = dft(col, W)
    img_back.T[i] = col_complex

img_back = img_back.real

# cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
img_back = img_back.astype(np.uint8)

show_images(img, padded, img_mag, img_back)
