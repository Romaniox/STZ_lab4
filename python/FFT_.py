import os
import cv2
import numpy as np
from pathlib import Path
from utils import show_images


ROOT = Path(os.getcwd())
img_path = ROOT.parent / 'cmake-build-debug' / 'images' / 'fourier.png'

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows, cols = img.shape
m = cv2.getOptimalDFTSize(rows)
n = cv2.getOptimalDFTSize(cols)

padded = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

img_complex = np.fft.fft2(padded)

img_mag = cv2.magnitude(img_complex.real, img_complex.imag)

img_mag = np.fft.fftshift(img_mag)

mat_ones = np.ones(img_mag.shape, dtype=img_mag.dtype)
img_mag = cv2.add(mat_ones, img_mag)  # switch to logarithmic scale
img_mag = cv2.log(img_mag)

cv2.normalize(img_mag, img_mag, 0, 1, cv2.NORM_MINMAX)

img_back = np.fft.ifft2(img_complex)

img_back = img_back.real
img_back = img_back.astype(np.uint8)

show_images(img, padded, img_mag, img_back)
