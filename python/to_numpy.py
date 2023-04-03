import cv2
import numpy as np
import pandas as pd

file_name = 'FFT_cpp_256_320'

data = pd.read_xml(f'../results/{file_name}.xml')

arr0 = np.array(data['data'])
arr = np.fromstring(arr0[0], dtype=np.float32, sep=' ')

rows = data['rows'][0]
cols = data['cols'][0]

arr = arr.reshape((rows, cols, -1))

np.save(f'../results/{file_name}.npy', arr)
