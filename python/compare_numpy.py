import numpy as np

file_name_1 = 'FFT_cpp_256_320'
file_name_2 = 'DFT_cpp_256_320'

np_1 = np.load(f'../results/{file_name_1}.npy').astype(np.int32)
np_2 = np.load(f'../results/{file_name_2}.npy').astype(np.int32)

dif = np.abs(np_1 - np_2).sum()

avg_img = (np.abs(np_1) + np.abs(np_2)) / 2
err = dif / avg_img.sum() * 100
acc = 100 - err

# print(dif)
print(f'Error: {err}%')
print(f'Accuracy: {acc}%')
