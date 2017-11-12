import numpy as np
import cv2
import os

def saveGeneratedBatch(tensor, num_per_row, idx, output_dir='output'):
    # Check output dir
    if not os.path.exists(output_dir):
        os.system('mkdir -p ' + output_dir)

    # Ensure the num_per_row is the factor of batch size
    if (np.shape(tensor)[0] // num_per_row) * num_per_row != np.shape(tensor)[0]:
        num_per_row = 1
        for i in range(1, np.shape(tensor)[0]):
            if (np.shape(tensor)[0] // i) * i == np.shape(tensor)[0]:
                num_per_row = i

    # Save grids
    res = None
    for i in range(np.shape(tensor)[0] // num_per_row):
        res_row = None
        for j in range(num_per_row):
            if j == 0:
                res_row = tensor[j]
            else:
                res_row = np.concatenate((res_row, tensor[i*num_per_row+j]), axis=1)
        if i == 0:
            res = res_row
        else:
            res = np.concatenate((res, res_row), axis=0)
    cv2.imwrite(output_dir + '/' +str(idx) + '.png', res)

if __name__ == '__main__':
    saveGeneratedBatch(np.ones([32, 28, 28, 1], dtype=np.float) * 100, 8, 0, 'output/data/wgan')