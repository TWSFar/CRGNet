from scipy.ndimage.filters import gaussian_filter
import numpy as np


pt2d = np.zeros((3, 5), dtype=np.float32)
pt2d[1, 2] = 1

temp = gaussian_filter(pt2d, [1.5, 0.9], mode="constant")
# print(temp[])
print(sum(sum(temp)))
# print(temp)