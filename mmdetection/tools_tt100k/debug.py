from mmdet.ops import nms
import numpy as np


box1 = np.array([[1, 1, 4, 4, 0.9], [1, 1, 3, 3, 0.8], [4, 4, 6, 6, 0.7]])
box2 = box1[box1[:, -1] > 1]

res = nms(box2, 0.2)[0].tolist()
pass
