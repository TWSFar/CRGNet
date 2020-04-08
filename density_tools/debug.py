from utils import iou_calc1
import numpy as np


a = np.array([[0, 0, 5, 5], [0, 0, 3, 3], [3,  3, 5, 5]])

b = np.array([[0, 0, 3, 3], [1, 1, 3, 3]])

c = a[0]


iou = iou_calc1(c[np.newaxis, :], b)

pass