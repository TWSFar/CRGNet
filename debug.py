import os
import sys
import numpy as np

m = np.array([[1, 2], [3, 4]])
m = np.pad(m, ((0, 1), (1, 0)), 'constant')
pass