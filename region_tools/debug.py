a = ["df", "re", "df"]
b = [12, 23, 45]
import torch as t
import numpy as np
m = t.tensor([23]).cuda()
n = np.sum(m.numpy())
print(b[:-1])
print("{}{}".format(*b[:-1]))