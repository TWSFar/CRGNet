a = ["df", "re", "df"]
b = [12, 23, 45]
import torch as t
import numpy as np
m = t.tensor([23.3, 32], requires_grad=True)
conv1 = t.nn.Conv2d(3, 3, 1, 1)

loss_1 = t.sum(m)
loss_2 = m.sum()
loss_1.backward()
loss_2.backward()
pass
