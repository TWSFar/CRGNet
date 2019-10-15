import torch
from torchvision import models

class A:
    def fun1(self):
        print('a')
    @staticmethod
    def fun2():
        A().fun1()

A().fun2()