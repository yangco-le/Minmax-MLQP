import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def dataloader(mode):
    file = open("data/two_spiral_{}_data.txt".format(mode), 'r')
    res = []
    for line in file.readlines():
        res.append([np.array([float(line.split()[0]), float(line.split()[1])]), int(line.split()[2])])
    file.close()
    return np.array(res)