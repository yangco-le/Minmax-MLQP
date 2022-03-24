import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))