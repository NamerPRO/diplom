import numpy as np


def log_sum(x, y):
    if x == -np.inf and y == -np.inf:
        return -np.inf
    if x > y:
        return x + np.log(1 + np.exp(y - x))
    return y + np.log(1 + np.exp(x - y))