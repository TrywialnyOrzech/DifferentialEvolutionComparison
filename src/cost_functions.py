import numpy as np


def rosenbrock(x):
    z_shift = 0
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + z_shift


def levy(x):
    w = 1 + (x - 1) / 4
    wp = w[:-1]
    wd = w[-1]
    a = np.sin(np.pi * w[0]) ** 2
    b = sum((wp - 1) ** 2 * (1 + 10 * np.sin(np.pi * wp + 1) ** 2))
    c = (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2)
    return a + b + c


def michalewicz(x):
    m = 10          # recommended value, determines steepness of valleys and ridges
    c = 0
    for i in range(0, len(x)):
        c += np.sin(x[i]) * np.sin(((i + 1) * x[i] ** 2) / np.pi) ** (2 * m)
    return -c


def bowl(x):
    value = 0
    for i in range(len(x)):
        value += x[i] ** 2
    return value / len(x)
