import numpy as np



class Evaluate():
    def __init__ (self):
        print("Evaluators object has been initialized.")

    def levy(self, x):
        w = 1 + (x - 1) / 4
        wp = w[:-1]
        wd = w[-1]
        a = np.sin(np.pi * w[0]) ** 2
        b = sum((wp - 1) ** 2 * (1 + 10 * np.sin(np.pi * wp + 1) ** 2))
        c = (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2)
        return a + b + c