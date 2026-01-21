#%%
import numpy as np
import jax.numpy as jnp
from time import time
class B_spline_bases:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError
    @staticmethod
    def beta3(*args):
        raise NotImplementedError
    @staticmethod
    def beta4(*args):
        raise NotImplementedError
class B_spline(B_spline_bases):
    cubic_0th = np.array([[ -1.0/6,  3.0/6, -3.0/6,  1.0/6 ], [  3.0/6, -6.0/6,  0.0/6,  4.0/6 ], [ -3.0/6,  3.0/6,  3.0/6,  1.0/6 ], [  1.0/6,  0.0/6,  0.0/6,  0.0/6 ]])
    cubic_1st = np.array([[ -1.0/2,  2.0/2, -1.0/2 ], [  3.0/2, -4.0/2,  0.0/2 ], [ -3.0/2,  2.0/2,  1.0/2 ], [  1.0/2,  0.0/2,  0.0/2 ]])
    cubic_2nd = np.array([[ -1.0,  1.0 ], [  3.0, -2.0 ], [ -3.0,  1.0 ], [  1.0,  0.0 ]])

    # Quartic Coefficients [Segment][Coeffs]
    quartic_0th = np.array([[   1.0/24,  -4.0/24,   6.0/24,  -4.0/24,   1.0/24 ], [  -4.0/24,  12.0/24,  -6.0/24, -12.0/24,  11.0/24 ], [   6.0/24, -12.0/24,  -6.0/24,  12.0/24,  11.0/24 ], 
        [  -4.0/24,   4.0/24,   6.0/24,   4.0/24,   1.0/24 ], [   1.0/24,   0.0/24,   0.0/24,   0.0/24,   0.0/24 ]])

    quartic_1st = np.array([[  1.0/6, -3.0/6,  3.0/6, -1.0/6 ], [ -4.0/6,  9.0/6, -3.0/6, -3.0/6 ], [  6.0/6, -9.0/6, -3.0/6,  3.0/6 ], 
        [ -4.0/6,  3.0/6,  3.0/6,  1.0/6 ], [  1.0/6,  0.0/6,  0.0/6,  0.0/6 ]])

    quartic_2nd = np.array([[  1.0/2, -2.0/2,  1.0/2 ], [ -4.0/2,  6.0/2, -1.0/2 ], [  6.0/2, -6.0/2, -1.0/2 ], [ -4.0/2,  2.0/2,  1.0/2 ], [  1.0/2,  0.0/2,  0.0/2 ]])
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def beta3(t):
        res = B_spline.cubic_0th[:,0:1]
        for i in range(1,4):
            res = t * res + B_spline.cubic_0th[:,i:i+1]
        return res
    @staticmethod
    def beta4(t):
        res = B_spline.quartic_0th[:,0:1]
        for i in range(1,5):
            res = t * res + B_spline.quartic_0th[:,i:i+1]
        return res

    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    t = np.linspace(0,1,100)
    times = []
    #func = B_spline(1)
    for i in range(1000):
        time_ = time()
        res = B_spline.beta3(t)
        times.append(time()-time_)
    print(np.mean(times))
