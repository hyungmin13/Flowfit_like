import numpy as np
import jax.numpy as jnp
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
    def __init__(self, all_params):
        self.all_params = all_params
    @staticmethod
    def beta3(t):

        a = jnp.abs(t)
        val1 = 2.0/3.0 - a**2 + 0.5 * a**3
        val2 = (1.0/6.0) * (2.0 - a)**3
        return jnp.where(a < 1.0, val1,
                        jnp.where(a < 2.0, val2, 0.0))
    @staticmethod
    def beta4(t):
        a = jnp.abs(t)

        region1 = (a < 0.5)
        val1 = (1.0/4.0) * a**4 - (5.0/8.0) * a**2 + (115.0/192.0)

        region2 = (a >= 0.5) & (a < 1.5)
        val2 = (-1.0/6.0) * a**4 + (5.0/6.0) * a**3 - (5.0/4.0) * a**2 \
            + (5.0/24.0) * a + (55.0/96.0)

        region3 = (a >= 1.5) & (a < 2.5)
        val3 = (1.0/24.0) * (2.5 - a)**4
        return jnp.where(region1, val1,
                        jnp.where(region2, val2,
                                jnp.where(region3, val3, 0.0)))

if __name__=="__main__":
    import matplotlib.pyplot as plt
