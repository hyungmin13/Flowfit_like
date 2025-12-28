#%%
import numpy as np
from B_spline import *

class Domainbase:
    @staticmethod
    def __init__(*args):
        raise NotImplementedError
    @staticmethod
    def generate_staggered_p(*args):
        raise NotImplementedError

class Domain(Domainbase):
    @staticmethod
    def init_params(**kwargs):
        domain_params = {}
        for key, value in kwargs.items():
            domain_params[key] = value
        return domain_params
    
    @staticmethod
    def bound_sampler(all_params, grids):
        bound_keys = all_params["domain"]["bound_keys"]
        arg_keys = ['x', 'y', 'z']
        total_bound = {bound_keys[i]:{arg_keys[j]:[] for j in range(len(arg_keys))} for i in range(len(bound_keys))}
        for i, bound_key in enumerate(bound_keys):
            for j in range(len(arg_keys)):
                total_bound[bound_key][arg_keys[j]] = grids['eqns'][arg_keys[j]]
            for arg_key in arg_keys:
                if arg_key in bound_key:
                    if 'u' in bound_key:
                        total_bound[bound_key][arg_key] = np.array([grids['eqns'][arg_key][-1]])
                    else:
                        total_bound[bound_key][arg_key] = np.array([grids['eqns'][arg_key][0]])
            if 'ic' in bound_key:
                total_bound[bound_key]['t'] = grids['eqns']['t'][0]
        grids.update(total_bound)
        return grids
    
    @staticmethod
    def generate_staggered_p(all_params):
        domain_range = all_params["domain"]["domain_range"]
        grid_size = all_params["domain"]["grid_size"]
        arg_keys = ['x', 'y', 'z']
        grids = {'eqns':{arg_keys[j]:[] for j in range(len(arg_keys))}}
        for i in range(len(arg_keys)-1):
            grids['eqns'][arg_keys[i]] = np.linspace(domain_range[arg_keys[i]][0], domain_range[arg_keys[i]][1], grid_size[i])
        grids = Domain.bound_sampler(all_params, grids)

        Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing="ij")
        p_u = np.stack([Xc + dx/2.0, Yc, Zc], axis=-1)
        p_v = np.stack([Xc, Yc + dy/2.0, Zc], axis=-1)
        p_w = np.stack([Xc, Yc, Zc + dz/2.0], axis=-1)

        return p_u, p_v, p_w
    @staticmethod

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    I = J = K = 256
    grid_shape = (I, J, K, 3)

    xc = np.linspace(-1.0, 1.0, I)
    yc = np.linspace(-1.0, 1.0, J)
    zc = np.linspace(-1.0, 1.0, K)
    dx = float(xc[1] - xc[0])
    dy = float(yc[1] - yc[0])
    dz = float(zc[1] - zc[0])
    p_u, p_v, p_w = Domain.generate_staggered_p(xc, yc, zc, dx, dy, dz)
