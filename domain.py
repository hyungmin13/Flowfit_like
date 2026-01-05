#%%
import numpy as np
from B_spline import *
import os
from glob import glob
import jax.numpy as jnp
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
    def normalize(all_params, grids):
        domain_range = all_params["domain"]["domain_range"]
        key_list = list(grids.keys())
        arg_keys = ['x', 'y', 'z']
        for i in range(len(key_list)):
            for arg_key in arg_keys:
                grids[key_list[i]][arg_key] = grids[key_list[i]][arg_key]/domain_range[arg_key][1]
            #grids[key_list[i]]['t'] = grids[key_list[i]]['t']*domain_range[arg_key][1]*frequency
        return grids

    @staticmethod
    def sampler(all_params):
        domain_range = all_params["domain"]["domain_range"]
        grid_size = all_params["domain"]["grid_size"]
        path = all_params["data"]['path']
        cur_dir = os.getcwd()
        filenames = sorted(glob(os.path.dirname(cur_dir)+path+'*.npy'))

        arg_keys = ['x', 'y', 'z']
        grids = {'eqns':{arg_keys[j]:[] for j in range(len(arg_keys))}}

        for i in range(len(arg_keys)):
            grids['eqns'][arg_keys[i]] = np.linspace(domain_range[arg_keys[i]][0], domain_range[arg_keys[i]][1], grid_size[i])
        grids = Domain.normalize(all_params, grids)
        all_params["domain"]["in_min"] = jnp.array([[domain_range['x'][0], domain_range['y'][0], domain_range['z'][0]]])
        all_params["domain"]["in_max"] = jnp.array([[domain_range['x'][1], domain_range['y'][1], domain_range['z'][1]]])
        return grids, all_params
    
    
    @staticmethod
    def generate_staggered_p(all_params):
        domain_range = all_params["domain"]["domain_range"]
        grid_size = all_params["domain"]["grid_size"]
        arg_keys = ['x', 'y', 'z']
        grids = {'eqns':{arg_keys[j]:[] for j in range(len(arg_keys))}}
        for i in range(len(arg_keys)):
            grids['eqns'][arg_keys[i]] = np.linspace(domain_range[arg_keys[i]][0], domain_range[arg_keys[i]][1], grid_size[i])
        grids = Domain.normalize(all_params, grids)
        all_params["domain"]["in_min"] = jnp.array([[domain_range['x'][0], domain_range['y'][0], domain_range['z'][0]]])
        all_params["domain"]["in_max"] = jnp.array([[domain_range['x'][1], domain_range['y'][1], domain_range['z'][1]]])
        dx = float(grids['eqns']['x'][1] - grids['eqns']['x'][0])
        dy = float(grids['eqns']['y'][1] - grids['eqns']['y'][0])
        dz = float(grids['eqns']['z'][1] - grids['eqns']['z'][0])
        Xc, Yc, Zc = np.meshgrid(grids['eqns']['x'], grids['eqns']['y'], grids['eqns']['z'], indexing="ij")
        p_p = np.stack([Xc, Yc, Zc], axis=-1)
        p_u = np.stack([Xc + dx/2.0, Yc, Zc], axis=-1)
        p_v = np.stack([Xc, Yc + dy/2.0, Zc], axis=-1)
        p_w = np.stack([Xc, Yc, Zc + dz/2.0], axis=-1)

        return all_params, p_p, p_u, p_v, p_w

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    all_params = {"domain":{}, "data":{}}
    viscosity = 15*10**(-6)
    
    domain_range= {'x':(-1.0, 1.0), 'y':(-1.0, 1.0), 'z':(-1.0, 1.0)}
    grid_size = [256, 256, 256]
    all_params["domain"] = Domain.init_params(domain_range = domain_range,
                                              grid_size = grid_size)

    all_params, p_p, p_u, p_v, p_w = Domain.generate_staggered_p(all_params)
    #grids = Domain.sampler(all_params)
