#%%
import numpy as np
import jax.numpy as jnp
from B_spline import *
from scipy.spatial import KDTree
import jax
from jax import lax
class VelocityPredictor:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError
    @staticmethod
    def velocity_pred(*args):
        raise NotImplementedError
    @staticmethod
    def velocity_grad(*args):
        raise NotImplementedError
    
class VelocityPrediction3D(VelocityPredictor):
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def init_params(**kwargs):
        pred_params = {}
        for key, value in kwargs.items():
            pred_params[key] = value
        return pred_params

    @staticmethod
    def find_indexes(data):
        offsets = jnp.array([1.5, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.5])
        col_indices = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        sizes = [5, 4, 4, 4, 5, 4, 4, 4, 5]
        base_indicies =jnp.floor(data[:, col_indices] - offsets).astype(jnp.int32)
        all_indices = [base_indicies[:, i:i+1] + jnp.arange(sizes[i]) for i in range(len(sizes))]
        return all_indices
    
    @staticmethod
    def velocity_pred(c, indexes, B_val):
        sections = [5, 9, 13, 18, 22, 26, 30, 34]
        idx_split = jnp.split(indexes, sections, axis=1)
        b_split = jnp.split(B_val, sections, axis=1)

        def compute_component(c_comp, ix, iy, iz, bx, by, bz):
            extracted_c = c_comp[
                ix[:, :, None, None], 
                iy[:, None, :, None], 
                iz[:, None, None, :]
            ]
            term = extracted_c * bx[:, :, None, None] * by[:, None, :, None] * bz[:, None, None, :]
            return jnp.sum(term, axis=(1, 2, 3))

        u_pred = compute_component(c[..., 0], idx_split[0], idx_split[1], idx_split[2], b_split[0], b_split[1], b_split[2])
        v_pred = compute_component(c[..., 1], idx_split[3], idx_split[4], idx_split[5], b_split[3], b_split[4], b_split[5])
        w_pred = compute_component(c[..., 2], idx_split[6], idx_split[7], idx_split[8], b_split[6], b_split[7], b_split[8])

        return u_pred, v_pred, w_pred

#%%
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from domain import *
    from projection import *
    from trackdata import *
    import os
    from time import time
    all_params = {"domain":{}, "data":{}, "projection":{}, "prediction":{}}
    domain_range= {'t': (0, 0.04), 'x':(0, 0.1), 'y':(0, 0.1), 'z':(0, 0.1)}
    grid_size = [256,256,256]
    coeff_shape = (grid_size[0]+4, grid_size[1]+4, grid_size[2]+4, 3)
    time_length = 3
    key = 42
    cur_dir = os.getcwd()
    path = '/ETFS/HIT/train_data/lv1/'
    data_keys = ['pos', 'vel',]
    viscosity = 15*10**(-6)

    u_ref = 1.5
    v_ref = 1.5
    w_ref = 0.9
    p_ref = 1.5
    all_params["data"] = Data.init_params(path = path, 
                                          data_keys = data_keys, 
                                          viscosity = viscosity,
                                          u_ref = u_ref,
                                          v_ref = v_ref,
                                          w_ref = w_ref,
                                          p_ref = p_ref)
    all_params["domain"] = Domain.init_params(domain_range = domain_range, 
                                              grid_size = grid_size)
    train_data, all_params = Data.train_data(all_params)
    all_params, p_p, p_u, p_v, p_w = Domain.generate_staggered_p(all_params)
    projection_params = FFTProjector.init_params(key, coeff_shape, time_length)
    prediction_params = VelocityPrediction3D.init_params(overlap=5)
#%%
    xc = np.linspace(all_params['domain']['domain_range']['x'][0], all_params['domain']['domain_range']['x'][1], all_params['domain']['grid_size'][0])
    yc = np.linspace(all_params['domain']['domain_range']['y'][0], all_params['domain']['domain_range']['y'][1], all_params['domain']['grid_size'][1])
    zc = np.linspace(all_params['domain']['domain_range']['z'][0], all_params['domain']['domain_range']['z'][1], all_params['domain']['grid_size'][2])
    dx = xc[1]-xc[0]
    dy = yc[1]-yc[0]
    dz = zc[1]-zc[0]

    pos = np.concatenate([train_data['pos'][:,1:2]/dx, train_data['pos'][:,2:3]/dy, train_data['pos'][:,3:4]/dz],1) + 2
    _,counts = np.unique(train_data['pos'][:,0],return_counts=True)
    pos_f = np.floor(pos)
    pos_n = pos - pos_f

    indexes_lists = []
    B_val_lists = []
    funcs = [B_spline.beta4, B_spline.beta3, B_spline.beta3, 
            B_spline.beta3, B_spline.beta4, B_spline.beta3,
            B_spline.beta3, B_spline.beta3, B_spline.beta4]
    for i in range(3):
        indexes = VelocityPrediction3D.find_indexes(pos[np.sum(counts[:i]):np.sum(counts[:i+1])])
        indexes_lists.append(np.concatenate(indexes,1))
        B_val = []
        for j in range(9):
            B_val.append(funcs[j](pos_n[np.sum(counts[:i]):np.sum(counts[:i+1]),j%3]))
        B_val = np.vstack(B_val).transpose()
        B_val_lists.append(B_val)

    u, v, w = VelocityPrediction3D.velocity_pred(projection_params['coefficients'][0], indexes_lists[0], B_val_lists[0])
    preds_ = np.concatenate([u.reshape(-1,1), v.reshape(-1,1), w.reshape(-1,1)], axis=-1)
#%%
    print([B_val_lists[i].shape for i in range(3)])
    print([indexes_lists[i].shape for i in range(3)])
#%%
    xc_ = np.append(np.append(np.insert(np.insert(xc, 0, xc[0]-dx), 0, xc[0]-2*dx),xc[-1]+dx), xc[-1]+2*dx)
    yc_ = np.append(np.append(np.insert(np.insert(yc, 0, yc[0]-dy), 0, yc[0]-2*dy),yc[-1]+dy), yc[-1]+2*dy)
    zc_ = np.append(np.append(np.insert(np.insert(zc, 0, zc[0]-dz), 0, zc[0]-2*dz),zc[-1]+dz), zc[-1]+2*dz)
    xt = xc_[60:140]
    yt = yc_[60:140]
    zt = zc_[60:140]
    xts, yts, zts = np.meshgrid(xt, yt, zt, indexing='ij')
    pos = np.concatenate([xts.reshape(-1,1)/dx, yts.reshape(-1,1)/dy, zts.reshape(-1,1)/dz],1)
    pos_f = np.floor(pos)
    pos_n = pos - pos_f

    indexes_lists = []
    B_val_lists = []
    funcs = [B_spline.beta4, B_spline.beta3, B_spline.beta3, 
            B_spline.beta3, B_spline.beta4, B_spline.beta3,
            B_spline.beta3, B_spline.beta3, B_spline.beta4]
    for i in range(3):
        indexes = VelocityPrediction3D.find_indexes(pos)
        indexes_lists.append(np.concatenate(indexes,1))
        B_val = []
        for j in range(9):
            B_val.append(funcs[j](pos_n[:,j%3]))
        B_val = np.vstack(B_val).transpose()
        B_val_lists.append(B_val)
    u, v, w = VelocityPrediction3D.velocity_pred(projection_params['coefficients'][0], indexes_lists[0], B_val_lists[0])
    preds_ = np.concatenate([u.reshape(-1,1), v.reshape(-1,1), w.reshape(-1,1)], axis=-1)
    preds_ = np.concatenate([u.reshape(xt.shape[0],80,80,1), v.reshape(80,80,80,1), w.reshape(80,80,80,1)], axis=-1)

    plt.imshow(FFTProjector.calculate_divergence_central(preds_)[:,:,10])
    plt.colorbar()
    plt.show()
    plt.imshow(preds_[:,:,10,0])
    plt.colorbar()
    plt.show()

# %%
    path = os.path.dirname(os.getcwd()) + '/ETFS/FlowFit_like/npyresult/HIT_flowfit_k1_penf4/'
    filenames = glob(path+'*.npy')
    file = np.load(filenames[0])
    plt.imshow(file[:,3].reshape(125,125,125)[15,:,:])
    plt.colorbar()
    plt.show()
#%%
    path = os.path.dirname(os.getcwd()) + '/ETFS/HIT/ground/'
    filenames = glob(path+'*.npy')
    file = np.load(filenames[0])
    plt.imshow(file[:,4].reshape(125,125,125)[15,:,:])
    plt.colorbar()
    plt.show()
