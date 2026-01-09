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
    def find_indexes(data, xc, yc, zc, dx, dy, dz):
        tree_xu = KDTree(xc.reshape(-1,1)+dx/2)
        tree_yu = KDTree(yc.reshape(-1,1))
        tree_zu = KDTree(zc.reshape(-1,1))

        _, i_xu = tree_xu.query(data[:,0:1],k=5)
        _, i_yu = tree_yu.query(data[:,1:2],k=5)
        _, i_zu = tree_zu.query(data[:,2:3],k=5)

        tree_xv = KDTree(xc.reshape(-1,1))
        tree_yv = KDTree(yc.reshape(-1,1)+dy/2)
        tree_zv = KDTree(zc.reshape(-1,1))

        _, i_xv = tree_xv.query(data[:,0:1],k=5)
        _, i_yv = tree_yv.query(data[:,1:2],k=5)
        _, i_zv = tree_zv.query(data[:,2:3],k=5)

        tree_xw = KDTree(xc.reshape(-1,1))
        tree_yw = KDTree(yc.reshape(-1,1))
        tree_zw = KDTree(zc.reshape(-1,1)+dz/2)

        _, i_xw = tree_xw.query(data[:,0:1],k=5)
        _, i_yw = tree_yw.query(data[:,1:2],k=5)
        _, i_zw = tree_zw.query(data[:,2:3],k=5)

        return i_xu, i_yu, i_zu, i_xv, i_yv, i_zv, i_xw, i_yw, i_zw

    @staticmethod
    def data_reshape(data, i_xu, i_yu, i_zu, i_xv, i_yv, i_zv, i_xw, i_yw, i_zw, dx, dy, dz, xc, yc, zc):
        x_u = np.concatenate([data[:,None,0:1]-xc[i_xu.reshape(-1)].reshape(-1,5)[:,:,None]+dx/2,
                                    data[:,None,1:2]-yc[i_yu.reshape(-1)].reshape(-1,5)[:,:,None],
                                    data[:,None,2:3]-zc[i_zu.reshape(-1)].reshape(-1,5)[:,:,None]],2)
        x_u_ = x_u.reshape(-1,3)
        x_v = np.concatenate([data[:,None,0:1]-xc[i_xv.reshape(-1)].reshape(-1,5)[:,:,None],
                                    data[:,None,1:2]-yc[i_yv.reshape(-1)].reshape(-1,5)[:,:,None]+dy/2,
                                    data[:,None,2:3]-zc[i_zv.reshape(-1)].reshape(-1,5)[:,:,None]],2)
        x_v_ = x_v.reshape(-1,3)
        x_w = np.concatenate([data[:,None,0:1]-xc[i_xw.reshape(-1)].reshape(-1,5)[:,:,None],
                                    data[:,None,1:2]-yc[i_yw.reshape(-1)].reshape(-1,5)[:,:,None],
                                    data[:,None,2:3]-zc[i_zw.reshape(-1)].reshape(-1,5)[:,:,None]+dz/2],2)
        x_w_ = x_w.reshape(-1,3)
        return x_u_, x_v_, x_w_

    @staticmethod
    def velocity_pred(c, indexes, dx, dy, dz, x_u, x_v, x_w):
        c_sliced = c[..., 0]
        b1 = x_u[:,0]/dx
        b_ = B_spline.beta4(x_u[:,0]/dx)
        b3_ux = B_spline.beta4(x_u[:,0]/dx).reshape(-1,5)
        b3_uy = B_spline.beta3(x_u[:,1]/dy).reshape(-1,5)
        b3_uz = B_spline.beta3(x_u[:,2]/dz).reshape(-1,5)

        idx_x = indexes[0, :, :, None, None] # (N, 5, 1, 1)
        idx_y = indexes[1, :, None, :, None] # (N, 1, 5, 1)
        idx_z = indexes[2, :, None, None, :] # (N, 1, 1, 5)
        extracted_c = c_sliced[idx_x, idx_y, idx_z]
        ux = b3_ux[:, :, np.newaxis, np.newaxis]
        uy = b3_uy[:, np.newaxis, :, np.newaxis]
        uz = b3_uz[:, np.newaxis, np.newaxis, :]

        u_pred = extracted_c * ux * uy * uz
        u_pred = np.sum(u_pred.reshape(-1,125),axis=-1)

        c_sliced = c[..., 1]

        b3_vx = B_spline.beta3(x_v[:,0]/dx).reshape(-1,5)
        b3_vy = B_spline.beta4(x_v[:,1]/dy).reshape(-1,5)
        b3_vz = B_spline.beta3(x_v[:,2]/dz).reshape(-1,5)

        idx_x = indexes[3, :, :, None, None] # (N, 5, 1, 1)
        idx_y = indexes[4, :, None, :, None] # (N, 1, 5, 1)
        idx_z = indexes[5, :, None, None, :] # (N, 1, 1, 5)

        extracted_c = c_sliced[idx_x, idx_y, idx_z]
        vx = b3_vx[:, :, np.newaxis, np.newaxis]
        vy = b3_vy[:, np.newaxis, :, np.newaxis]
        vz = b3_vz[:, np.newaxis, np.newaxis, :]

        v_pred = extracted_c * vx * vy * vz
        v_pred = np.sum(v_pred.reshape(-1,125),axis=-1)

        c_sliced = c[..., 2]

        b3_wx = B_spline.beta3(x_w[:,0]/dx).reshape(-1,5)
        b3_wy = B_spline.beta3(x_w[:,1]/dy).reshape(-1,5)
        b3_wz = B_spline.beta4(x_w[:,2]/dz).reshape(-1,5)

        idx_x = indexes[6, :, :, None, None] # (N, 5, 1, 1)
        idx_y = indexes[7, :, None, :, None] # (N, 1, 5, 1)
        idx_z = indexes[8, :, None, None, :] # (N, 1, 1, 5)

        extracted_c = c_sliced[idx_x, idx_y, idx_z]
        wx = b3_wx[:, :, np.newaxis, np.newaxis]
        wy = b3_wy[:, np.newaxis, :, np.newaxis]
        wz = b3_wz[:, np.newaxis, np.newaxis, :]

        w_pred = extracted_c * wx * wy * wz
        w_pred = np.sum(w_pred.reshape(-1,125),axis=-1)

        return u_pred, v_pred, w_pred
    
    @staticmethod
    def velocity_pred_p(c, ix_u, iy_u, iz_u, ix_v, iy_v, iz_v, ix_w, iy_w, iz_w, dx, dy, dz, x_u, x_v, x_w):
        b3_ux = B_spline.beta4(x_u[:,0]/dx).reshape(-1,5)
        b3_uy = B_spline.beta3(x_u[:,1]/dy).reshape(-1,5)
        b3_uz = B_spline.beta3(x_u[:,2]/dz).reshape(-1,5)
        u_comp = jnp.zeros((b3_ux.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    u_comp_temp = (c[ix_u[:,i], iy_u[:,j], iz_u[:,k],0]*b3_ux[:,i]*b3_uy[:,j]*b3_uz[:,k])
                    u_comp = u_comp.at[:,i,j,k].set(u_comp_temp)

        u_pred = jnp.sum(u_comp.reshape(-1,125),1)

        b3_vx = B_spline.beta3(x_v[:,0]/dx).reshape(-1,5)
        b3_vy = B_spline.beta4(x_v[:,1]/dy).reshape(-1,5)
        b3_vz = B_spline.beta3(x_v[:,2]/dz).reshape(-1,5)
        v_comp = jnp.zeros((b3_vx.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    v_comp_temp = (c[ix_v[:,i], iy_v[:,j], iz_v[:,k],1]*b3_vx[:,i]*b3_vy[:,j]*b3_vz[:,k])
                    v_comp = v_comp.at[:,i,j,k].set(v_comp_temp)
        v_pred = jnp.sum(v_comp.reshape(-1,125),1)

        b3_wx = B_spline.beta3(x_w[:,0]/dx).reshape(-1,5)
        b3_wy = B_spline.beta3(x_w[:,1]/dy).reshape(-1,5)
        b3_wz = B_spline.beta4(x_w[:,2]/dz).reshape(-1,5)
        w_comp = jnp.zeros((b3_wx.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    w_comp_temp = (c[ix_w[:,i], iy_w[:,j], iz_w[:,k],2]*b3_wx[:,i]*b3_wy[:,j]*b3_wz[:,k])
                    w_comp = w_comp.at[:,i,j,k].set(w_comp_temp)
        w_pred = jnp.sum(w_comp.reshape(-1,125),1)
        return u_pred, v_pred, w_pred



    @staticmethod
    def velocity_grad(c, ix_u, iy_u, iz_u, ix_v, iy_v, iz_v, ix_w, iy_w, iz_w, dx, dy, dz, x_u, x_v, x_w, x_s):
        b3_ux1 = B_spline.beta4(x_v[:,0]/dx).reshape(-1,5)
        b3_ux2 = B_spline.beta4(x_s[:,0]/dx).reshape(-1,5)
        b3_uy = B_spline.beta3(x_u[:,1]/dy).reshape(-1,5)
        b3_uz = B_spline.beta3(x_u[:,2]/dz).reshape(-1,5)
        u_comp = jnp.zeros((b3_ux1.shape[0],)+(5,5,5))

        for i in range(5):
            for j in range(5):
                for k in range(5):
                    u_comp_temp = c[ix_u[:,i], iy_u[:,j], iz_u[:,k],0]*(b3_ux1[:,i]-b3_ux2[:,i])*b3_uy[:,j]*b3_uz[:,k]
                    u_comp[:,i,j,k] = u_comp_temp

        u_grad = jnp.sum(u_comp.reshape(-1,125),1)

        b3_vx = B_spline.beta3(x_v[:,0]/dx).reshape(-1,5)
        b3_vy1 = B_spline.beta4(x_u[:,1]/dy).reshape(-1,5)
        b3_vy2 = B_spline.beta4(x_s[:,1]/dy).reshape(-1,5)
        b3_vz = B_spline.beta3(x_v[:,2]/dz).reshape(-1,5)
        v_comp = jnp.zeros((b3_vx.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    v_comp_temp = c[ix_v[:,i], iy_v[:,j], iz_v[:,k],1]*b3_vx[:,i]*(b3_vy1[:,j]-b3_vy2[:,j])*b3_vz[:,k]
                    v_comp[:,i,j,k] = v_comp_temp
        v_grad = jnp.sum(v_comp.reshape(-1,125),1)

        b3_wx = B_spline.beta3(x_w[:,0]/dx).reshape(-1,5)
        b3_wy = B_spline.beta3(x_w[:,1]/dy).reshape(-1,5)
        b3_wz1 = B_spline.beta4(x_s[:,2]/dz).reshape(-1,5)
        b3_wz2 = B_spline.beta4(x_u[:,2]/dz).reshape(-1,5)
        w_comp = jnp.zeros((b3_wx.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    w_comp_temp = c[ix_w[:,i], iy_w[:,j], iz_w[:,k],2]*b3_wx[:,i]*b3_wy[:,j]*(b3_wz1[:,k]-b3_wz2[:,k])
                    w_comp[:,i,j,k] = w_comp_temp
        w_grad = jnp.sum(w_comp.reshape(-1,125),1)
        return u_grad, v_grad, w_grad
#%%
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from domain import *
    from projection import *
    from trackdata import *
    import os
    from time import time
    all_params = {"domain":{}, "data":{}, "projection":{}, "prediction":{}}
    domain_range= {'t': (0, 0.04), 'x':(-1.0, 1.0), 'y':(-1.0, 1.0), 'z':(-1.0, 1.0)}
    grid_size = [256,256,256]
    coeff_shape = (grid_size[0], grid_size[1], grid_size[2], 3)
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

#%% Velocity prediction process for training data at certain time step
    xc = np.linspace(all_params['domain']['domain_range']['x'][0], all_params['domain']['domain_range']['x'][1], all_params['domain']['grid_size'][0])
    yc = np.linspace(all_params['domain']['domain_range']['y'][0], all_params['domain']['domain_range']['y'][1], all_params['domain']['grid_size'][1])
    zc = np.linspace(all_params['domain']['domain_range']['z'][0], all_params['domain']['domain_range']['z'][1], all_params['domain']['grid_size'][2])
    dx = xc[1]-xc[0]
    dy = yc[1]-yc[0]
    dz = zc[1]-zc[0]

    vals, counts = np.unique(train_data['pos'][:,0], return_counts=True)
    #counts = np.concatenate([[0], counts])
    indexes_list = []
    x_u_list = []
    x_v_list = []
    x_w_list = []
    for i in range(time_length):
        indexes = VelocityPrediction3D.find_indexes(train_data['pos'][np.sum(counts[:i]):np.sum(counts[:i+1]),1:],xc, yc, zc, dx, dy, dz)
        x_u, x_v, x_w = VelocityPrediction3D.data_reshape(train_data['pos'][np.sum(counts[:i]):np.sum(counts[:i+1]),1:], *indexes, dx, dy, dz, xc, yc, zc)
        indexes_list.append(np.array(indexes))
        x_u_list.append(x_u)
        x_v_list.append(x_v)
        x_w_list.append(x_w)

    #u_pred, v_pred, w_pred = VelocityPrediction3D.velocity_pred_p(projection_params['coefficients'][2], *indexes, dx, dy, dz, x_u, x_v, x_w)
    u_pred_, v_pred_, w_pred_ = VelocityPrediction3D.velocity_pred(projection_params['coefficients'][2], indexes_list[2], dx, dy, dz, x_u_list[2], x_v_list[2], x_w_list[2])

# %% Check divergence free of predicted velocity field version 1 (Very slow)
    Xc, Yc, Zc = np.meshgrid(xc[1:64], yc[1:64], zc[1:64], indexing="ij")
    x_grid_data = np.stack([Xc, Yc, Zc], axis=-1)
    x_grid_data = x_grid_data.reshape(-1,3)
    indexes = VelocityPrediction3D.find_indexes(x_grid_data, xc, yc, zc, dx, dy, dz)
    x_u, x_v, x_w = VelocityPrediction3D.data_reshape(x_grid_data, *indexes, dx, dy, dz, xc, yc, zc)
    time_ = time()
    u_pred, v_pred, w_pred = VelocityPrediction3D.velocity_pred_p(projection_params['coefficients'][2], *indexes, dx, dy, dz, x_u, x_v, x_w)
    print(time()-time_)
    preds = np.concatenate([u_pred.reshape(63,63,63,1), v_pred.reshape(63,63,63,1), w_pred.reshape(63,63,63,1)], axis=-1)
    plt.imshow(FFTProjector.calculate_divergence_central(preds)[:,:,10])
    plt.colorbar()
    plt.show()

#%% Check divergence free of predicted velocity field version 1 (Fast with advanced indexing)
    indexes_list = []
    x_u_list = []
    x_v_list = []
    x_w_list = []
    for i in range(3):
        indexes = VelocityPrediction3D.find_indexes(x_grid_data, xc, yc, zc, dx, dy, dz)
        x_u, x_v, x_w = VelocityPrediction3D.data_reshape(x_grid_data, *indexes, dx, dy, dz, xc, yc, zc)
        indexes_list.append(np.array(indexes))
        x_u_list.append(x_u)
        x_v_list.append(x_v)
        x_w_list.append(x_w)
    time_ = time()
    u_pred_, v_pred_, w_pred_ = VelocityPrediction3D.velocity_pred(projection_params['coefficients'][2], indexes_list[2], dx, dy, dz, x_u_list[2], x_v_list[2], x_w_list[2])
    print(time()-time_)
    preds_ = np.concatenate([u_pred_.reshape(63,63,63,1), v_pred_.reshape(63,63,63,1), w_pred_.reshape(63,63,63,1)], axis=-1)
    plt.imshow(FFTProjector.calculate_divergence_central(preds_)[:,:,10])
    plt.colorbar()
    plt.show()
    plt.imshow(preds_[:,:,10,0])
    plt.colorbar()
    plt.show()
    plt.imshow(projection_params['coefficients'][2][:,:,10])
    plt.colorbar()
    plt.show()
# %%
    plt.imshow(preds_[:,:,10,0])
    plt.colorbar()
    plt.show()
    plt.imshow(projection_params['coefficients'][2][:,:,10,0])
    plt.colorbar()
    plt.show()
# %%
    print(projection_params['coefficients'][2][:,:,10].shape)
# %%
