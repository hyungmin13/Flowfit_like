import numpy as np
import jax.numpy as jnp
from B_spline import *
from scipy.spatial import KDTree

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
    def find_indexes(data, coefficient):
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


    @staticmethod
    def velocity_pred(c, ix_u, iy_u, iz_u, ix_v, iy_v, iz_v, ix_w, iy_w, iz_w, dx, dy, dz, x_u, x_v, x_w):
        b3_ux = B_spline.beta4(x_u[:,0]/dx).reshape(-1,5)
        b3_uy = B_spline.beta3(x_u[:,1]/dy).reshape(-1,5)
        b3_uz = B_spline.beta3(x_u[:,2]/dz).reshape(-1,5)
        u_comp = jnp.zeros((b3_ux.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    u_comp_temp = c[ix_u[:,i], iy_u[:,j], iz_u[:,k],0]*b3_ux[:,i]*b3_uy[:,j]*b3_uz[:,k]
                    u_comp[:,i,j,k] = u_comp_temp

        u_pred = jnp.sum(u_comp.reshape(-1,125),1)

        b3_vx = B_spline.beta3(x_v[:,0]/dx).reshape(-1,5)
        b3_vy = B_spline.beta4(x_v[:,1]/dy).reshape(-1,5)
        b3_vz = B_spline.beta3(x_v[:,2]/dz).reshape(-1,5)
        v_comp = jnp.zeros((b3_vx.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    v_comp_temp = c[ix_v[:,i], iy_v[:,j], iz_v[:,k],1]*b3_vx[:,i]*b3_vy[:,j]*b3_vz[:,k]
                    v_comp[:,i,j,k] = v_comp_temp
        v_pred = jnp.sum(v_comp.reshape(-1,125),1)

        b3_wx = B_spline.beta3(x_w[:,0]/dx).reshape(-1,5)
        b3_wy = B_spline.beta3(x_w[:,1]/dy).reshape(-1,5)
        b3_wz = B_spline.beta4(x_w[:,2]/dz).reshape(-1,5)
        w_comp = jnp.zeros((b3_wx.shape[0],)+(5,5,5))
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    w_comp_temp = c[ix_w[:,i], iy_w[:,j], iz_w[:,k],2]*b3_wx[:,i]*b3_wy[:,j]*b3_wz[:,k]
                    w_comp[:,i,j,k] = w_comp_temp
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

if __name__=="__main__":
    import matplotlib.pyplot as plt
