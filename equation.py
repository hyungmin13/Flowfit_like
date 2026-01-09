#%%
import jax
import jax.numpy as jnp


class Equation:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

class Linear(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, indexes, dx, dy, dz, x_u, x_v, x_z, particle_vel, model_fns):
        all_params["projection"]["coefficients"] = dynamic_params
        u_pred, v_pred, w_pred = model_fns(dynamic_params, indexes, dx, dy, dz, x_u, x_v, x_z) 
        
        loss_u = u_pred - particle_vel[:,0]
        loss_v = v_pred - particle_vel[:,1]
        loss_w = w_pred - particle_vel[:,2]

        loss_u = jnp.mean(loss_u**2)
        loss_v = jnp.mean(loss_v**2)
        loss_w = jnp.mean(loss_w**2)
        
        total_loss = loss_u + loss_v + loss_w
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, indexes, dx, dy, dz, x_u, x_v, x_z, particle_vel, model_fns):
        all_params["projection"]["coefficients"] = dynamic_params
        u_pred, v_pred, w_pred = model_fns(dynamic_params, indexes, dx, dy, dz, x_u, x_v, x_z) 
        loss_u = u_pred - particle_vel[:,0:1]
        loss_v = v_pred - particle_vel[:,1:2]
        loss_w = w_pred - particle_vel[:,2:3]

        loss_u = jnp.mean(loss_u**2)
        loss_v = jnp.mean(loss_v**2)
        loss_w = jnp.mean(loss_w**2)

        total_loss = loss_u + loss_v + loss_w
        return total_loss

