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
    def Loss(dynamic_params, all_params, indexes, B_val, particle_vel, model_fns):
        all_params["projection"]["coefficients"] = dynamic_params
        u_pred, v_pred, w_pred = model_fns(dynamic_params, indexes, B_val) 
        
        loss_u = u_pred - particle_vel[:,0]
        loss_v = v_pred - particle_vel[:,1]
        loss_w = w_pred - particle_vel[:,2]

        loss_u = jnp.sum(loss_u**2)
        loss_v = jnp.sum(loss_v**2)
        loss_w = jnp.sum(loss_w**2)
        
        #hp_filter = jnp.array([-0.25, 0.5, -0.25])*1e-2
        sum_of_squared_errors = 0.0
        for k in range(3):
            vel_comp = dynamic_params[...,k]
            hp_x = 0.5 * vel_comp[1:-1, :, :] - 0.25 * (vel_comp[:-2, :, :] + vel_comp[2:, :, :])
            hp_y = 0.5 * vel_comp[:, 1:-1, :] - 0.25 * (vel_comp[:, :-2, :] + vel_comp[:, 2:, :])
            hp_z = 0.5 * vel_comp[:, :, 1:-1] - 0.25 * (vel_comp[:, :, :-2] + vel_comp[:, :, 2:])
            sum_of_squared_errors += jnp.sum(hp_x**2) + jnp.sum(hp_y**2) + jnp.sum(hp_z**2)

        total_loss = loss_u + loss_v + loss_w + 0.5 * sum_of_squared_errors
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, indexes, B_val, particle_vel, model_fns):
        all_params["projection"]["coefficients"] = dynamic_params
        u_pred, v_pred, w_pred = model_fns(dynamic_params, indexes, B_val) 
        loss_u = u_pred - particle_vel[:,0:1]
        loss_v = v_pred - particle_vel[:,1:2]
        loss_w = w_pred - particle_vel[:,2:3]

        loss_u = jnp.sum(loss_u**2)
        loss_v = jnp.sum(loss_v**2)
        loss_w = jnp.sum(loss_w**2)

        #hp_filter = jnp.array([-0.25, 0.5, -0.25])*1e-2
        sum_of_squared_errors = 0.0
        for k in range(3):
            vel_comp = dynamic_params[...,k]
            hp_x = 0.5 * vel_comp[1:-1, :, :] - 0.25 * (vel_comp[:-2, :, :] + vel_comp[2:, :, :])
            hp_y = 0.5 * vel_comp[:, 1:-1, :] - 0.25 * (vel_comp[:, :-2, :] + vel_comp[:, 2:, :])
            hp_z = 0.5 * vel_comp[:, :, 1:-1] - 0.25 * (vel_comp[:, :, :-2] + vel_comp[:, :, 2:])
            sum_of_squared_errors += jnp.sum(hp_x**2) + jnp.sum(hp_y**2) + jnp.sum(hp_z**2)

        total_loss = loss_u + loss_v + loss_w + 0.5 * sum_of_squared_errors
        return total_loss, loss_u, loss_v, loss_w, sum_of_squared_errors

