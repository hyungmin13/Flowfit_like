#%%
import numpy as np
from B_spline import *
from domain import *
from jax import random
class Projector:
    @staticmethod
    def __init__(*args):
        raise NotImplementedError

class FFTProjector(Projector):
    def __init__(self, all_params):
        self.all_params = all_params

    @staticmethod
    def init_params(key, coeff_shape, time_length):
        project_key = random.PRNGKey(key)
        keys = random.split(project_key, num = time_length)

        c0s = [random.normal(key = keys[i], shape=coeff_shape)*0.1 for i in range(time_length)]
        coefficients = np.concatenate([np.expand_dims(FFTProjector.helmholtz_hodge_decomposition(c0s[i])[0],axis=0) for i in range(len(c0s))])
        
        projection_params = {"coefficients": coefficients, "time_legth": time_length}
        return projection_params
    
    @staticmethod
    def solve_discrete_poisson_fft(b, h=1.0):
        N = b.shape[0]

        idx = jnp.arange(N)
        k_discrete = 2 * (jnp.cos(2 * jnp.pi * idx / N) - 1) / (h**2)

        kx, ky, kz = jnp.meshgrid(k_discrete, k_discrete, k_discrete, indexing='ij')
        
        L_hat = kx + ky + kz
        
        b_hat = jnp.fft.fftn(b)
        
        #L_hat[0, 0, 0] = 1.0
        L_hat = L_hat.at[0,0,0].set(1.0)
        #b_hat[0, 0, 0] = 0.0
        b_hat = b_hat.at[0,0,0].set(0.0)
        phi_hat = b_hat / L_hat
        phi = jnp.real(jnp.fft.ifftn(phi_hat))
        
        return phi
    
    @staticmethod
    def calculate_divergence_backward(F, h=1.0):
        Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
        dFx = (Fx - jnp.roll(Fx, 1, axis=0)) / h
        dFy = (Fy - jnp.roll(Fy, 1, axis=1)) / h
        dFz = (Fz - jnp.roll(Fz, 1, axis=2)) / h
        return dFx + dFy + dFz
    
    @staticmethod
    def calculate_divergence_central(F, h=1.0):
        Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
        dFx = (jnp.roll(Fx, 1, axis=0) - jnp.roll(Fx, 1, axis=0)) / (2*h)
        dFy = (jnp.roll(Fy, 1, axis=1) - jnp.roll(Fy, 1, axis=1)) / (2*h)
        dFz = (jnp.roll(Fz, 1, axis=2) - jnp.roll(Fz, 1, axis=2)) / (2*h)
        return dFx + dFy + dFz
    
    @staticmethod
    def calculate_gradient_forward(phi, h=1.0):
        dphi_dx = (jnp.roll(phi, -1, axis=0) - phi) / h
        dphi_dy = (jnp.roll(phi, -1, axis=1) - phi) / h
        dphi_dz = (jnp.roll(phi, -1, axis=2) - phi) / h
        return jnp.stack((dphi_dx, dphi_dy, dphi_dz), axis=-1)
    
    @staticmethod
    def helmholtz_hodge_decomposition(F):

        div_F = FFTProjector.calculate_divergence_backward(F)
        mean_div_F = jnp.mean(div_F)
        b = div_F - mean_div_F
        phi = FFTProjector.solve_discrete_poisson_fft(b)
        F_irr = FFTProjector.calculate_gradient_forward(phi)
        F_sol = F - F_irr

        return F_sol, F_irr

if __name__=='__main__':
    import matplotlib.pyplot as plt

    all_params = {"domain":{}, "data":{}, "coefficient":{}}
    domain_range= {'x':(-1.0, 1.0), 'y':(-1.0, 1.0), 'z':(-1.0, 1.0)}
    grid_size = [256,256,256]
    coeff_shape = (grid_size[0], grid_size[1], grid_size[2], 3)
    time_length = 10
    key = 42
    all_params["domain"] = Domain.init_params(domain_range = domain_range,
                                              grid_size = grid_size)    
    all_params, p_p, p_u, p_v, p_w = Domain.generate_staggered_p(all_params)
    projection_params = FFTProjector.init_params(key, coeff_shape, time_length)

    plt.imshow(FFTProjector.calculate_divergence_central(projection_params['coefficients'][5])[:,:,10])
    plt.colorbar()
    plt.show()
