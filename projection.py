#%%
import numpy as np
from B_spline import *
from domain import *

class Projector:
    @staticmethod
    def __init__(*args):
        raise NotImplementedError

class FFTProjector(Projector):
    def __init__(self, all_params):
        self.all_params = all_params
    @staticmethod
    def solve_discrete_poisson_fft(b, h=1.0):
        N = b.shape[0]

        idx = np.arange(N)
        k_discrete = 2 * (np.cos(2 * np.pi * idx / N) - 1) / (h**2)

        kx, ky, kz = np.meshgrid(k_discrete, k_discrete, k_discrete, indexing='ij')
        
        L_hat = kx + ky + kz
        
        b_hat = np.fft.fftn(b)
        
        L_hat[0, 0, 0] = 1.0
        b_hat[0, 0, 0] = 0.0

        phi_hat = b_hat / L_hat
        phi = np.real(np.fft.ifftn(phi_hat))
        
        return phi
    
    @staticmethod
    def calculate_divergence_backward(F, h=1.0):
        Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
        dFx = (Fx - np.roll(Fx, 1, axis=0)) / h
        dFy = (Fy - np.roll(Fy, 1, axis=1)) / h
        dFz = (Fz - np.roll(Fz, 1, axis=2)) / h
        return dFx + dFy + dFz
    
    @staticmethod
    def calculate_divergence_central(F, h=1.0):
        Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
        dFx = (np.roll(Fx, 1, axis=0) - np.roll(Fx, 1, axis=0)) / (2*h)
        dFy = (np.roll(Fy, 1, axis=1) - np.roll(Fy, 1, axis=1)) / (2*h)
        dFz = (np.roll(Fz, 1, axis=2) - np.roll(Fz, 1, axis=2)) / (2*h)
        return dFx + dFy + dFz
    
    @staticmethod
    def calculate_gradient_forward(phi, h=1.0):
        dphi_dx = (np.roll(phi, -1, axis=0) - phi) / h
        dphi_dy = (np.roll(phi, -1, axis=1) - phi) / h
        dphi_dz = (np.roll(phi, -1, axis=2) - phi) / h
        return np.stack((dphi_dx, dphi_dy, dphi_dz), axis=-1)
    
    @staticmethod
    def helmholtz_hodge_decomposition(F):

        div_F = FFTProjector.calculate_divergence_backward(F)
        mean_div_F = np.mean(div_F)
        b = div_F - mean_div_F
        phi = FFTProjector.solve_discrete_poisson_fft(b)
        F_irr = FFTProjector.calculate_gradient_forward(phi)
        F_sol = F - F_irr

        return F_sol, F_irr

if __name__=='__main__':
    import matplotlib.pyplot as plt
    I = J = K = 32  
    grid_shape = (I, J, K, 3)

    xc = np.linspace(-1.0, 1.0, I)
    yc = np.linspace(-1.0, 1.0, J)
    zc = np.linspace(-1.0, 1.0, K)
    dx = float(xc[1] - xc[0])
    dy = float(yc[1] - yc[0])
    dz = float(zc[1] - zc[0])
    X,Y,Z = np.meshgrid(xc, yc, zc, indexing='ij')
    c0 = np.random.normal(size=grid_shape) * 0.1

    F=c0
    F_solenoidal = F.copy()
    F_solenoidal, F_irrotational = FFTProjector.helmholtz_hodge_decomposition(F_solenoidal)

    plt.imshow(FFTProjector.calculate_divergence_central(F_solenoidal)[:,:,10])
    plt.colorbar()
    plt.show()
# %%
