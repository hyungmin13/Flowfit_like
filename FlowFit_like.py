#%%
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from domain import *
from projection import *
from velocity_pred import *
from B_spline import *
from typing import Any
from flax import struct
from jax import random
import optax
import jax
#%%
class FlowFitmodel(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)
    
@partial(jax.jit, static_arnums=())
def FlowFit_update(model_states,):
    return

class FlowFitbase:
    def __init__(self, c):
        c.get_outdirs()
        c.save_constants_file()
        self.c = c

class FlowFit3(FlowFitbase):
    def train(self):
        all_params = {"domain":{}, "data":{}, "projection":{}, "prediction":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        all_params["projection"] = self.c.projection.init_params(**self.c.projection_init_kwargs)
        all_params["prediction"] = self.c.prediction.init_params(**self.c.prediction_init_kwargs)
        
        global_key = random.PRNGKey(42)
        key, coefficient_key = random.split(global_key)
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate, b1=0.95, b2=0.95,
                                                                 weight_decay=0.01, precondition_frequency=5)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = self.c.network.network_fn
        dynamic_params = all_params["network"].pop("layers")
        equation_fn = self.c.equation.Loss
        report_fn = self.c.equation.Loss_report
        all_params, p_p, p_u, p_v, p_w = self.c.domain.generate_staggered_p(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        c0 = np.random.normal(size=p_p.sape)*0.1
        c_sol, c_irrot = self.c.projection.helmholtz_hodge_decomposition(c0)

        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        static_params = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        static_leaves = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        static_keys = (static_leaves, treedef)

        return
#%%
if __name__ == "__main__":

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
    F_solenoidal, F_irrotational = helmholtz_hodge_decomposition(F_solenoidal, N)


# %%



    xt = np.linspace(-0.5, 0.5, I)
    yt = np.linspace(-0.5, 0.5, J)
    zt = np.linspace(-0.5, 0.5, K)
    Xt, Yt, Zt = np.meshgrid(xt, yt, zt, indexing="ij")
    p_u, p_v, p_w = generate_staggered_p(xc, yc, zc, dx, dy, dz)
    p_c = np.stack([Xt, Yt, Zt], axis=-1)
    c_flat = F_solenoidal.reshape(-1)

    tree = KDTree(np.c_[p_u[...,0].ravel(), p_u[...,1].ravel(), p_u[...,2].ravel()])
    x_data = p_c.reshape(-1,3)
    _, ii = tree.query(x_data)

    val, counts = np.unique(ii,return_counts=True)

    tree_xu = KDTree(xc.reshape(-1,1)+dx/2)
    tree_yu = KDTree(yc.reshape(-1,1))
    tree_zu = KDTree(zc.reshape(-1,1))

    _, i_xu = tree_xu.query(x_data[:,0:1],k=5)
    _, i_yu = tree_yu.query(x_data[:,1:2],k=5)
    _, i_zu = tree_zu.query(x_data[:,2:3],k=5)

    tree_xv = KDTree(xc.reshape(-1,1))
    tree_yv = KDTree(yc.reshape(-1,1)+dy/2)
    tree_zv = KDTree(zc.reshape(-1,1))

    _, i_xv = tree_xv.query(x_data[:,0:1],k=5)
    _, i_yv = tree_yv.query(x_data[:,1:2],k=5)
    _, i_zv = tree_zv.query(x_data[:,2:3],k=5)

    tree_xw = KDTree(xc.reshape(-1,1))
    tree_yw = KDTree(yc.reshape(-1,1))
    tree_zw = KDTree(zc.reshape(-1,1)+dz/2)

    _, i_xw = tree_xw.query(x_data[:,0:1],k=5)
    _, i_yw = tree_yw.query(x_data[:,1:2],k=5)
    _, i_zw = tree_zw.query(x_data[:,2:3],k=5)

    tree_xs = KDTree(xc.reshape(-1,1)+dx)
    tree_ys = KDTree(yc.reshape(-1,1)+dy)
    tree_zs = KDTree(zc.reshape(-1,1)+dz)

    _, i_xs = tree_xu.query(x_data[:,0:1],k=5)
    _, i_ys = tree_yu.query(x_data[:,1:2],k=5)
    _, i_zs = tree_zu.query(x_data[:,2:3],k=5)


#%%
    x_u = np.concatenate([x_data[:,None,0:1]-xc[i_xu.reshape(-1)].reshape(-1,5)[:,:,None]+dx/2,
                                x_data[:,None,1:2]-yc[i_yu.reshape(-1)].reshape(-1,5)[:,:,None],
                                x_data[:,None,2:3]-zc[i_zu.reshape(-1)].reshape(-1,5)[:,:,None]],2)
    x_u_ = x_u.reshape(-1,3)
    x_v = np.concatenate([x_data[:,None,0:1]-xc[i_xv.reshape(-1)].reshape(-1,5)[:,:,None],
                                x_data[:,None,1:2]-yc[i_yv.reshape(-1)].reshape(-1,5)[:,:,None]+dy/2,
                                x_data[:,None,2:3]-zc[i_zv.reshape(-1)].reshape(-1,5)[:,:,None]],2)
    x_v_ = x_v.reshape(-1,3)
    x_w = np.concatenate([x_data[:,None,0:1]-xc[i_xw.reshape(-1)].reshape(-1,5)[:,:,None],
                                x_data[:,None,1:2]-yc[i_yw.reshape(-1)].reshape(-1,5)[:,:,None],
                                x_data[:,None,2:3]-zc[i_zw.reshape(-1)].reshape(-1,5)[:,:,None]+dz/2],2)
    x_w_ = x_w.reshape(-1,3)
    x_s = np.concatenate([x_data[:,None,0:1]-xc[i_xu.reshape(-1)].reshape(-1,5)[:,:,None]+dx,
                                x_data[:,None,1:2]-yc[i_yu.reshape(-1)].reshape(-1,5)[:,:,None]+dy,
                                x_data[:,None,2:3]-zc[i_zu.reshape(-1)].reshape(-1,5)[:,:,None]+dz],2)
    x_s_ = x_s.reshape(-1,3)
#%%
    u_preds = v_c3(F_solenoidal, i_xu, i_yu, i_zu, i_xv, i_yv, i_zv, i_xw, i_yw, i_zw, dx, dy, dz, x_u_.copy(), x_v_.copy(), x_w_.copy())
    u_grads = v_grad3(F_solenoidal, i_xu, i_yu, i_zu, i_xv, i_yv, i_zv, i_xw, i_yw, i_zw, dx, dy, dz, x_u_.copy(), x_v_.copy(), x_w_.copy(), x_s_.copy())
#%%
    u_grad = np.concatenate([u_grads[0].reshape(32,32,32,1), u_grads[1].reshape(32,32,32,1), u_grads[2].reshape(32,32,32,1)],-1)
    u_pred = np.concatenate([u_preds[0].reshape(32,32,32,1), u_preds[1].reshape(32,32,32,1), u_preds[2].reshape(32,32,32,1)],-1)
# %%
    plt.imshow(u_grads[0].reshape(32,32,32)[:,:,10])
    plt.colorbar()
    plt.show()
# %%
    print(u_grads.shape)
# %%
