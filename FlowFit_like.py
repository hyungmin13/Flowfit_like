#%%
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from functools import partial
from domain import *
from projection import *
from velocity_pred import *
from B_spline import *
from typing import Any
from flax import struct
from jax import random
from jax import value_and_grad
import optax
import jax
#%%
class FlowFitmodel(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)
    
@partial(jax.jit, static_argnums=(1, 2, 5, 20, 21))
def FlowFit_update(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, index_list, index_mask, dx, dy, dz, 
                   xu_list, xu_mask, xv_list, xv_mask, xw_list, xw_mask, particle_vel, particle_vel_mask, particle_acc, projection_fn, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)

    def get_in_axis(leaf):
        if hasattr(leaf, 'ndim') and leaf.ndim > 0:
            batch_size = dynamic_params.shape[0]
            for axis, size in enumerate(leaf.shape):
                if size == batch_size: return axis
        return None

    def get_out_axis(leaf):
        in_ax = get_in_axis(leaf)
        if in_ax is not None: return in_ax
        if hasattr(leaf, 'ndim'): return 0
        return None

    state_in_axes = jax.tree_util.tree_map(get_in_axis, model_states)
    state_out_axes = jax.tree_util.tree_map(get_out_axis, model_states)

    def single_update(d_p, state, idx, xu, xv, xw, pv):
        def local_loss(p):
            return equation_fn(p, all_params, idx, dx, dy, dz, xu, xv, xw, pv, model_fn)
        val, grad = jax.value_and_grad(local_loss)(d_p)
        print("Grad : ",grad.shape)
        grad_p = projection_fn(grad)[0]
        updates, new_state = optimiser_fn(grad_p, state, d_p, value=val, grad=grad, value_fn=local_loss)
        new_param = optax.apply_updates(d_p, updates)
        return new_param, new_state, val

    vmap_fn = jax.vmap(single_update, in_axes=(0, state_in_axes, 0, 0, 0, 0, 0), out_axes=(0, state_out_axes, 0))
    
    new_params, new_states, lossvals = vmap_fn(dynamic_params, model_states, index_list, xu_list, xv_list, xw_list, particle_vel)

    def fix_scalars(leaf, in_axis):
        if in_axis is None and hasattr(leaf, 'ndim') and leaf.ndim > 0:
            return leaf[0] 
        return leaf

    new_states = jax.tree_util.tree_map(fix_scalars, new_states, state_in_axes)

    return lossvals, new_states, new_params


class FlowFitbase:
    def __init__(self, c):
        c.get_outdirs()
        c.save_constants_file()
        self.c = c

class FlowFit3(FlowFitbase):
    def train(self):
        def safe_stack(*args):
            # 입력된 요소가 jax array(또는 넘파이)인 경우에만 stack 수행
            if isinstance(args[0], (jnp.ndarray, jax.Array)):
                return jnp.stack(args)
            # 데이터가 없는 빈 상태(EmptyState 등)인 경우 그냥 첫 번째 요소를 반환
            return args[0]
        all_params = {"domain":{}, "data":{}, "projection":{}, "prediction":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        all_params["projection"] = self.c.projection.init_params(**self.c.projection_init_kwargs)
        all_params["prediction"] = self.c.prediction.init_params(**self.c.prediction_init_kwargs)
        
        global_key = random.PRNGKey(42)
        key, projection_key = random.split(global_key)
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = optax.lbfgs()

        model_states = optimiser.init(all_params["projection"]['coefficients'])

        optimiser_fn = optimiser.update
        model_fn = self.c.prediction.velocity_pred
        dynamic_params = all_params["projection"].pop("coefficients")
        equation_fn = self.c.equation.Loss
        report_fn = self.c.equation.Loss_report
        projection_fn = self.c.projection.helmholtz_hodge_decomposition
        all_params, p_p, p_u, p_v, p_w = self.c.domain.generate_staggered_p(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        

        xc = np.linspace(all_params['domain']['domain_range']['x'][0], all_params['domain']['domain_range']['x'][1], all_params['domain']['grid_size'][0])
        yc = np.linspace(all_params['domain']['domain_range']['y'][0], all_params['domain']['domain_range']['y'][1], all_params['domain']['grid_size'][1])
        zc = np.linspace(all_params['domain']['domain_range']['z'][0], all_params['domain']['domain_range']['z'][1], all_params['domain']['grid_size'][2])
        dx = xc[1]-xc[0]
        dy = yc[1]-yc[0]
        dz = zc[1]-zc[0]

        vals, counts = np.unique(train_data['pos'][:,0], return_counts=True)
        index_list = []
        xu_list = []
        xv_list = []
        xw_list = []
        particle_vel = []
        particle_acc = []
        for i in range(len(vals)):
            indexes = VelocityPrediction3D.find_indexes(train_data['pos'][np.sum(counts[:i]):np.sum(counts[:i+1]),1:],xc, yc, zc, dx, dy, dz)
            xu, xv, xw = VelocityPrediction3D.data_reshape(train_data['pos'][np.sum(counts[:i]):np.sum(counts[:i+1]),1:], *indexes, dx, dy, dz, xc, yc, zc)
            particle_vel.append(train_data['vel'][np.sum(counts[:i]):np.sum(counts[:i+1]),:])
            particle_acc.append(train_data['acc'][np.sum(counts[:i]):np.sum(counts[:i+1]),:])
            index_list.append(np.array(indexes))
            xu_list.append(xu)
            xv_list.append(xv)
            xw_list.append(xw)
        max_n = max(x.shape[0] for x in xu_list)
        max_n2 = max(x.shape[1] for x in index_list)
        max_n3 = max(x.shape[0] for x in particle_vel)
        def pad_array(arr, target_shape):
            pad_width = [(0, target_shape - arr.shape[0])] + [(0, 0)] * (len(arr.shape) - 1)
            return jnp.pad(arr, pad_width), (jnp.arange(target_shape) < arr.shape[0])
        def pad_array2(arr, target_shape):
            n_i = arr.shape[1]
            pad_size = target_shape - n_i
            padded_arr = jnp.pad(arr, ((0, 0), (0, pad_size), (0, 0)), mode='constant')
            mask = jnp.zeros((arr.shape[0], target_shape), dtype=jnp.float32)
            mask = mask.at[:, :n_i].set(1.0)
            return padded_arr, mask

        particle_vel_padded, particle_vel_mask = zip(*[pad_array(x, max_n3) for x in particle_vel])
        xu_padded, xu_mask = zip(*[pad_array(x, max_n) for x in xu_list])
        xv_padded, xv_mask = zip(*[pad_array(x, max_n) for x in xv_list])
        xw_padded, xw_mask = zip(*[pad_array(x, max_n) for x in xw_list])
        particle_vel = jnp.stack(particle_vel_padded)[:3,:,:]
        particle_vel_mask = jnp.stack(particle_vel_mask)[:3,:]
        index_padded, index_mask = zip(*[pad_array2(x, max_n2) for x in index_list])

        xu_list = jnp.stack(xu_padded)[:3,:,:]
        xv_list = jnp.stack(xv_padded)[:3,:,:]
        xw_list = jnp.stack(xw_padded)[:3,:,:]
        
        xu_mask = jnp.stack(xu_mask)[:3,:]
        xv_mask = jnp.stack(xv_mask)[:3,:]
        xw_mask = jnp.stack(xw_mask)[:3,:]
        print(xu_mask)
        print(xu_mask.shape)
        index_list = jnp.stack(index_padded)[:3,:,:,:]
        index_mask = jnp.stack(index_mask)[:3,:,:]
        
        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        static_params = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        static_leaves = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        static_keys = (static_leaves, treedef)
        update = FlowFit_update.lower(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, index_list, index_mask, dx, dy, dz, 
                   xu_list, xu_mask, xv_list, xv_mask, xw_list, xw_mask, particle_vel, particle_vel_mask, particle_acc, projection_fn, model_fn).compile()
        
        while 1:
            lossvals, model_states, new_params_list = update(model_states, dynamic_params, static_params, index_list, index_mask, dx, dy, dz, 
                                                                xu_list, xu_mask, xv_list, xv_mask, xw_list, xw_mask, particle_vel, particle_vel_mask, particle_acc)
            print('check')
        return
#%%
if __name__ == "__main__":
    from domain import *
    from trackdata import *
    from projection import *
    from constants import *
    from equation import *
    from txt_reader import *
    import argparse

    parser = argparse.ArgumentParser(description='FlowFit')
    parser.add_argument('-n', '--name', type=str, help='run name', default='HIT_k1')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='test_txt')
    args = parser.parse_args()
    cur_dir = os.getcwd()
    input_txt = cur_dir + '/' + args.config + '.txt' 
    data = parse_tree_structured_txt(input_txt)
    c = Constants(**data)

    run = FlowFit3(c)
    run.train()
