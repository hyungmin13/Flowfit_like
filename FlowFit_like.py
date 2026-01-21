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
from flax.serialization import to_state_dict, from_state_dict
#%%
class FlowFitmodel(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)
    
@partial(jax.jit, static_argnums=(1, 2, 5, 13, 14))
def FlowFit_update(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, index_list, index_mask, B_val_list, 
                   B_val_mask, particle_vel, particle_vel_mask, particle_acc, projection_fn, model_fn):
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

    def single_update(d_p, state, idx, B_val, pv):
        def local_loss(p):
            return equation_fn(p, all_params, idx, B_val, pv, model_fn)
        val, grad = jax.value_and_grad(local_loss)(d_p)
        grad_projection = projection_fn(grad)[0]
        updates, new_state = optimiser_fn(grad_projection, state, d_p, value=val, grad=grad_projection, value_fn=local_loss)
        new_param = optax.apply_updates(d_p, updates)
        return new_param, new_state, val

    vmap_fn = jax.vmap(single_update, in_axes=(0, state_in_axes, 0, 0, 0, 0, 0), out_axes=(0, state_out_axes, 0))
    #dynamic_zero = dynamic_params[0,:,:,:,:]
    #lossval, grad = value_and_grad(equation_fn, argnums=0)(dynamic_params, all_params, index_list[0,:,:,:], dx, dy, dz, xu_list[0,:,:], xv_list[0,:,:], xw_list[0,:,:], particle_vel[0,:,:], model_fn)
    #updates, new_state = optimiser_fn(grad, model_states, dynamic_params)
    #dynamic_params = optax.apply_updates(dynamic_params, updates)
    new_params, new_states, lossvals = vmap_fn(dynamic_params, model_states, index_list, B_val_list, particle_vel)
    
    def fix_scalars(leaf, in_axis):
        if in_axis is None and hasattr(leaf, 'ndim') and leaf.ndim > 0:
            return leaf[0] 
        return leaf

    new_states = jax.tree_util.tree_map(fix_scalars, new_states, state_in_axes)
    #return lossval, new_state, dynamic_params
    return lossvals, new_states, new_params


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
        key, projection_key = random.split(global_key)
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = optax.adam(1e-3)

        model_states = optimiser.init(all_params["projection"]['coefficients'])

        optimiser_fn = optimiser.update
        model_fn = self.c.prediction.velocity_pred
        dynamic_params = all_params["projection"].pop("coefficients")
        #dynamic_params = dynamic_params[0,:,:,:,:]
        equation_fn = self.c.equation.Loss
        report_fn = self.c.equation.Loss_report
        projection_fn = self.c.projection.helmholtz_hodge_decomposition
        all_params, p_p, p_u, p_v, p_w = self.c.domain.generate_staggered_p(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        

        xc = np.linspace(all_params['domain']['domain_range']['x'][0], all_params['domain']['domain_range']['x'][1], all_params['domain']['grid_size'][0]-4)
        yc = np.linspace(all_params['domain']['domain_range']['y'][0], all_params['domain']['domain_range']['y'][1], all_params['domain']['grid_size'][1]-4)
        zc = np.linspace(all_params['domain']['domain_range']['z'][0], all_params['domain']['domain_range']['z'][1], all_params['domain']['grid_size'][2]-4)
        dx = xc[1]-xc[0]
        dy = yc[1]-yc[0]
        dz = zc[1]-zc[0]
        pos = np.concatenate([train_data['pos'][:,1:2]/dx, train_data['pos'][:,2:3]/dy, train_data['pos'][:,3:4]/dz],1) + 2
        _, counts = np.unique(train_data['pos'][:,0], return_counts=True)
        pos_f = np.floor(pos)
        pos_n = pos - pos_f

        index_list = []
        B_val_lists = []
        particle_vel = []
        particle_acc = []
        funcs = [B_spline.beta4, B_spline.beta3, B_spline.beta3, 
                B_spline.beta3, B_spline.beta4, B_spline.beta3,
                B_spline.beta3, B_spline.beta3, B_spline.beta4]
        
        for i in range(len(counts)):
            indexes = VelocityPrediction3D.find_indexes(pos[np.sum(counts[:i]):np.sum(counts[:i+1]),1:])
            B_val = []
            for j in range(9):
                B_val.append(funcs[j](pos_n[np.sum(counts[:i]):np.sum(counts[:i+1]),j%3]))
            B_val = np.vstack(B_val).transpose()
            B_val_lists.append(B_val)
            particle_vel.append(train_data['vel'][np.sum(counts[:i]):np.sum(counts[:i+1]),:])
            particle_acc.append(train_data['acc'][np.sum(counts[:i]):np.sum(counts[:i+1]),:])
            index_list.append(np.array(indexes))

        max_n = max(x.shape[0] for x in B_val_lists)
        max_n2 = max(x.shape[0] for x in index_list)
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
        particle_vel = jnp.stack(particle_vel_padded)[:all_params["projection"]['time_length'],:,:]
        particle_vel_mask = jnp.stack(particle_vel_mask)[:all_params["projection"]['time_length'],:]
        B_val_padded, B_val_mask = zip(*[pad_array(x, max_n) for x in B_val_list])
        B_val_list = jnp.stack(B_val_padded)[:all_params["projection"]['time_length'],:,:]
        B_val_mask = jnp.stack(B_val_mask)[:all_params["projection"]['time_length'],:]

        index_padded, index_mask = zip(*[pad_array2(x, max_n2) for x in index_list])
        index_list = jnp.stack(index_padded)[:all_params["projection"]['time_length'],:,:]
        index_mask = jnp.stack(index_mask)[:all_params["projection"]['time_length'],:]
        
        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        static_params = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        static_leaves = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        static_keys = (static_leaves, treedef)
        update = FlowFit_update.lower(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, index_list, index_mask, B_val_list,
                                      B_val_mask, particle_vel, particle_vel_mask, particle_acc, projection_fn, model_fn).compile()
        i = 0
        
        for i in tqdm(range(100000)):
            lossvals, model_states, dynamic_params = update(model_states, dynamic_params, static_params, index_list, index_mask, B_val_list, 
                                                            B_val_mask, particle_vel, particle_vel_mask, particle_acc)
            
            self.report(i, report_fn, dynamic_params, all_params, index_list, index_mask, B_val_list, 
                        B_val_mask, particle_vel, particle_vel_mask, particle_acc, 
                        self.c.optimization_init_kwargs["save_step"], model_fn)
            
            self.save_model(i, dynamic_params, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)
            i +=1
        return
    
    def save_model(self, i, dynamic_params, all_params, save_step, model_fns):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["projection"]['coefficients'] = dynamic_params
            model = FlowFitmodel(all_params["projection"]['coefficients'], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
        return
    
    def report(self, i, report_fn, dynamic_params, all_params, index_list, index_mask, B_val_list, 
               B_val_mask, particle_vel, particle_vel_mask, 
               particle_acc, save_step, model_fns):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["projection"]['coefficients'] = dynamic_params

            t_loss, u_loss, v_loss, w_loss, hp_loss = report_fn(dynamic_params[0,:,:,:,:], all_params, index_list[0,:,:], B_val_list[0,:,:])
            u_pred, v_pred, w_pred = model_fns(dynamic_params[0,:,:,:,:], index_list[0,:,:], B_val_list[0,:,:])
            u_error = jnp.sqrt(jnp.mean((u_pred - particle_vel[0,:,0])**2)/jnp.mean(particle_vel[0,:,0]**2))
            v_error = jnp.sqrt(jnp.mean((v_pred - particle_vel[0,:,1])**2)/jnp.mean(particle_vel[0,:,1]**2))
            w_error = jnp.sqrt(jnp.mean((w_pred - particle_vel[0,:,2])**2)/jnp.mean(particle_vel[0,:,2]**2))

            print(f"step_num : {i:<{12}} t_loss : {t_loss:<{12}.{5}} u_loss : {u_loss:<{12}.{5}} v_loss : {v_loss:<{12}.{5}} w_loss : {w_loss:<{12}.{5}} \
                  hp_loss : {hp_loss:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}} ")
            with open(self.c.report_out_dir + "reports.txt", "a") as f:
                f.write(f"{i:<{12}} {t_loss:<{12}.{5}} {u_loss:<{12}.{5}} {u_error:<{12}.{5}} {v_loss:<{12}.{5}} {w_loss:<{12}.{5}} {hp_loss:<{12}.{5}} \
                         {v_error:<{12}.{5}} {w_error:<{12}.{5}}\n")
            f.close()
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
