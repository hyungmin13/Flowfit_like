#%%
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from domain import *
from projection import *
from velocity_pred import *
from B_spline import *
import argparse
from Tecplot_mesh import tecplot_Mesh

#%%
class FlowFitmodel(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class FlowFitbase:
    def __init__(self, c):
        c.get_outdirs()
        self.c = c

class FlowFit3_eval(FlowFitbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "projection":{}, "prediction":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        
        all_params["projection"] = self.c.projection.init_params(**self.c.projection_init_kwargs)
        all_params["prediction"] = self.c.prediction.init_params(**self.c.prediction_init_kwargs)
        global_key = random.PRNGKey(42)
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        #all_params = self.c.problem.constraints(all_params)
        #valid_data = self.c.problem.exact_solution(all_params)
        model_fn = self.c.prediction.velocity_pred
        return all_params, model_fn, train_data
    

def Derivatives(dynamic_param, all_params, index, dx, dy, dz, xu, xv, xw, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']

    #all_params["projection"]["coefficients"] = dynamic_params
    #out, out_x = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    #out, out_y = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    #out, out_z = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)    
    u_pred, v_pred, w_pred = model_fns(dynamic_param, index, dx, dy, dz, xu, xv, xw)
    #uvwp = np.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    #uvwp = np.concatenate([u_pred.reshape(-1,1), v_pred.reshape(-1,1), w_pred.reshape(-1,1)],1)

    #deriv_mat = np.concatenate([np.expand_dims(uxs,2),np.expand_dims(uys,2),np.expand_dims(uzs,2)],2)
    #vor_mag = np.sqrt((deriv_mat[:,1,2]-deriv_mat[:,2,1])**2+
    #                  (deriv_mat[:,2,0]-deriv_mat[:,0,2])**2+
    #                  (deriv_mat[:,0,1]-deriv_mat[:,1,0])**2)
    #Q = 0.5 * sum(-np.abs(0.5 * (deriv_mat[:, i, j] + deriv_mat[:, j, i]))**2 +
    #              np.abs(0.5 * (deriv_mat[:, i, j] - deriv_mat[:, j, i]))**2 
    #              for i in range(3) for j in range(3))
    return u_pred, v_pred, w_pred #, vor_mag, Q, deriv_mat

def Tecplotfile_gen(path, name, all_params, train_data, domain_range, output_shape, order, timestep, is_ground, is_mean, model_fn):
    
    # Load the parameters
    pos_ref = all_params["domain"]["in_max"].flatten()
    dynamic_params = all_params["projection"].pop("coefficients")
    xc = np.linspace(all_params['domain']['domain_range']['x'][0], all_params['domain']['domain_range']['x'][1], all_params['domain']['grid_size'][0])
    yc = np.linspace(all_params['domain']['domain_range']['y'][0], all_params['domain']['domain_range']['y'][1], all_params['domain']['grid_size'][1])
    zc = np.linspace(all_params['domain']['domain_range']['z'][0], all_params['domain']['domain_range']['z'][1], all_params['domain']['grid_size'][2])
    dx = xc[1]-xc[0]
    dy = yc[1]-yc[0]
    dz = zc[1]-zc[0]
    # Create the evaluation grid
    gridbase = [np.linspace(domain_range[key][0], domain_range[key][1], output_shape[i]) for i, key in enumerate(['t', 'x', 'y', 'z'])]

    if order[0] == 0:
        if order[1] == 1:
            z_e, y_e, x_e = np.meshgrid(gridbase[-1], gridbase[-2], gridbase[-3], indexing='ij')
        else:
            y_e, z_e, x_e = np.meshgrid(gridbase[-2], gridbase[-1], gridbase[-3], indexing='ij')
    elif order[0] == 1:
        if order[1] == 0:
            z_e, x_e, y_e = np.meshgrid(gridbase[-1], gridbase[-3], gridbase[-2], indexing='ij')
        else:
            y_e, x_e, z_e = np.meshgrid(gridbase[-2], gridbase[-3], gridbase[-1], indexing='ij')
    elif order[0] == 2:
        if order[1] == 0:
            x_e, z_e, y_e = np.meshgrid(gridbase[-3], gridbase[-1], gridbase[-2], indexing='ij')
        else:
            x_e, y_e, z_e = np.meshgrid(gridbase[-3], gridbase[-2], gridbase[-1], indexing='ij') 
    #t_e = np.zeros(output_shape[1:]) + gridbase[0][timestep]

    eval_grid_e = np.concatenate([x_e.reshape(-1,1), y_e.reshape(-1,1), z_e.reshape(-1,1)], axis=1)
    # Load Ground truth data if is_ground is True

    print(np.max(eval_grid_e[:,0]), np.min(eval_grid_e[:,0]))
    print(np.max(eval_grid_e[:,1]), np.min(eval_grid_e[:,1]))
    print(np.max(eval_grid_e[:,2]), np.min(eval_grid_e[:,2]))
    # Evaluate the derivatives
    #uvwp, vor_mag, Q, deriv_mat = zip(*[Derivatives(dynamic_params, all_params, eval_grid[i:i+10000], model_fn)
    #                                    for i in range(0, eval_grid.shape[0], 10000)])
    indexes = VelocityPrediction3D.find_indexes(eval_grid_e, xc, yc, zc, dx, dy, dz)
    xu, xv, xw = VelocityPrediction3D.data_reshape(eval_grid_e, *indexes, dx, dy, dz, xc, yc, zc)
    indexes = np.array(indexes)
    u_pred, v_pred, w_pred = zip(*[Derivatives(dynamic_params[i,:,:,:,:], all_params, indexes, dx, dy, dz, xu, xv, xw, model_fn)
                            for i in range(len(timestep))])
    
    #vals, counts = np.unique(train_data['pos'][:,0], return_counts=True)
    #indexes = VelocityPrediction3D.find_indexes(train_data['pos'][np.sum(counts[:0]):np.sum(counts[:0+1]),1:],xc, yc, zc, dx, dy, dz)
    #xu, xv, xw = VelocityPrediction3D.data_reshape(train_data['pos'][np.sum(counts[:0]):np.sum(counts[:0+1]),1:], *indexes, dx, dy, dz, xc, yc, zc)
    #indexes = np.array(indexes)
    #u_pred, v_pred, w_pred = Derivatives(dynamic_params[0,:,:,:,:], all_params, indexes, dx, dy, dz, xu, xv, xw, model_fn)
    u_pred = np.array(u_pred)
    v_pred = np.array(v_pred)
    w_pred = np.array(w_pred)
    #print(u_pred)
    #print(train_data['vel'][np.sum(counts[:0]):np.sum(counts[:0+1]),0])
    uvwp = np.concatenate([u_pred.reshape(len(timestep),-1,1), v_pred.reshape(len(timestep),-1,1), w_pred.reshape(len(timestep),-1,1)],2)
    
    
    for j in range(timestep[0], timestep[-1]):
        if is_ground:
            ground_data = np.load(path + 'ground/ts_' + str(j).zfill(2) + '.npy')
        if is_mean:
            mean_data = np.load(path + 'mean')


        if is_ground:
            grounds = [ground_data[:,i+4].reshape(output_shape[1:]) for i in range(3)]
            errors = [np.sqrt(np.square(uvwp[j,:,i].reshape(output_shape[1:]) - grounds[i])) for i in range(3)]
            #if ground_data.shape[1] > 7:
            #    p_ground = ground_data[:,7].reshape(output_shape[1:])
            #    p_error = np.sqrt(np.square(uvwp[:,3].reshape(output_shape[1:]) - p_ground))
            #if ground_data.shape[1] > 8:
            #    temp_ground = ground_data[:,8].reshape(output_shape[1:])
            #    temp_error = np.sqrt(np.square(uvwp[:,4].reshape(output_shape[1:]) - temp_ground))
        if is_mean:
            means = [mean_data['vel'][:,i].reshape(output_shape[1:]) for i in range(3)]
            flucs = [uvwp[:,i].reshape(output_shape[1:]) - means[i] for i in range(3)]

        # Tecplot file generation

        filename = path + 'Tecplotfile/' + name + '/ts_' + str(j) + '.dat'
        if os.path.isdir(path + 'Tecplotfile/' + name):
            pass
        else:
            os.mkdir(path + 'Tecplotfile/' + name)
        X, Y, Z = output_shape[1:]
        vars = [('u_pred[m/s]',np.float32(uvwp[j,:,0].reshape(-1))), ('v_pred[m/s]',np.float32(uvwp[j,:,1].reshape(-1))), 
                ('w_pred[m/s]',np.float32(uvwp[j,:,2].reshape(-1))),]
        if is_ground:
            vars += [('u_error[m/s]', np.float32(errors[0].reshape(-1))),
                    ('v_error[m/s]', np.float32(errors[1].reshape(-1))),
                    ('w_error[m/s]', np.float32(errors[2].reshape(-1)))]
            #if ground_data.shape[1] > 7:
            #    vars += [('p_error[Pa]', np.float32(p_error.reshape(-1)))]
            #if ground_data.shape[1] > 8:
            #    vars += [('temp_error[K]', np.float32(temp_error.reshape(-1)))]
        if is_mean:
            vars += [('u_fluc[m/s]', np.float32(flucs[0].reshape(-1))),
                    ('v_fluc[m/s]', np.float32(flucs[1].reshape(-1))),
                    ('w_fluc[m/s]', np.float32(flucs[2].reshape(-1)))]
        pad = 27
        tecplot_Mesh(filename, X, Y, Z, x_e.reshape(-1), y_e.reshape(-1), z_e.reshape(-1), vars, pad)

        if os.path.isdir(path + 'npyresult/' + name):
            pass
        else:
            print('check')
            os.mkdir(path + 'npyresult/' + name)
        np.save(path + 'npyresult/' + name + f'/ts_{j:02d}' + '.npy', np.concatenate([eval_grid_e, uvwp[j,:,:]], axis=1))
    
#%%
if __name__ == "__main__":
    from domain import *
    from trackdata import *
    from projection import *
    from constants import *
    from equation import *
    from txt_reader import *
    import os
    parser = argparse.ArgumentParser(description='FlowFit')
    parser.add_argument('-f', '--foldername', type=str, help='foldername', default='HIT')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='test_txt')
    args = parser.parse_args()

    # Get evaluation configuration
    cur_dir = os.getcwd()
    config_txt = cur_dir + '/' + args.config + '.txt'
    data = parse_tree_structured_txt(config_txt)

    # Get model constants
    with open(os.path.dirname(cur_dir)+ '/' + data['path'] + args.foldername +'/summary/constants.pickle','rb') as f:
        constants = pickle.load(f)
    values = list(constants.values())
    
    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                projection_init_kwargs = values[3],
                prediction_init_kwargs = values[4],
                optimization_init_kwargs = values[5],
                equation_init_kwargs = values[6],)
    run = FlowFit3_eval(c)

    # Get model parameters
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    with open(checkpoint_list[-1],"rb") as f:
        model_params = pickle.load(f)
    all_params, model_fn, train_data = run.test()
    model = FlowFitmodel(all_params["projection"]["coefficients"], model_fn)
    all_params["projection"]["coefficients"] = from_state_dict(model, model_params).params
    
    domain_range = data['tecplot_init_kwargs']['domain_range']
    output_shape = data['tecplot_init_kwargs']['out_shape']
    order = data['tecplot_init_kwargs']['order']
    timesteps = data['tecplot_init_kwargs']['timestep']
    is_ground = data['tecplot_init_kwargs']['is_ground']
    path = data['tecplot_init_kwargs']['path']
    is_mean = data['tecplot_init_kwargs']['is_mean']
    path = os.path.dirname(cur_dir) + '/' + path
    pos_ref = all_params["domain"]["in_max"].flatten()
    #for timestep in timesteps:
    Tecplotfile_gen(path, args.foldername, all_params, train_data, domain_range, output_shape, order, timesteps, is_ground, is_mean, model_fn)
