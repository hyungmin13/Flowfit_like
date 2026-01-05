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
        weights = all_params["problem"]["loss_weights"]

        u_pred, v_pred, w_pred = model_fns(dynamic_params, *indexes, dx, dy, dz, x_u, x_v, x_z)                                                                           

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]

        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, particles, particle_vel, boundaries, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        def second_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            def u_tt(batch):
                return jax.jvp(u_t,(batch,), (cotangent, ))[1]
            out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
            return out_x, out_xx
        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_x, out_xx = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_y, out_yy = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        out_z, out_zz = second_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

        p_out = model_fns(all_params, particles)                                                                        

        u = all_params["data"]['u_ref']*out[:,0:1]
        v = all_params["data"]['v_ref']*out[:,1:2]
        w = all_params["data"]['w_ref']*out[:,2:3]
        p = all_params["data"]['p_ref']*out[:,3:4]

        ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["domain"]["domain_range"]["t"][1]
        vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["domain"]["domain_range"]["t"][1]
        wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["domain"]["domain_range"]["t"][1]

        ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["domain"]["domain_range"]["x"][1]
        vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["domain"]["domain_range"]["x"][1]
        wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["domain"]["domain_range"]["x"][1]
        px = all_params["data"]['p_ref']*out_x[:,3:4]/all_params["domain"]["domain_range"]["x"][1]


        uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["domain"]["domain_range"]["y"][1]
        vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["domain"]["domain_range"]["y"][1]
        wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["domain"]["domain_range"]["y"][1]
        py = all_params["data"]['p_ref']*out_y[:,3:4]/all_params["domain"]["domain_range"]["y"][1]

        uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["domain"]["domain_range"]["z"][1]
        vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["domain"]["domain_range"]["z"][1]
        wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["domain"]["domain_range"]["z"][1]
        pz = all_params["data"]['p_ref']*out_z[:,3:4]/all_params["domain"]["domain_range"]["z"][1]

        uxx = all_params["data"]['u_ref']*out_xx[:,0:1]/all_params["domain"]["domain_range"]["x"][1]**2
        vxx = all_params["data"]['v_ref']*out_xx[:,1:2]/all_params["domain"]["domain_range"]["x"][1]**2
        wxx = all_params["data"]['w_ref']*out_xx[:,2:3]/all_params["domain"]["domain_range"]["x"][1]**2

        uyy = all_params["data"]['u_ref']*out_yy[:,0:1]/all_params["domain"]["domain_range"]["y"][1]**2
        vyy = all_params["data"]['v_ref']*out_yy[:,1:2]/all_params["domain"]["domain_range"]["y"][1]**2
        wyy = all_params["data"]['w_ref']*out_yy[:,2:3]/all_params["domain"]["domain_range"]["y"][1]**2

        uzz = all_params["data"]['u_ref']*out_zz[:,0:1]/all_params["domain"]["domain_range"]["z"][1]**2
        vzz = all_params["data"]['v_ref']*out_zz[:,1:2]/all_params["domain"]["domain_range"]["z"][1]**2
        wzz = all_params["data"]['w_ref']*out_zz[:,2:3]/all_params["domain"]["domain_range"]["z"][1]**2

        loss_u = all_params["data"]['u_ref']*p_out[:,0:1] - particle_vel[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = all_params["data"]['v_ref']*p_out[:,1:2] - particle_vel[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = all_params["data"]['w_ref']*p_out[:,2:3] - particle_vel[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        loss_con = ux + vy + wz
        loss_con = jnp.mean(loss_con**2)
        loss_NS1 = ut + u*ux + v*uy + w*uz + px - all_params["data"]["viscosity"]*(uxx+uyy+uzz)-2.22*10**(-1)/(3*0.43685**2)*u
        loss_NS1 = jnp.mean(loss_NS1**2)
        loss_NS2 = vt + u*vx + v*vy + w*vz + py - all_params["data"]["viscosity"]*(vxx+vyy+vzz)-2.22*10**(-1)/(3*0.43685**2)*v
        loss_NS2 = jnp.mean(loss_NS2**2)
        loss_NS3 = wt + u*wx + v*wy + w*wz + pz - all_params["data"]["viscosity"]*(wxx+wyy+wzz)-2.22*10**(-1)/(3*0.43685**2)*w
        loss_NS3 = jnp.mean(loss_NS3**2)

        total_loss = weights[0]*loss_u + weights[1]*loss_v + weights[2]*loss_w + \
                    weights[3]*loss_con + weights[4]*loss_NS1 + weights[5]*loss_NS2 + \
                    weights[6]*loss_NS3 
        return total_loss, loss_u, loss_v, loss_w, loss_con, loss_NS1, loss_NS2, loss_NS3

