# [depends] init_cond1.gsd
# [makes] gsd
import hoomd
import numpy as np
from lib import hoomd_utils
import os
from lib.hoomd_utils import BaseControlPolicy
import cupy as cp
from lib.utils import get_array_module, array_module, asnumpy
from scipy.integrate import simpson
from sp_chase_ex2_gsd import SetPoint

xp = array_module('cupy')


def main():
    tracer_seed = 18
    n_traced = 1000
    meas_order = 10
    bin_size = 0.00115
    control_area = 8
    wr001_max = 3
    max_speed = (1 / np.tanh(wr001_max) - 1 / wr001_max)
    oscillations_start = 15
    scaled_speed = 0.9 * max_speed
    period = control_area / 2 / scaled_speed
    n_periods = 1.25

    oscillations_stop = period * 2 * np.pi * n_periods + oscillations_start
    sim_time = oscillations_stop + 5

    ld2 = 10

    w = 10  # box width
    h = 10  # box height
    meas_z_pts = np.linspace(-h / 2, h / 2, 301)
    ctrl_dt = 0.01

    dt = 0.001

    dirname = os.path.dirname(__file__)

    init_filename = os.path.join(
        dirname,
        "build/init_cond1.gsd")
    dest_filename = os.path.join(
        dirname,
        "build/heur_sol_ex1_gsd.gsd")
    seed = 10

    ell = 1  # ell: run length,
    D_r = 1  # tau^{-1}: tau_r
    zeta_t = 1  # mu•tau^{-1}: mu = mass of particle

    delta = ell / np.sqrt(ld2)  # delta: microscopic length
    a = np.sqrt(3 / 4) * delta  # ell: particle radius

    use_t_diff = True
    np.random.seed(seed)

    eta = zeta_t / (6 * np.pi * a)  # mu•tau^{-1}•sigma^{-1}
    zeta_r = 8 * np.pi * eta * a ** 3  # mu•sigma^{2}•tau^{-1}
    kbt = D_r * zeta_r  # mu•sigma^{2}•tau^{-2}
    D_t = 4 / 3 * a ** 2 * D_r  # sigma^{2}•tau^{-1}
    tau = 1 / D_r  # tau
    u0 = np.sqrt(ld2 * D_t / tau)  # sigma•tau^{-1}

    fp = zeta_t * u0  # mu•sigma•tau^{-2}
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=seed)
    sim.timestep = 0
    sim.create_state_from_gsd(filename=init_filename)
    integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)
    if use_t_diff:
        brownian = hoomd.md.methods.Brownian(kT=kbt, filter=hoomd.filter.All())
    else:
        brownian = hoomd.md.methods.Brownian(kT=0, filter=hoomd.filter.All())
    brownian.gamma.default = zeta_t
    brownian.gamma_r.default = [zeta_r, zeta_r, zeta_r]
    integrator.methods.append(brownian)
    sim.operations.integrator = integrator
    snapshot = sim.state.get_snapshot()
    box_height = snapshot.configuration.box[1]
    pcl_pos = np.clip(snapshot.particles.position[:, 1], -h / 2, h / 2)
    pcl_quarts = cp.array(snapshot.particles.orientation)
    pcl_thetas = hoomd_utils.gpu_quaternion_to_euler_angle_vectorized2(pcl_quarts[:, 0],
                                                                       pcl_quarts[:, 1],
                                                                       pcl_quarts[:, 2],
                                                                       pcl_quarts[:, 3],
                                                                       xp)[-1] * np.pi / 180
    bd_zpts = np.linspace(-box_height / 2, box_height / 2, 1001)
    full_state0 = hoomd_utils.state_eval(pcl_pos, pcl_thetas.get(), bin_size, bd_zpts, meas_z_pts, meas_order)
    np.random.seed(tracer_seed)
    traced = hoomd.filter.Tags(list(np.random.randint(0, snapshot.particles.N - 1, n_traced)))

    active1 = hoomd.md.force.Active(filter=hoomd.filter.All())
    active1.active_force['A'] = (fp, 0, 0)
    integrator.forces.append(active1)
    walls = [hoomd.wall.Plane((0, -h / 2 - 2 ** (1 / 6) * a, 0), (0, 1, 0)),
             hoomd.wall.Plane((0, h / 2 + 2 ** (1 / 6) * a, 0), (0, -1, 0))]
    lj = hoomd.md.external.wall.ForceShiftedLJ(walls=walls)
    lj.params['A'] = {"sigma": a, "epsilon": 5.0, "r_cut": a * (2 ** (1 / 6))}

    plate_gap = h
    n_ctrl_pts = 31
    sim_dt = dt
    policy_args = {'full_state0': full_state0,
                   'plate_gap': plate_gap,
                   'bin_size': bin_size,
                   'bd_zpts': bd_zpts,
                   'env_zpts': meas_z_pts,
                   'meas_order': meas_order,
                   'n_ctrl_pts': n_ctrl_pts,
                   'zeta_r': zeta_r,
                   'control_area': control_area,
                   'wr001_max': wr001_max,
                   'oscillations_start': oscillations_start,
                   'oscillations_stop': oscillations_stop,
                   'period': period,
                   'ctrl_dt': ctrl_dt,
                   'sample_dt': ctrl_dt,
                   'sim_dt': sim_dt
                   }

    hoomd_controller = PaperSandboxEx1OptPolicy(**policy_args)
    integrator.forces.append(lj)
    integrator.forces.append(hoomd_controller)
    logger = hoomd.logging.Logger()
    logger['dt'] = (lambda: dt, 'scalar')
    logger['u0'] = (lambda: u0, 'scalar')
    logger['ld2'] = (lambda: ld2, 'scalar')
    logger['zeta_r'] = (lambda: zeta_r, 'scalar')
    logger['env_zpts'] = (hoomd_controller, 'env_zpts', 'sequence')
    logger['state_cost'] = (hoomd_controller, 'state_cost', 'scalar')
    for key in full_state0.keys():
        logger[key] = (hoomd_controller, key, 'sequence')
    logger['x_sp'] = (hoomd_controller, 'x_sp', 'scalar')
    gsd_writer = hoomd.write.GSD(filename=dest_filename,
                                 trigger=hoomd.trigger.Periodic(int(np.round(1 / (dt / ctrl_dt)))),
                                 mode='wb', filter=traced
                                 )
    gsd_writer.log = logger
    sim.operations.writers.append(gsd_writer)

    sim.run(sim_time * (int(np.round(1 / dt))))


class PaperSandboxEx1OptPolicy(BaseControlPolicy):
    def __init__(self,
                 full_state0,
                 plate_gap,
                 bin_size,
                 bd_zpts,
                 env_zpts,
                 meas_order,
                 n_ctrl_pts,
                 zeta_r,
                 control_area,
                 wr001_max,
                 oscillations_start,
                 oscillations_stop,
                 period,
                 ctrl_dt,
                 sample_dt,
                 sim_dt):
        super().__init__(full_state0,
                         plate_gap,
                         bin_size,
                         bd_zpts,
                         env_zpts,
                         meas_order)
        self.zeta_r = zeta_r
        self.control_area = control_area
        self.wr001_max = wr001_max
        self.oscillations_start = oscillations_start
        self.oscillations_stop = oscillations_stop
        self.period = period
        self.ctrl_dt = ctrl_dt
        self.sample_dt = sample_dt
        self.sim_dt = sim_dt
        self._x_sp = 0
        self.state_cost_multiplier = 6400
        self._state_cost = 0
        self.sp_obj = SetPoint()
        self.sp_fcn = self.sp_obj.sp_fcn

    def w_ctrl_fcn(self, z, sp):
        return -self.wr001_max * np.tanh(100 * (z - sp))

    def set_forces(self, timestep):
        if timestep % (self.sample_dt / self.sim_dt) == 0:
            pcl_pos, pcl_thetas = self.measure(evalulate_meas_vars=True)
        else:
            pcl_pos, pcl_thetas = self.measure(evalulate_meas_vars=False)

        if timestep % (self.ctrl_dt / self.sim_dt) == 0:
            time = timestep * self.sim_dt
            sp = self.sp_fcn(time)
            self._state_cost = self.get_state_cost(sp)
            self._x_sp = sp

        with self.gpu_local_force_arrays as arrays:
            arrays.torque[:, -1] = xp.asarray(
                self.w_ctrl_fcn(asnumpy(pcl_pos), self._x_sp)) * xp.cos(
                pcl_thetas) * self.zeta_r

    def get_state_cost(self, x_sp):
        z_sp = x_sp
        integrand = self.full_state['pr000'] * (self.env_zpts - z_sp) ** 2

        cost = self.state_cost_multiplier * simpson((integrand, self.env_zpts))
        return cost

    @property
    def x_sp(self):
        return self._x_sp

    @property
    def state_cost(self):
        return self._state_cost


if __name__ == "__main__":
    main()
