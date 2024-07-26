# [depends] init_cond1.gsd
# [makes] gsd
import hoomd
import gsd.hoomd
import numpy as np
from lib import utils
from lib.hoomd_utils import BaseControlPolicy
from lib import hoomd_utils
import casadi as cs
import mpctools as mpc
from time import time
from scipy.integrate import simpson
from lib.eqn_generator import pars_to_state_space_v2
from scipy.interpolate import interp1d
from lib.utils import get_array_module, array_module, asnumpy
import os

xp = array_module('cupy')


class SetPoint:
    def __init__(self, plate_gap):
        self.center = 0
        self.center1 = -2.5
        self.center2 = 2.5
        self.trigger_times = [0, 10, 15, 25, 30, 60, 65, 95, 100]
        self.sim_time = 105
        self.w001_max = 3
        self.plate_gap = plate_gap

        self.int1 = interp1d([self.trigger_times[1],
                              self.trigger_times[2]],
                             [self.center, self.center1])
        self.int2 = interp1d([self.trigger_times[1], self.trigger_times[2]], [self.center, self.center2])

        self.int3 = interp1d([self.trigger_times[3], self.trigger_times[4]], [0, 1], fill_value=(0, 1),
                             bounds_error=False)
        self.int4 = interp1d([self.trigger_times[5], self.trigger_times[6]], [1, -1],
                             fill_value=(1, -1),
                             bounds_error=False)

        self.int5 = interp1d([self.trigger_times[7], self.trigger_times[8]], [-1, 0],
                             fill_value=(-1, 0), bounds_error=False)

    def sp_fcn(self, t):
        if self.trigger_times[0] <= t <= self.trigger_times[1]:
            sp1 = self.center
            sp2 = self.center
            sp3 = 0
        elif self.trigger_times[1] <= t <= self.trigger_times[2]:
            sp1 = self.int1(t)
            sp2 = self.int2(t)
            sp3 = 0
        elif self.trigger_times[3] <= t <= self.trigger_times[5]:
            sp1 = self.center1
            sp2 = self.center2
            sp3 = self.int3(t)
        elif self.trigger_times[5] <= t <= self.trigger_times[7]:
            sp1 = self.center1
            sp2 = self.center2
            sp3 = self.int4(t)
        elif self.trigger_times[7] <= t <= self.trigger_times[8]:
            sp1 = self.center1
            sp2 = self.center2
            sp3 = self.int5(t)
        else:
            sp1 = self.center1
            sp2 = self.center2
            sp3 = 0
        return sp1, sp2, sp3

    def desired_mx_fcn(self, z):
        desired_pts = 0.4 * np.sin(z * 2 * np.pi / self.plate_gap)
        return desired_pts


def main():
    tracer_seed = 20
    n_traced = 2000

    dirname = os.path.dirname(__file__)
    init_filename = os.path.join(
        dirname,
        "build/init_cond1.gsd")
    dest_filename = os.path.join(
        dirname,
        f"build/{os.path.basename(__file__)}.gsd")
    seed = 10

    hoomd_controller = ControlPolicy(init_filename)
    sim_dt = hoomd_controller.sim_dt
    sample_dt = hoomd_controller.sample_dt
    kbt = hoomd_controller.kbt
    zeta_t = hoomd_controller.zeta_t
    zeta_r = hoomd_controller.zeta_r
    a = hoomd_controller.a
    fp = hoomd_controller.fp
    plate_gap = hoomd_controller.plate_gap
    ld2 = hoomd_controller.ld2
    full_state0 = hoomd_controller.full_state0
    hoomd_controller.initialize()

    use_t_diff = True
    np.random.seed(seed)

    if hoomd.version.gpu_enabled:
        device = hoomd.device.GPU()
    else:
        device = hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=seed)
    sim.timestep = 0
    sim.create_state_from_gsd(filename=init_filename)
    integrator = hoomd.md.Integrator(dt=sim_dt, integrate_rotational_dof=True)
    if use_t_diff:
        brownian = hoomd.md.methods.Brownian(kT=kbt, filter=hoomd.filter.All())
    else:
        brownian = hoomd.md.methods.Brownian(kT=0, filter=hoomd.filter.All())
    brownian.gamma.default = zeta_t
    brownian.gamma_r.default = [zeta_r, zeta_r, zeta_r]
    integrator.methods.append(brownian)
    sim.operations.integrator = integrator
    snapshot = sim.state.get_snapshot()

    np.random.seed(tracer_seed)
    traced = hoomd.filter.Tags(
        list(np.random.choice(np.arange(0, snapshot.particles.N - 1),
                              size=n_traced, replace=False)
             ))
    active1 = hoomd.md.force.Active(filter=hoomd.filter.All())
    active1.active_force['A'] = (fp, 0, 0)
    integrator.forces.append(active1)
    walls = [hoomd.wall.Plane((0, -plate_gap / 2 - 2 ** (1 / 6) * a, 0), (0, 1, 0)),
             hoomd.wall.Plane((0, plate_gap / 2 + 2 ** (1 / 6) * a, 0), (0, -1, 0))]
    lj = hoomd.md.external.wall.ForceShiftedLJ(walls=walls)
    lj.params['A'] = {"sigma": a, "epsilon": 5.0, "r_cut": a * (2 ** (1 / 6))}
    sp_obj = SetPoint(plate_gap=plate_gap)

    integrator.forces.append(lj)
    integrator.forces.append(hoomd_controller)
    logger = hoomd.logging.Logger()
    logger['dt'] = (lambda: sim_dt, 'scalar')
    # logger['u0'] = (lambda: u0, 'scalar')
    logger['ld2'] = (lambda: ld2, 'scalar')
    logger['zeta_r'] = (lambda: zeta_r, 'scalar')
    logger['wr001_coeffs'] = (hoomd_controller, 'wr001_coeffs', 'sequence')
    logger['wi001_coeffs'] = (hoomd_controller, 'wi001_coeffs', 'sequence')
    logger['env_zpts'] = (hoomd_controller, '_env_zpts', 'sequence')
    logger['solve_time'] = (hoomd_controller, 'solve_time', 'scalar')
    logger['state_cost'] = (hoomd_controller, 'state_cost', 'scalar')
    logger['ctrl_cost'] = (hoomd_controller, 'ctrl_cost', 'scalar')
    for key in full_state0.keys():
        logger[key] = (hoomd_controller, key, 'sequence')
    for horizon in range(hoomd_controller.n_horizon + 1):
        logger[f'x{horizon}'] = (hoomd_controller, f'x{horizon}', 'sequence')
    for horizon in range(hoomd_controller.n_horizon):
        logger[f'u{horizon}'] = (hoomd_controller, f'u{horizon}', 'sequence')
    gsd_writer = hoomd.write.GSD(filename=dest_filename,
                                 trigger=hoomd.trigger.Periodic(int(np.round(1 / (sim_dt / sample_dt)))),
                                 mode='wb', filter=traced)
    gsd_writer.log = logger
    sim.operations.writers.append(gsd_writer)
    sim_time = sp_obj.sim_time
    sim.run(sim_time * (int(np.round(1 / sim_dt))))


class ControlPolicy(BaseControlPolicy):
    def __init__(self,
                 init_filename):
        frame = gsd.hoomd.open(init_filename, 'rb')[0]
        box_height = frame.configuration.box[1]
        self.plate_gap = 10
        self.ld2 = 10

        self.sample_dt = 0.1

        self.sim_dt = 0.001

        self.meas_order = 10
        self.bin_size = 0.00115
        pcl_pos = np.clip(frame.particles.position[:, 1], -self.plate_gap / 2, self.plate_gap / 2)
        pcl_quarts = xp.asarray(frame.particles.orientation)
        pcl_thetas = hoomd_utils.quaternion_to_euler_angle_vectorized(pcl_quarts[:, 0],
                                                                      pcl_quarts[:, 1],
                                                                      pcl_quarts[:, 2],
                                                                      pcl_quarts[:, 3])[-1] * np.pi / 180
        self.bd_zpts = np.linspace(-box_height / 2, box_height / 2, 1001)
        self.meas_zpts = np.linspace(-self.plate_gap / 2, self.plate_gap / 2, 301)
        self.full_state0 = hoomd_utils.state_eval(pcl_pos, asnumpy(pcl_thetas), self.bin_size,
                                                  self.bd_zpts, self.meas_zpts, self.meas_order)

        self.ell = 1  # ell: run length,
        self.D_r = 1  # tau^{-1}: tau_r
        self.zeta_t = 1  # mu•tau^{-1}: mu = mass of particle

        self.delta = self.ell / np.sqrt(self.ld2)  # delta: microscopic length
        self.a = np.sqrt(3 / 4) * self.delta  # ell: particle radius

        self.eta = self.zeta_t / (6 * np.pi * self.a)  # mu•tau^{-1}•sigma^{-1}
        self.zeta_r = 8 * np.pi * self.eta * self.a ** 3  # mu•sigma^{2}•tau^{-1}
        self.kbt = self.D_r * self.zeta_r  # mu•sigma^{2}•tau^{-2}
        self.D_t = 4 / 3 * self.a ** 2 * self.D_r  # sigma^{2}•tau^{-1}
        self.tau = 1 / self.D_r  # tau
        self.u0 = np.sqrt(self.ld2 * self.D_t / self.tau)  # sigma•tau^{-1}
        self.fp = self.zeta_t * self.u0  # mu•sigma•tau^{-2}

        super().__init__(self.full_state0,
                         self.plate_gap,
                         self.bin_size,
                         self.bd_zpts,
                         self.meas_zpts,
                         self.meas_order)

        self.n_state_pts = 40
        self.n_wr001_pts = 20
        self.n_wi001_pts = 20
        self.ctrl_order = 5
        self.n_time_pts = 1

        self.n_horizon = 14
        self.ctrl_dt = 0.5
        self.ctrl_key_list = ['wr001', 'wi001']
        self.con_zpts = np.linspace(-self.plate_gap / 2, self.plate_gap / 2, 80)

        sp_obj = SetPoint(self.plate_gap)
        self.sp_fcn = sp_obj.sp_fcn
        self.mx_fcn = sp_obj.desired_mx_fcn
        self.w001_max = sp_obj.w001_max

        self.state_cost_multiplier = 6400
        self.diff_cost_multiplier = self.state_cost_multiplier
        self.mx_state_cost_multiplier = self.state_cost_multiplier * 1e3
        self.diff_cost_multiplier = self.state_cost_multiplier * 10
        self.wr001_dz2_ctrl_cost_multiplier = 1e-4
        self.wi001_dz2_ctrl_cost_multiplier = self.wr001_dz2_ctrl_cost_multiplier
        self.dt_ctrl_cost_multiplier = 0.0001

        self.mini_w_ctrl = 0 * np.ones(self.n_wr001_pts)
        self._solve_time = 0
        self._state_cost = 0
        self._ctrl_cost = 0

        state_pts_all, self.state_dz, self.state_dz2, self.state_ig = utils.colloc(self.n_state_pts,
                                                                                   left=True,
                                                                                   right=True,
                                                                                   plate_gap=self.plate_gap,
                                                                                   shifted=False)
        self.state_pts_all = state_pts_all.flatten()
        self.state_pts = self.state_pts_all[1:-1]
        self.wr001_pts, self.wr001_dz, self.wr001_dz2, self.wr001_ig = utils.colloc(self.n_wr001_pts,
                                                                                    plate_gap=self.plate_gap,
                                                                                    shifted=False)
        self.wr001_pts = self.wr001_pts.flatten()
        self.wi001_pts, self.wi001_dz, self.wi001_dz2, self.wi001_ig = utils.colloc(self.n_wi001_pts,
                                                                                    plate_gap=self.plate_gap,
                                                                                    shifted=False)
        self.wi001_pts = self.wi001_pts.flatten()

        self.wr001_to_state_mat = utils.polinterp(self.wr001_pts, self.state_pts)
        self.wr001_to_con_mat = utils.polinterp(self.wr001_pts, self.con_zpts)

        self.wi001_to_state_mat = utils.polinterp(self.wi001_pts, self.state_pts)
        self.wi001_to_con_mat = utils.polinterp(self.wi001_pts, self.con_zpts)

        self.w_n = 2
        self.d_n = 1

        n_vars = (2 * self.ctrl_order - 1)
        self.n_x = n_vars * self.n_state_pts
        self.n_u = self.n_wr001_pts + self.n_wi001_pts

        state_pars_dict = dict(n_state_pts=self.n_state_pts,
                               plate_gap=self.plate_gap,
                               ld2=self.ld2,
                               model_order=self.ctrl_order,
                               w_ctrl_order=self.w_n,
                               d_ctrl_order=self.d_n,
                               state_pts=self.state_pts_all,
                               state_dz=self.state_dz,
                               state_dz2=self.state_dz2,
                               state_ig=self.state_ig,
                               ctrl_key_list=self.ctrl_key_list)
        self.state_a_mat, self.state_ctrl_mats, self.inv22a21 = pars_to_state_space_v2(state_pars_dict,
                                                                                       return_inv_mat=True)
        self.state_keys = ['pr000']
        for i in range(1, self.ctrl_order):
            self.state_keys.extend([f'pr{i:0=3d}', f'pi{i:0=3d}'])
        self.controller = None
        for key in self.full_state0.keys():
            setattr(self, key, self.full_state0[key])
        for horizon in range(self.n_horizon + 1):
            setattr(self, f'x{horizon}', np.zeros(self.n_x))
        for horizon in range(self.n_horizon):
            setattr(self, f'u{horizon}', np.ones(self.n_u))

        self._wr001_coeffs = asnumpy(xp.polyfit(xp.asarray(self.wr001_pts),
                                                xp.zeros_like(self.wr001_pts),
                                                self.n_wr001_pts - 1))

        self._wi001_coeffs = asnumpy(xp.polyfit(xp.asarray(self.wi001_pts),
                                                xp.zeros_like(self.wi001_pts),
                                                self.n_wi001_pts - 1))

        half_pts, _, _, self.diff_ig = utils.colloc(31, plate_gap=self.plate_gap / 2, shifted=True)
        self.state_to_left_side = utils.polinterp(self.state_pts, -half_pts)
        self.state_to_right_side = utils.polinterp(self.state_pts, half_pts)

    def initialize(self):
        fode = mpc.getCasadiFunc(self.odefunc, [self.n_x, self.n_u], ["x", "u"],
                                 funcname="fode")
        e = mpc.getCasadiFunc(self.con, [self.n_x, self.n_u], ["x", "u"], "e")

        ell = mpc.getCasadiFunc(self.stagecost,
                                [self.n_x, self.n_u, self.n_x, self.n_u, self.n_u],
                                ["x", "u", "x_sp", "u_sp", "Du"],
                                funcname="l")
        Pf = mpc.getCasadiFunc(self.termcost, [self.n_x, self.n_x],
                               ["x", "x_sp"], funcname="Pf")
        lb = dict(u=-self.w001_max * np.ones(self.n_u))
        ub = dict(u=self.w001_max * np.ones(self.n_u))

        N = {"x": self.n_x, "u": self.n_u, "t": self.n_horizon,
             "c": self.n_time_pts, "e": len(self.con_zpts)}

        funcargs = {"fode": ["x", "u"],
                    "l": ["x", "u", "x_sp", "u_sp", "Du"],
                    "Pf": ["x", "x_sp"],
                    "e": ["x", "u"]}
        sp = dict(x=np.zeros(self.n_x), u=np.ones(self.n_u))

        state = {}
        for key in self.state_keys:
            state_vals = asnumpy(xp.interp(xp.asarray(self.state_pts_all),
                                           xp.asarray(self._env_zpts),
                                           xp.asarray(self.full_state0[key])))
            if 'pr000' in key:
                nbd_int = simpson(state_vals, self.state_pts_all)
            state_vals /= nbd_int
            state[key] = state_vals
        x0, z0 = utils.dict_to_xz(state, self.state_keys)

        mini_sps = self.sp_fcn(0)
        dummy_sp = np.zeros(self.n_x)
        for j in range(len(mini_sps)):
            dummy_sp[j] = mini_sps[j]
        self._state_cost = self.get_state_cost(x0, dummy_sp).flatten()[0]

        nmpcargs = {
            "f": fode,
            "l": ell,
            "Pf": Pf,
            "N": N,
            "x0": x0,
            "lb": lb,
            "ub": ub,
            "Delta": self.ctrl_dt,
            "verbosity": 1,
            "funcargs": funcargs,
            "sp": sp,
            "timelimit": int(1e8),
            "e": e,
            'uprev': np.ones(self.n_u)
        }
        self.controller = mpc.nmpc(**nmpcargs)
        self.controller.initialize(solveroptions=dict(max_iter=int(1e8), max_cpu_time=int(1e8)))

    def odefunc(self, x, u):
        a_mat = self.state_a_mat
        wr001 = u[:self.n_wr001_pts] * np.pi
        wi001 = u[self.n_wr001_pts:] * np.pi
        xdot = (a_mat
                + cs.kron(self.state_ctrl_mats['wr001'],
                          cs.diag(self.wr001_to_state_mat @ wr001))
                + cs.kron(self.state_ctrl_mats['wi001'],
                          cs.diag(self.wi001_to_state_mat @ wi001))
                ) @ x
        return xdot

    def con(self, x, u):
        wr001 = u[:self.n_wr001_pts]
        wi001 = u[self.n_wr001_pts:]
        bound = ((self.wr001_to_con_mat @ wr001) ** 2
                 + (self.wi001_to_con_mat @ wi001) ** 2
                 - self.w001_max ** 2)
        return bound

    def stagecost(self, x, u, x_sp, u_sp, Deltau):
        n = x[0:self.n_state_pts]
        mx = x[self.n_state_pts:2 * self.n_state_pts]
        wr001 = u[:self.n_wr001_pts]
        wi001 = u[self.n_wr001_pts:]

        z_sps = x_sp[[0, 1]]
        mx_ctrl = x_sp[2]

        outer_diffs = np.subtract.outer(self.state_pts, z_sps)
        diffs1 = outer_diffs[:, 0]
        diffs2 = outer_diffs[:, 1]
        diff_mat = cs.diag(cs.fmin((diffs1 ** 2), (diffs2 ** 2)))
        integrand = mpc.util.mtimes(diff_mat, n)
        integrand1 = self.state_to_left_side @ n
        integrand2 = self.state_to_right_side @ n
        state_penalty = self.state_cost_multiplier * mpc.util.mtimes(self.state_ig[1:-1].T, integrand)

        desired_pts = self.mx_fcn(self.state_pts)
        integrand3 = (n * mx_ctrl * desired_pts - mx) ** 2
        mx_penalty = self.mx_state_cost_multiplier * mpc.util.mtimes(self.state_ig[1:-1].T, integrand3)

        diff_penalty = self.diff_cost_multiplier * (mpc.util.mtimes(self.diff_ig.T, integrand1)
                                                    - mpc.util.mtimes(self.diff_ig.T, integrand2)) ** 2

        d2wr001dz2 = mpc.util.mtimes(self.wr001_dz2, wr001)
        dz2_wr001_penalty = self.wr001_dz2_ctrl_cost_multiplier * mpc.util.mtimes(d2wr001dz2.T, d2wr001dz2)

        d2wi001dz2 = mpc.util.mtimes(self.wi001_dz2, wi001)
        dz2_wi001_penalty = self.wi001_dz2_ctrl_cost_multiplier * mpc.util.mtimes(d2wi001dz2.T, d2wi001dz2)

        dt_penalty = self.dt_ctrl_cost_multiplier * mpc.util.mtimes(Deltau.T, Deltau)
        penalty = (state_penalty + diff_penalty + mx_penalty +
                   dz2_wr001_penalty + dz2_wi001_penalty + dt_penalty)
        return penalty

    def get_state_cost(self, x, x_sp):
        n = x[0:self.n_state_pts]
        mx = x[self.n_state_pts:2 * self.n_state_pts]

        z_sps = np.array(x_sp[[0, 1]]).flatten()
        mx_ctrl = x_sp[2]

        outer_diffs = np.subtract.outer(self.state_pts, z_sps)
        diffs1 = outer_diffs[:, 0]
        diffs2 = outer_diffs[:, 1]
        diff_mat = cs.diag(cs.fmin((diffs1 ** 2), (diffs2 ** 2)))
        integrand = mpc.util.mtimes(diff_mat, n)
        integrand1 = self.state_to_left_side @ n
        integrand2 = self.state_to_right_side @ n
        state_penalty = self.state_cost_multiplier * mpc.util.mtimes(self.state_ig[1:-1].T, integrand)

        desired_pts = self.mx_fcn(self.state_pts)
        integrand3 = (n * mx_ctrl * desired_pts - mx) ** 2
        mx_penalty = self.mx_state_cost_multiplier * mpc.util.mtimes(self.state_ig[1:-1].T, integrand3)

        diff_penalty = self.diff_cost_multiplier * (mpc.util.mtimes(self.diff_ig.T, integrand1)
                                                    - mpc.util.mtimes(self.diff_ig.T, integrand2)) ** 2

        penalty = (state_penalty + diff_penalty + mx_penalty)
        return penalty

    def get_ctrl_cost(self, u, Deltau):
        wr001 = u[:self.n_wr001_pts]
        wi001 = u[self.n_wr001_pts:]

        d2wr001dz2 = mpc.util.mtimes(self.wr001_dz2, wr001)
        dz2_wr001_penalty = self.wr001_dz2_ctrl_cost_multiplier * mpc.util.mtimes(d2wr001dz2.T, d2wr001dz2)

        d2wi001dz2 = mpc.util.mtimes(self.wi001_dz2, wi001)
        dz2_wi001_penalty = self.wi001_dz2_ctrl_cost_multiplier * mpc.util.mtimes(d2wi001dz2.T, d2wi001dz2)

        dt_penalty = self.dt_ctrl_cost_multiplier * mpc.util.mtimes(Deltau.T, Deltau)
        penalty = (dz2_wr001_penalty + dz2_wi001_penalty + dt_penalty)
        return penalty

    def termcost(self, x, x_sp):
        n = x[0:self.n_state_pts]
        mx = x[self.n_state_pts:2 * self.n_state_pts]

        z_sps = x_sp[[0, 1]]
        mx_ctrl = x_sp[2]

        outer_diffs = np.subtract.outer(self.state_pts, z_sps)
        diffs1 = outer_diffs[:, 0]
        diffs2 = outer_diffs[:, 1]
        diff_mat = cs.diag(cs.fmin((diffs1 ** 2), (diffs2 ** 2)))
        integrand = mpc.util.mtimes(diff_mat, n)
        integrand1 = self.state_to_left_side @ n
        integrand2 = self.state_to_right_side @ n
        state_penalty = self.state_cost_multiplier * mpc.util.mtimes(self.state_ig[1:-1].T, integrand)

        desired_pts = self.mx_fcn(self.state_pts)
        integrand3 = (n * mx_ctrl * desired_pts - mx) ** 2
        mx_penalty = self.mx_state_cost_multiplier * mpc.util.mtimes(self.state_ig[1:-1].T, integrand3)

        diff_penalty = self.diff_cost_multiplier * (mpc.util.mtimes(self.diff_ig.T, integrand1)
                                                    - mpc.util.mtimes(self.diff_ig.T, integrand2)) ** 2
        penalty = state_penalty + diff_penalty + mx_penalty
        return penalty

    def w_ctrl_fcn(self, z):
        wr001 = xp.polyval(xp.asarray(self._wr001_coeffs), z)
        wi001 = xp.polyval(xp.asarray(self._wi001_coeffs), z)
        return wr001, wi001

    def set_forces(self, timestep):
        if timestep % (self.sample_dt / self.sim_dt) == 0:
            pcl_pos, pcl_thetas = self.measure(evalulate_meas_vars=True)
        else:
            pcl_pos, pcl_thetas = self.measure(evalulate_meas_vars=False)
        if timestep % int(self.ctrl_dt / self.sim_dt) == 0:
            sim_time = timestep * self.sim_dt
            for i in range(self.n_horizon + 1):
                rel_t = sim_time + (i * self.ctrl_dt)
                mini_sps = self.sp_fcn(rel_t)
                dummy_sp = np.zeros(self.n_x)
                for j in range(len(mini_sps)):
                    dummy_sp[j] = mini_sps[j]
                self.controller.par["x_sp", i] = dummy_sp
            state = {}
            for key in self.state_keys:
                state_vals = asnumpy(xp.interp(xp.asarray(self.state_pts_all),
                                               xp.asarray(self._env_zpts),
                                               xp.asarray(self.full_state[key])))
                if 'pr000' in key:
                    nbd_int = simpson(state_vals, self.state_pts_all)
                state_vals /= nbd_int
                state[key] = state_vals
            x, z = utils.dict_to_xz(state, self.state_keys)
            self.controller.fixvar("x", 0, x)
            toc = time()
            self.controller.solve()
            tic = time()
            self._solve_time = tic - toc

            self._state_cost = self.get_state_cost(
                np.squeeze(self.controller.var["x", 0]),
                np.squeeze(self.controller.par["x_sp", 0])
            ).flatten()[0]
            self._ctrl_cost = self.get_ctrl_cost(
                np.squeeze(self.controller.var["u", 0]),
                np.squeeze(self.controller.var["Du", 0])
            ).flatten()[0]
            for horizon in range(self.n_horizon + 1):
                setattr(self, f'x{horizon}', np.squeeze(self.controller.var["x", horizon]))
            for horizon in range(self.n_horizon):
                setattr(self, f'u{horizon}', np.squeeze(self.controller.var["u", horizon]))
            if 'Infeasible_Problem_Detected' not in self.controller.stats["status"]:
                wr001_root_vals = np.squeeze(self.controller.var["u", 0]).reshape(-1, 1)[:self.n_wr001_pts]
                wi001_root_vals = np.squeeze(self.controller.var["u", 0]).reshape(-1, 1)[self.n_wr001_pts:]
                self._wr001_coeffs = asnumpy(xp.polyfit(xp.asarray(self.wr001_pts),
                                                        xp.asarray(wr001_root_vals.flatten()),
                                                        self.n_wr001_pts - 1))
                self._wi001_coeffs = asnumpy(xp.polyfit(xp.asarray(self.wi001_pts),
                                                        xp.asarray(wi001_root_vals.flatten()),
                                                        self.n_wi001_pts - 1))

        with self.gpu_local_force_arrays as arrays:
            wr001, wi001 = self.w_ctrl_fcn(pcl_pos)
            arrays.torque[:, -1] = (wr001 * xp.cos(pcl_thetas) + wi001 * xp.sin(pcl_thetas)) * self.zeta_r

    @property
    def solve_time(self):
        return self._solve_time

    @property
    def wr001_coeffs(self):
        return self._wr001_coeffs

    @property
    def wi001_coeffs(self):
        return self._wi001_coeffs

    @property
    def state_cost(self):
        return self._state_cost

    @property
    def ctrl_cost(self):
        return self._ctrl_cost

if __name__ == "__main__":
    main()

