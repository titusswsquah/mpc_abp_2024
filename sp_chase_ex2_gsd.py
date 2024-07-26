# [depends] init_cond1.gsd
# [makes] gsd
import os
import gsd.hoomd
import hoomd
import numpy as np
from lib import utils
from lib.hoomd_utils import BaseControlPolicy
from lib import hoomd_utils
import casadi as cs
import mpctools as mpc
from time import time
from scipy.integrate import simpson
from lib.eqn_generator import pars_to_state_space_v2
from lib.utils import get_array_module, array_module, asnumpy
from scipy.special import iv
from scipy.linalg import expm
from casadi import Importer

xp = array_module('cupy')


def analytical(x):
    return iv(1, x) / iv(0, x)


class SetPoint:
    def __init__(self):
        self.w001_max = 3
        self.control_area = 8
        self.max_speed = analytical(self.w001_max)
        self.oscillations_start = 15
        self.scaled_speed = 1.25 * self.max_speed
        self.period = self.control_area / 2 / self.scaled_speed
        self.n_periods = 1.25
        self.oscillations_stop = (self.period * 2 * np.pi * self.n_periods
                                  + self.oscillations_start)
        self.sim_time = self.oscillations_stop + 5

    def sp_fcn(self, _time):
        if self.oscillations_start < _time < self.oscillations_stop:
            rel_t = _time - self.oscillations_start - self.period * np.pi / 2
            pos_sp = self.control_area / 2 * np.sin(rel_t / self.period)
        else:
            pos_sp = 0.0
        return pos_sp


def main():
    tracer_seed = 20
    n_traced = 2000
    dirname, basename = utils.get_dir_base(__file__)
    init_filename = os.path.join(
        dirname,
        "build/init_cond1.gsd")
    dest_filename = os.path.join(
        dirname,
        f"build/{basename}.gsd")
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
    walls = [
        hoomd.wall.Plane((0, -plate_gap / 2 - 2 ** (1 / 6) * a, 0), (0, 1, 0)),
        hoomd.wall.Plane((0, plate_gap / 2 + 2 ** (1 / 6) * a, 0), (0, -1, 0))]
    lj = hoomd.md.external.wall.ForceShiftedLJ(walls=walls)
    lj.params['A'] = {"sigma": a, "epsilon": 5.0, "r_cut": a * (2 ** (1 / 6))}
    sp_obj = SetPoint()

    integrator.forces.append(lj)
    integrator.forces.append(hoomd_controller)
    logger = hoomd.logging.Logger()
    logger['dt'] = (lambda: sim_dt, 'scalar')
    # logger['u0'] = (lambda: u0, 'scalar')
    logger['ld2'] = (lambda: ld2, 'scalar')
    logger['zeta_r'] = (lambda: zeta_r, 'scalar')
    logger['wr001_coeffs'] = (hoomd_controller, 'wr001_coeffs', 'sequence')
    logger['env_zpts'] = (hoomd_controller, '_env_zpts', 'sequence')
    logger['solve_time'] = (hoomd_controller, 'solve_time', 'scalar')
    logger['solve_status'] = (hoomd_controller, 'solve_status', 'string')
    logger['guess_type'] = (hoomd_controller, 'guess_type', 'string')
    logger['state_cost'] = (hoomd_controller, 'state_cost', 'scalar')
    logger['ctrl_cost'] = (hoomd_controller, 'ctrl_cost', 'scalar')
    logger['x_sp'] = (hoomd_controller, 'x_sp', 'scalar')
    logger['plate_gap'] = (hoomd_controller, 'plate_gap', 'scalar')
    logger['w001_max'] = (hoomd_controller, 'w001_max', 'scalar')
    for key in full_state0.keys():
        logger[key] = (hoomd_controller, key, 'sequence')
    for horizon in range(hoomd_controller.n_horizon + 1):
        logger[f'x{horizon}'] = (hoomd_controller, f'x{horizon}', 'sequence')
    for horizon in range(hoomd_controller.n_horizon):
        logger[f'u{horizon}'] = (hoomd_controller, f'u{horizon}', 'sequence')
    gsd_writer = hoomd.write.GSD(filename=dest_filename,
                                 trigger=hoomd.trigger.Periodic(
                                     int(np.round(1 / (sim_dt / sample_dt)))),
                                 mode='wb', filter=traced)
    gsd_writer.log = logger
    sim.operations.writers.append(gsd_writer)

    logger = hoomd.logging.Logger(categories=['scalar', 'string'])
    logger.add(sim, quantities=['timestep', 'final_timestep'])
    logger['solve_time'] = (hoomd_controller, 'solve_time', 'scalar')
    logger['solve_status'] = (hoomd_controller, 'solve_status', 'string')
    logger['guess_type'] = (hoomd_controller, 'guess_type', 'string')
    file = open('sp_chase_ex2_gsd.log', mode='w', newline='\n')
    table = hoomd.write.Table(
        output=file,
        trigger=hoomd.trigger.Periodic(
            int(np.round(1 / (sim_dt / hoomd_controller.ctrl_dt)))),
        logger=logger)
    sim.operations.writers.append(table)

    sim_time = sp_obj.sim_time
    sim.run(sim_time * (int(np.round(1 / sim_dt))))
    file.close()


class ControlPolicy(BaseControlPolicy):
    def __init__(self,
                 init_filename):
        self.current_status = None
        frame = gsd.hoomd.open(init_filename, 'rb')[0]
        box_height = frame.configuration.box[1]
        self._plate_gap = 10
        self.ld2 = 10

        self.sample_dt = 0.1

        self.sim_dt = 0.001

        self.meas_order = 10
        self.bin_size = 0.00115
        pcl_pos = np.clip(frame.particles.position[:, 1], -self._plate_gap / 2,
                          self._plate_gap / 2)
        pcl_quarts = xp.asarray(frame.particles.orientation)
        pcl_thetas = \
            hoomd_utils.quaternion_to_euler_angle_vectorized(
                pcl_quarts[:, 0],
                pcl_quarts[:, 1],
                pcl_quarts[:, 2],
                pcl_quarts[:, 3])[-1] * np.pi / 180
        self.bd_zpts = np.linspace(-box_height / 2, box_height / 2, 1001)
        self.meas_zpts = np.linspace(-self._plate_gap / 2, self._plate_gap / 2,
                                     301)
        self.full_state0 = hoomd_utils.state_eval(pcl_pos, asnumpy(pcl_thetas),
                                                  self.bin_size,
                                                  self.bd_zpts, self.meas_zpts,
                                                  self.meas_order)

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
                         self._plate_gap,
                         self.bin_size,
                         self.bd_zpts,
                         self.meas_zpts,
                         self.meas_order)

        self.n_state_pts = 40
        self.n_wr001_pts = 20
        self.ctrl_order = 5

        self.n_time_pts = 1
        self.n_horizon = 14
        self.ctrl_dt = 0.5
        self.ctrl_key_list = ['wr001']
        self.constrained_zpts = np.linspace(-self._plate_gap / 2,
                                            self._plate_gap / 2,
                                            80)

        sp_obj = SetPoint()
        self.sp_fcn = sp_obj.sp_fcn
        self._w001_max = sp_obj.w001_max

        self.state_cost_multiplier = 6400
        self.term_cost_multiplier = self.state_cost_multiplier
        self.dz2_ctrl_cost_multiplier = 5e-5
        self.dt_ctrl_cost_multiplier = 0.001

        self._solve_time = 0
        self._state_cost = 0
        self._ctrl_cost = 0
        self._x_sp = 0

        self._guess_type = ''

        state_info = utils.colloc(
            self.n_state_pts,
            left=True,
            right=True,
            plate_gap=self._plate_gap,
            shifted=False)
        state_pts_all, self.state_dz, self.state_dz2, self.state_ig = state_info
        self.state_pts_all = state_pts_all.flatten()
        self.state_pts = self.state_pts_all[1:-1]
        wr001_info = utils.colloc(
            self.n_wr001_pts,
            plate_gap=self._plate_gap,
            shifted=False)
        (self.wr001_pts, self.wr001_dz,
         self.wr001_dz2, self.wr001_ig) = wr001_info
        self.wr001_pts = self.wr001_pts.flatten()
        self.wr001_to_state_mat = utils.polinterp(self.wr001_pts,
                                                  self.state_pts)
        self.wr001_to_constrained_z = utils.polinterp(self.wr001_pts,
                                                      self.constrained_zpts)
        self.w_n = 2
        self.d_n = 1

        n_vars = (2 * self.ctrl_order - 1)
        self.n_x = n_vars * self.n_state_pts
        self.n_u = self.n_wr001_pts

        state_pars_dict = dict(n_state_pts=self.n_state_pts,
                               plate_gap=self._plate_gap,
                               ld2=self.ld2,
                               model_order=self.ctrl_order,
                               w_ctrl_order=self.w_n,
                               d_ctrl_order=self.d_n,
                               state_pts=self.state_pts_all,
                               state_dz=self.state_dz,
                               state_dz2=self.state_dz2,
                               state_ig=self.state_ig,
                               ctrl_key_list=self.ctrl_key_list)
        (self.state_a_mat,
         self.state_ctrl_mats,
         self.inv22a21) = pars_to_state_space_v2(
            state_pars_dict,
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

    def initialize(self):
        with_jit = False
        compiler = ''
        jit_options = dict(jit=with_jit, compiler=compiler, cse=True)
        fode = mpc.getCasadiFunc(self.odefunc, [self.n_x, self.n_u], ["x", "u"],
                                 funcname="fode",
                                 casadi_options=jit_options,
                                 casaditype='SX')
        e = mpc.getCasadiFunc(self.con, [self.n_x, self.n_u], ["x", "u"], "e",
                              casadi_options=jit_options, )

        ell = mpc.getCasadiFunc(self.stagecost,
                                [self.n_x, self.n_u, self.n_x, self.n_u,
                                 self.n_u],
                                ["x", "u", "x_sp", "u_sp", "Du"],
                                funcname="l",
                                casadi_options=jit_options, )
        Pf = mpc.getCasadiFunc(self.termcost, [self.n_x, self.n_x],
                               ["x", "x_sp"], funcname="Pf",
                               casadi_options=jit_options, )
        lb = dict(u=-self._w001_max * np.ones(self.n_u))
        ub = dict(u=self._w001_max * np.ones(self.n_u))

        N = {"x": self.n_x, "u": self.n_u, "t": self.n_horizon,
             "c": self.n_time_pts, "e": len(self.constrained_zpts)}

        funcargs = {"fode": ["x", "u"],
                    "l": ["x", "u", "x_sp", "u_sp", "Du"],
                    "Pf": ["x", "x_sp"],
                    "e": ["x", "u"]}
        sp = dict(x=np.zeros(self.n_x), u=np.ones(self.n_u))

        state = {}
        nbd_int = 1
        for key in self.state_keys:
            state_coeffs = xp.polyfit(xp.asarray(self._env_zpts),
                                      xp.asarray(self.full_state0[key]),
                                      self.n_state_pts + 1)
            state_vals = asnumpy(xp.polyval(state_coeffs,
                                            xp.asarray(self.state_pts_all)))

            if 'pr000' in key:
                nbd_int = self.state_ig.flatten() @ state_vals.flatten()
            state_vals /= nbd_int
            state[key] = state_vals
        x0, z0 = utils.dict_to_xz(state, self.state_keys)

        guess = dict(
            x=np.tile(x0, (self.n_horizon + 1, 1))
        )

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
            'uprev': np.ones(self.n_u),
            'casaditype': 'SX',
            'guess': guess
        }
        jit_options.pop('cse')
        self.controller = mpc.nmpc(**nmpcargs)

        self.controller.initialize(
            solveroptions=dict(max_iter=int(1e8), max_cpu_time=int(1e8)),
            casadioptions=jit_options)
        self.current_status = "None"

    def odefunc(self, x, u):
        a_mat = self.state_a_mat
        ctrl_mats = self.state_ctrl_mats
        conv_mat = self.wr001_to_state_mat
        xdot = (a_mat
                + cs.kron(ctrl_mats['wr001'],
                          cs.diag(conv_mat @ u * np.pi))) @ x
        return xdot

    def con(self, x, u):
        bound = (self.wr001_to_constrained_z @ u) ** 2 - self._w001_max ** 2
        return bound

    def stagecost(self, x, u, x_sp, u_sp, Deltau):
        n = x[0:self.n_state_pts]
        z_sp = x_sp[0]
        integrand = n * (self.state_pts - z_sp) ** 2
        d2udz2 = mpc.util.mtimes(self.wr001_dz2, u)

        cost = (self.state_cost_multiplier * mpc.util.mtimes(
            self.state_ig[1:-1].T, integrand)
                + self.dz2_ctrl_cost_multiplier * mpc.util.mtimes(d2udz2.T,
                                                                  d2udz2)
                + self.dt_ctrl_cost_multiplier * mpc.util.mtimes(Deltau.T,
                                                                 Deltau))
        return cost

    def get_state_cost(self, x_sp):
        z_sp = x_sp
        integrand = self.full_state['pr000'] * (self.env_zpts - z_sp) ** 2

        cost = self.state_cost_multiplier * simpson((integrand, self.env_zpts))
        return cost

    def get_ctrl_cost(self, u, Deltau):
        d2udz2 = mpc.util.mtimes(self.wr001_dz2, u)

        cost = (self.dz2_ctrl_cost_multiplier * mpc.util.mtimes(d2udz2.T,
                                                                d2udz2)
                + self.dt_ctrl_cost_multiplier * mpc.util.mtimes(Deltau.T,
                                                                 Deltau))
        return cost

    def termcost(self, x, x_sp):
        n = x[0:self.n_state_pts]
        z_sp = x_sp[0]
        integrand = n * (self.state_pts - z_sp) ** 2
        return self.state_cost_multiplier * mpc.util.mtimes(
            self.state_ig[1:-1].T, integrand)

    def w_ctrl_fcn(self, z):
        wr001 = xp.polyval(xp.asarray(self._wr001_coeffs), z)
        return wr001

    def state_exact_step(self, _x, _u, _dt, _ctrl_key_list):
        ctrl_part = np.kron(self.state_ctrl_mats['wr001'], np.diag(_u))
        return expm((self.state_a_mat + ctrl_part) * _dt) @ _x

    def set_forces(self, timestep):
        if timestep % (self.sample_dt / self.sim_dt) == 0:
            pcl_pos, pcl_thetas = self.measure(evalulate_meas_vars=True)
            sim_time = timestep * self.sim_dt
            self._x_sp = self.sp_fcn(sim_time)
            self._state_cost = self.get_state_cost(self._x_sp)
        else:
            pcl_pos, pcl_thetas = self.measure(evalulate_meas_vars=False)
        if timestep % int(self.ctrl_dt / self.sim_dt) == 0:
            sim_time = timestep * self.sim_dt
            for i in range(self.n_horizon + 1):
                sp = self.sp_fcn(sim_time + (i * self.ctrl_dt))
                self.controller.par["x_sp", i] = (
                        np.ones(self.n_x) * sp
                )
            state = {}
            nbd_int = 1
            for key in self.state_keys:
                state_coeffs = xp.polyfit(xp.asarray(self._env_zpts),
                                          xp.asarray(self.full_state0[key]),
                                          self.n_state_pts + 1)
                state_vals = asnumpy(xp.polyval(state_coeffs,
                                                xp.asarray(self.state_pts_all)))
                # state_vals = asnumpy(xp.interp(xp.asarray(self.state_pts_all),
                #                                xp.asarray(self._env_zpts),
                #                                xp.asarray(self.full_state0[key])))
                if 'pr000' in key:
                    nbd_int = self.state_ig.flatten() @ state_vals.flatten()
                state_vals /= nbd_int
                state[key] = state_vals
            x, z = utils.dict_to_xz(state, self.state_keys)

            x_guess = np.array(self.controller.var['x', 1:]).squeeze(-1)
            xc_guess = np.array(self.controller.var['xc', 1:])
            u_guess = np.array(self.controller.var['u', 1:]).squeeze(-1)
            u_guess = np.vstack((u_guess,
                                 u_guess[-1]))
            du_guess = np.array(self.controller.var['Du', 1:]).squeeze(-1)
            du_guess = np.vstack((du_guess,
                                  np.zeros(self.n_u)))

            w_ctrl = (self.wr001_to_state_mat @ u_guess[-1,
                                                :self.n_wr001_pts])
            xp1_guess = self.state_exact_step(x_guess[-1],
                                              w_ctrl,
                                              self.ctrl_dt,
                                              self.ctrl_key_list)
            x_guess = np.vstack((x_guess,
                                 xp1_guess))

            xcp1_guess = [self.state_exact_step(x_guess[-1],
                                                w_ctrl,
                                                _dt, self.ctrl_key_list)
                          for _dt in
                          self.controller.misc['colloc']['r'][1:-1]]
            xcp1_guess = np.expand_dims(np.array(xcp1_guess).T, 0)
            xc_guess = np.vstack((xc_guess,
                                  xcp1_guess))
            emergency_guess = dict(x=x_guess,
                                   xc=xc_guess,
                                   u=u_guess,
                                   Du=du_guess)

            self.controller.fixvar("x", 0, x)
            self._guess_type = 'first'
            toc = time()
            self.controller.solve()
            tic = time()
            if 'Solve' not in self.controller.stats["status"]:
                self._guess_type = 'last'
                self.controller.saveguess(newguess=emergency_guess,
                                          infercolloc=False)
                toc = time()
                self.controller.solve()
                tic = time()
            self.controller.saveguess(default=True)
            self._solve_time = tic - toc

            self._ctrl_cost = self.get_ctrl_cost(
                self.controller.var["u", 0],
                self.controller.var["Du", 0]
            ).flatten()[0]
            for horizon in range(self.n_horizon + 1):
                setattr(self, f'x{horizon}',
                        np.squeeze(self.controller.var["x", horizon]))
            for horizon in range(self.n_horizon):
                setattr(self, f'u{horizon}',
                        np.squeeze(self.controller.var["u", horizon]))
            self.current_status = self.controller.stats["status"]
            if 'Solve' in self.controller.stats["status"]:
                root_vals = np.squeeze(
                    self.controller.var["u", 0]).reshape(-1, 1)
                self._wr001_coeffs = asnumpy(
                    xp.polyfit(xp.asarray(self.wr001_pts),
                               xp.asarray(root_vals.flatten()),
                               self.n_wr001_pts - 1))

        with self.gpu_local_force_arrays as arrays:
            arrays.torque[:, -1] = self.w_ctrl_fcn(pcl_pos) * xp.cos(
                pcl_thetas) * self.zeta_r

    @property
    def solve_time(self):
        return self._solve_time

    @property
    def solve_status(self):
        return self.current_status

    @property
    def wr001_coeffs(self):
        return self._wr001_coeffs

    @property
    def state_cost(self):
        return self._state_cost

    @property
    def ctrl_cost(self):
        return self._ctrl_cost

    @property
    def x_sp(self):
        return self._x_sp

    @property
    def plate_gap(self):
        return self._plate_gap

    @property
    def w001_max(self):
        return self._w001_max

    @property
    def guess_type(self):
        return self._guess_type


if __name__ == "__main__":
    main()

