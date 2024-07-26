# [depends] init_cond0.gsd
# [makes] gsd

import gsd.hoomd
import hoomd
import numpy as np
from lib import hoomd_utils
import os


def main():
    sim_time = 50
    ld2 = 10
    n_particles = int(1e6)

    w = 10  # box width
    h = 10  # box height

    dt = 0.001
    sample_dt = 0.1
    dirname = os.path.dirname(__file__)

    init_filename = os.path.join(
        dirname,
        "build/init_cond0.gsd")
    dest_filename = os.path.join(
        dirname,
        "build/init_cond1.gsd")
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
    y = np.random.uniform(-h / 2 + 2 * a, h / 2 - 2 * a, n_particles)
    thetas = np.random.uniform(0, 2 * np.pi, n_particles).reshape(-1, 1)
    # Build HOOMD

    x = np.random.uniform(-w / 2, w / 2, n_particles)

    position = np.column_stack((x, y, np.zeros(n_particles)))
    position = [tuple(e) for e in position]

    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = n_particles
    snapshot.configuration.box = [w, h + 2 * 2 ** (1 / 6) * a, 0, 0, 0, 0]
    snapshot.particles.typeid = [0] * n_particles
    snapshot.particles.position = list(position[0:n_particles])
    snapshot.particles.types = ['A']
    snapshot.particles.moment_inertia = np.ones(
        (n_particles, 3)) * 2 / 5  # note we are picking rho so m=1 mu
    snapshot.particles.diameter = np.ones(n_particles) * 2 * a

    qx, qy, qz, qw = hoomd_utils.get_quaternion_from_euler(thetas, 0, 0)
    snapshot.particles.orientation = np.hstack((qx, qy, qz, qw))
    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=seed)
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

    all = hoomd.filter.All()
    active1 = hoomd.md.force.Active(filter=hoomd.filter.All())
    active1.active_force['A'] = (fp, 0, 0)
    integrator.forces.append(active1)
    walls = [hoomd.wall.Plane((0, -h / 2 - 2 ** (1 / 6) * a, 0), (0, 1, 0)),
             hoomd.wall.Plane((0, h / 2 + 2 ** (1 / 6) * a, 0), (0, -1, 0))]
    lj = hoomd.md.external.wall.ForceShiftedLJ(walls=walls)
    lj.params['A'] = {"sigma": a, "epsilon": 5.0, "r_cut": a * (2 ** (1 / 6))}

    integrator.forces.append(lj)
    logger = hoomd.logging.Logger()
    logger['dt'] = (lambda: dt, 'scalar')
    logger['u0'] = (lambda: u0, 'scalar')
    logger['ld2'] = (lambda: ld2, 'scalar')
    logger['zeta_r'] = (lambda: zeta_r, 'scalar')
    n_steps = sim_time * (int(np.round(1 / dt)))
    gsd_writer = hoomd.write.GSD(filename=dest_filename,
                                 trigger=hoomd.trigger.On(n_steps-1),
                                 mode='wb',
                                 truncate=True)
    gsd_writer.log = logger
    sim.operations.writers.append(gsd_writer)
    sim.run(n_steps)


if __name__ == "__main__":
    main()

