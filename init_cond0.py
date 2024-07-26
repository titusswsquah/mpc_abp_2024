# [makes] gsd

import gsd.hoomd
import numpy as np
from lib import hoomd_utils
import os


def main():
    ld2 = 10
    n_particles = int(1e6)

    w = 10  # box width
    h = 10  # box height

    dirname = os.path.dirname(__file__)

    init_filename = os.path.join(
        dirname,
        "build/init_cond0.gsd")
    seed = 10

    ell = 1  # ell: run length,
    D_r = 1  # tau^{-1}: tau_r
    zeta_t = 1  # mu•tau^{-1}: mu = mass of particle

    delta = ell / np.sqrt(ld2)  # delta: microscopic length
    a = np.sqrt(3 / 4) * delta  # ell: particle radius

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
    with gsd.hoomd.open(name=init_filename, mode='wb+') as f:
        f.append(snapshot)


if __name__ == "__main__":
    main()

