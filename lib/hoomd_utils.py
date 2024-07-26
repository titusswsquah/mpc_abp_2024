import hoomd
import numpy as np
from scipy.interpolate import interp1d
from abc import abstractmethod
from lib.utils import get_array_module, array_module, asnumpy
from numba import jit
from KDEpy import FFTKDE
from KDEpy.binning import grid_is_sorted
from KDEpy.utils import cartesian
from lib.my_binning import linear_binning
from scipy.integrate import simpson
import numbers
from collections.abc import Sequence
import warnings
from scipy.signal import convolve
from scipy.interpolate import CubicSpline


class BaseControlPolicy(hoomd.md.force.Custom):
    def __init__(self,
                 full_state0,
                 plate_gap,
                 bin_size,
                 bd_zpts,
                 env_zpts,
                 meas_order
                 ):
        super().__init__(aniso=False)
        self._plate_gap = plate_gap
        self.xp = array_module('cupy')

        self.bin_size = bin_size
        self.bd_zpts = bd_zpts
        self._env_zpts = env_zpts
        self.meas_order = meas_order
        self.full_state = None
        for key in full_state0.keys():
            setattr(self, key, full_state0[key])

    def measure(self,
                evalulate_meas_vars):
        with self._state.gpu_local_snapshot as snapshot:
            pcl_pos = self.xp.asarray(snapshot.particles.position)[:, 1]
            pcl_quarts = self.xp.asarray(snapshot.particles.orientation)
            pcl_orient = gpu_quaternion_to_euler_angle_vectorized2(
                pcl_quarts[:, 0],
                pcl_quarts[:, 1],
                pcl_quarts[:, 2],
                pcl_quarts[:, 3], self.xp)
            pcl_thetas = pcl_orient[-1] * np.pi / 180
            if evalulate_meas_vars:
                self.full_state = state_eval(np.clip(asnumpy(pcl_pos),
                                                     -self._plate_gap / 2,
                                                     self._plate_gap / 2),
                                             asnumpy(pcl_thetas),
                                             self.bin_size,
                                             self.bd_zpts,
                                             self._env_zpts,
                                             self.meas_order)

                for key in self.full_state.keys():
                    setattr(self, key, self.full_state[key])
            return pcl_pos, pcl_thetas

    @abstractmethod
    def set_forces(self, timestep):
        pass

    @property
    def env_zpts(self):
        return self._env_zpts


@jit(nopython=True)
def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return qx, qy, qz, qw


def quaternion_to_euler_angle_vectorized(w, x, y, z):
    xp = array_module('cupy')
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = xp.degrees(xp.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = xp.clip(t2, a_min=-1.0, a_max=1.0)
    Y = xp.degrees(xp.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = xp.degrees(xp.arctan2(t3, t4))

    return X, Y, Z


class MyFFTKDE(FFTKDE):
    def __init__(self, kernel="gaussian", bw=1, norm=2):
        self.norm = norm
        super().__init__(kernel, bw, norm)
        assert isinstance(self.norm, numbers.Number) and self.norm > 0

    def fit(self, data, weights=None):
        """
        Fit the kernel density estimator to the data.
        This method converts the data to shape (obs, dims) and the weights
        to (obs,).

        Parameters
        ----------
        data : array-like or Sequence
            May be array-like of shape (obs,), shape (obs, dims) or a
            Python Sequence, e.g. a list or tuple.
        weights : array-like, Sequence or None
            May be array-like of shape (obs,), shape (obs, dims), a
            Python Sequence, e.g. a list or tuple, or None.
        """

        # -------------- Set up the data depending on input -------------------
        # In the end, the data should be an ndarray of shape (obs, dims)
        data = self._process_sequence(data)

        obs, dims = data.shape

        if not obs > 0:
            raise ValueError("Data must contain at least one data point.")
        assert dims > 0
        self.data = data

        # -------------- Set up the weights depending on input ----------------
        if weights is not None:
            self.weights = self._process_sequence(weights).ravel()
            self.weights = self.weights
            if not obs == len(self.weights):
                raise ValueError("Number of data obs must match weights")
        else:
            self.weights = weights

        # TODO: Move bandwidth selection from evaluate to fit

        # Test quickly that the method has done what is was supposed to do
        assert len(self.data.shape) == 2
        if self.weights is not None:
            assert len(self.weights.shape) == 1
            assert self.data.shape[0] == len(self.weights)

        if isinstance(self.bw_method, (np.ndarray, Sequence)):
            self.bw = self.bw_method
        elif callable(self.bw_method):
            self.bw = self.bw_method(self.data, self.weights)
        else:
            self.bw = self.bw_method
        return self

    def evaluate(self, grid_points=None):
        super(FFTKDE, self).evaluate(grid_points)

        # Extra verification for FFTKDE (checking the sorting property)
        if not grid_is_sorted(self.grid_points):
            raise ValueError("The grid must be sorted.")

        if isinstance(self.bw, numbers.Number) and self.bw > 0:
            bw = self.bw
        else:
            raise ValueError("The bw must be a number.")
        self.bw = bw

        # Step 0 - Make sure data points are inside of the grid
        min_grid = np.min(self.grid_points, axis=0)
        max_grid = np.max(self.grid_points, axis=0)

        min_data = np.min(self.data, axis=0)
        max_data = np.max(self.data, axis=0)
        if not ((min_grid < min_data).all() and (max_grid > max_data).all()):
            raise ValueError("Every data point must be inside of the grid.")

        # Step 1 - Obtaining the grid counts
        # TODO: Consider moving this to the fitting phase instead
        data = linear_binning(self.data, grid_points=self.grid_points,
                              weights=self.weights)

        # Step 2 - Computing kernel weights
        g_shape = self.grid_points.shape[1]
        num_grid_points = np.array(list(
            len(np.unique(self.grid_points[:, i])) for i in range(g_shape)))

        num_intervals = num_grid_points - 1
        dx = (max_grid - min_grid) / num_intervals

        # Find the real bandwidth, the support times the desired bw factor
        if self.kernel.finite_support:
            real_bw = self.kernel.support * self.bw
        else:
            # The parent class should compute this already. If not, compute
            # it again. This optimization only dominates a little bit with
            # few data points
            try:
                real_bw = self._kernel_practical_support
            except AttributeError:
                real_bw = self.kernel.practical_support(self.bw)

        # Compute L, the number of dx'es to move out from 0 in kernel
        L = np.minimum(np.floor(real_bw / dx), num_intervals + 1)
        assert (dx * L <= real_bw).all()

        # Evaluate the kernel once
        grids = [np.linspace(-dx * L, dx * L, int(L * 2 + 1)) for (dx, L) in
                 zip(dx, L)]
        kernel_grid = cartesian(grids)
        kernel_weights = self.kernel(kernel_grid, bw=self.bw, norm=self.norm)

        # Reshape in preparation to
        kernel_weights = kernel_weights.reshape(*[int(k * 2 + 1) for k in L])
        data = data.reshape(*tuple(num_grid_points))

        # Step 3 - Performing the convolution

        # The following code block surpressed the warning:
        #        anaconda3/lib/python3.6/site-packages/mkl_fft/_numpy_fft.py:
        #            FutureWarning: Using a non-tuple sequence for multidimensional ...
        #        output = mkl_fft.rfftn_numpy(a, s, axes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ans = convolve(data, kernel_weights, mode="same").reshape(-1, 1)

        return self._evalate_return_logic(ans, self.grid_points)


def state_kde(pcl_pos, pcl_thetas, bin_size, model_order):
    kde_dict = {}
    for order in range(model_order):
        if order == 0:
            kde_dict['pr{0:0=3d}'.format(order)] = MyFFTKDE(kernel='gaussian',
                                                            bw=bin_size).fit(
                pcl_pos)
        else:
            weights = np.cos(order * pcl_thetas).flatten()
            kde_dict['pr{0:0=3d}'.format(order)] = MyFFTKDE(kernel='gaussian',
                                                            bw=bin_size).fit(
                pcl_pos,
                weights=weights)
            weights = np.sin(order * pcl_thetas).flatten()
            kde_dict['pi{0:0=3d}'.format(order)] = MyFFTKDE(kernel='gaussian',
                                                            bw=bin_size).fit(
                pcl_pos,
                weights=weights)
    return kde_dict


def state_eval(pcl_pos, pcl_thetas, bin_size, bd_zpts, env_zpts, model_order):
    kde_dict = state_kde(pcl_pos.reshape(-1, 1), pcl_thetas.reshape(-1, 1),
                         bin_size, model_order)
    state = {}
    nbd_int = 1

    for key, kde in kde_dict.items():
        state_spline = CubicSpline(bd_zpts, kde.evaluate(bd_zpts))
        state_bd = state_spline(env_zpts).flatten()
        if 'pr000' in key:
            nbd_int = simpson(state_bd, env_zpts.flatten())
        state_bd /= nbd_int
        state[key] = state_bd
    return state


def gpu_quaternion_to_euler_angle_vectorized2(w, x, y, z, xp):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = xp.degrees(xp.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = xp.clip(t2, a_min=-1.0, a_max=1.0)
    Y = xp.degrees(xp.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = xp.degrees(xp.arctan2(t3, t4))

    return X, Y, Z
