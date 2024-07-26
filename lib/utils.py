# import custompath
# custompath.add()
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import casadi as cs
from scipy.special import roots_legendre, roots_chebyu, roots_chebyt, \
    roots_jacobi, roots_hermite, eval_legendre
from types import ModuleType
import inspect
import logging
import numpy as np
import pickle
import re
from lib import color_brewer

_warn_array_module_once = False


def polinterp(xcol, xint):
    ncol = len(xcol)
    nint = len(xint)
    w = np.zeros((nint, ncol))
    for i in range(ncol):
        d = 1.0
        for j in range(ncol):
            if j != i:
                d = d * (xcol[i] - xcol[j])

        for k in range(nint):
            xx = xint[k]
            n = 1.0
            for j in range(ncol):
                if j != i:
                    n = n * (xx - xcol[j])

            w[k, i] = n / d
    return w


def eval_legendre_deriv(n, x):
    return (x * eval_legendre(n, x)
            - eval_legendre(n - 1, x)) / ((x ** 2 - 1) / n)


def export_dict(info_dict, filename):
    exp_dict = info_dict.copy()
    with open(filename, 'wb') as handle:
        pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def import_dict(filename):
    with open(filename, 'rb') as handle:
        info_dict = pickle.load(handle)
    return info_dict


def save_pdf(figure_list, filename, dpi=100):
    with PdfPages(filename) as pdf:
        for figure in figure_list:
            plt.figure(figure.number)
            pdf.savefig(bbox_inches='tight', dpi=dpi)
    return None


def tex_writer(text, tex_file):
    with open(tex_file, 'w') as texfile:
        for i in range(len(text)):
            texfile.write(text[i])
            texfile.write("\n")


def lazy_pdf(figures, script_path, dpi=100, savefig_kwargs=None):
    if savefig_kwargs is None:
        savefig_kwargs = {}
    dirname = os.getcwd()
    filename = (f"{dirname}/"
                f"{os.path.basename(script_path).replace('.py', '.pdf')}")
    with PdfPages(filename) as pdf:
        for figure in figures:
            plt.figure(figure.number)
            pdf.savefig(bbox_inches='tight', dpi=dpi, **savefig_kwargs)


def lazy_pickle_export(info, script_path):
    dirname = os.getcwd()
    in_build_dir = ('build' not in os.listdir())
    if in_build_dir:
        filename = (f"{dirname}/"
                    f"{os.path.splitext(os.path.basename(script_path))[0]}"
                    f".pickle")
    else:
        filename = (f"{dirname}/build/"
                    f"{os.path.splitext(os.path.basename(script_path))[0]}"
                    f".pickle")
    export_dict(info, filename)


def lazy_pickle_import(script_path):
    dirname = os.getcwd()
    in_build_dir = ('build' not in os.listdir())
    suffixes = ['_pdf', '_mp4', '_tex', '_gsd']
    base_name = os.path.splitext(os.path.basename(script_path))[0]
    for suffix in suffixes:
        base_name = base_name.replace(suffix, '')
    if in_build_dir:
        filename = f"{dirname}/{base_name}_pickle.pickle"
    else:
        filename = f"{dirname}/build/{base_name}_pickle.pickle"
    return import_dict(filename)


def lazy_pickle_import_v2(script_path):
    info_names = []
    with open(script_path) as f:
        lines = f.readlines()
    counter = 0
    line = lines[counter]
    while '# [depends]' in line:
        info_names.extend(re.findall(r"(\S+\.pickle)", line))
        counter += 1
        line = lines[counter]
    info = []
    dirname = os.getcwd()
    in_build_dir = ('build' not in os.listdir())
    for info_name in info_names:
        if in_build_dir:
            filename = f"{dirname}/{info_name}"
            info.append(import_dict(filename))
        else:
            filename = f"{dirname}/build/{info_name}"
            info.append(import_dict(filename))
    return info


def lazy_tex(text, script_path):
    dirname = os.path.dirname(script_path)
    filename = os.path.join(
        dirname,
        "build/"
        f"{os.path.basename(script_path).replace('.py', '.tex')}")

    with open(filename, 'w') as texfile:
        for i in range(len(text)):
            texfile.write(text[i])
            texfile.write("\n")


def array_module(xp=None) -> np:
    """
    Find the array module to use, for example **numpy** or **cupy**.
    :param xp: The array module to use, for example, 'numpy'
               (normal CPU-based module) or 'cupy' (GPU-based module).
               If not given, will try to read
               from the ARRAY_MODULE environment variable. If not given and
               ARRAY_MODULE is not set,
               will use numpy. If 'cupy' is requested, will
               try to 'import cupy'. If that import fails, will
               revert to numpy.
    :type xp: optional, string or Python module
    :rtype: Python module
    >>> from lib.utils import array_module
    >>> xp = array_module() # will look at environment variable
    >>> print(xp.zeros(3))
    [0. 0. 0.]
    >>> xp = array_module('cupy') # will try to import 'cupy'
    >>> print(xp.zeros(3))
    [0. 0. 0.]
    """
    xp = xp or os.environ.get("ARRAY_MODULE", "numpy")

    if isinstance(xp, ModuleType):
        return xp

    if xp == "numpy":
        return np

    if xp == "cupy":
        try:
            import cupy as cp

            return cp
        except (ModuleNotFoundError, ImportError) as e:
            global _warn_array_module_once
            if not _warn_array_module_once:
                logging.warning(f"Using numpy. ({e})")
                _warn_array_module_once = True
            return np
    return np
    raise ValueError(f"Don't know ARRAY_MODULE '{xp}'")


def asnumpy(a):
    """
    Given an array created with any array module, return the equivalent
    numpy array. (Returns a numpy array unchanged.)
    >>> from lib.utils import asnumpy, array_module
    >>> xp = array_module('cupy')
    >>> zeros_xp = xp.zeros((3)) # will be cupy if available
    >>> zeros_np = asnumpy(zeros_xp) # will be numpy
    >>> zeros_np
    array([0., 0., 0.])
    """
    if isinstance(a, np.ndarray):
        return a
    return a.get()


def get_array_module(a):
    """
    Given an array, returns the array's
    module, for example, **numpy** or **cupy**.
    Works for numpy even when cupy is not available.
    >>> import numpy as np
    >>> zeros_np = np.zeros((3))
    >>> xp = get_array_module(zeros_np)
    >>> xp.ones((3))
    array([1., 1., 1.])
    """
    submodule = inspect.getmodule(type(a))
    module_name = submodule.__name__.split(".")[0]
    xp = array_module(module_name)
    return xp


def domovie(fig, filename, frame_func, N, axes=None, beautify=False,
            display=True, incremental=True, fps=15, dpi=100, ffmpeg_args=None):
    # Sort out some arguments.
    if axes is None:
        axes = []

    # Create save object object.
    if ffmpeg_args is None:
        ffmpeg_args = []
    ffmpeg = animation.writers["ffmpeg"]
    mp4 = ffmpeg(fps=fps, extra_args=ffmpeg_args)
    mp4.setup(fig, filename, dpi)

    def saveframe():
        """Saves 1 frame to mp4."""
        mp4.grab_frame()

    def finish():
        """Finishes mp4 file creation."""
        mp4.finish()

    # Then make the frames.
    plt.ioff()
    if display:
        print("\n")
    for i in range(N):
        if display:
            print("\rFrame %4d of %d" % (i + 1, N), end="")
            sys.stdout.flush()
        frame_func(i)
        saveframe()
    print("\n")
    plt.ion()
    finish()


def colloc(n_pts, left=False, right=False, plate_gap=1, roots='legendre',
           alpha=0, beta=0, shifted=True):
    roots_to_fcn = {'legendre': roots_legendre,
                    'chebyu': roots_chebyu,
                    'chebyt': roots_chebyt,
                    'jacobi': roots_jacobi,
                    'hermite': roots_hermite}
    if n_pts == 0:
        r = np.empty((1, 1))
        q = np.empty((1, 1))
    elif 'jacobi' in roots:
        r, q = roots_to_fcn[roots](n_pts, alpha, beta)
    else:
        r, q = roots_to_fcn[roots](n_pts)
    r = ((r + 1) / 2).reshape(-1, 1)
    q = (q / 2).reshape(-1, 1)
    if left:
        r = np.vstack((np.zeros((1, 1)),
                        r))
        q = np.vstack((np.zeros((1, 1)),
                        q))
    if right:
        r = np.vstack((r,
                        np.ones((1, 1))))
        q = np.vstack((q,
                        np.zeros((1, 1))))

    roots = r.flatten()
    tvar = cs.SX.sym("t")
    n = len(roots)
    polys = [None] * n
    for j in range(n):
        p = cs.DM(1)
        for i in range(n):
            if i != j:
                p *= (tvar - roots[i]) / (roots[j] - roots[i])
        polys[j] = p
    eval_a_at = roots
    n_i = len(eval_a_at)
    n_j = len(polys)
    a = cs.DM.zeros((n_i, n_j))
    b = cs.DM.zeros((n_i, n_j))
    for j, p in enumerate(polys):
        pder = cs.Function("pder", [tvar], [cs.jacobian(p, tvar)])
        pder2 = cs.Function("pder2", [tvar], [cs.hessian(p, tvar)[0]])
        for i in range(n_i):
            a[i, j] = pder(eval_a_at[i])
            b[i, j] = pder2(eval_a_at[i])
    r = r * plate_gap
    a = np.array(a) / plate_gap
    b = np.array(b) / plate_gap ** 2
    q = q * plate_gap
    if not shifted:
        r -= plate_gap / 2
    return r.flatten(), a, b, q

def dict_to_xz(input_dict, key_names=None):
    x0 = None
    z0 = None
    if key_names is None:
        key_names = list(input_dict.keys())
    for key in key_names:
        val = input_dict[key]
        if x0 is None:
            if len(val.shape) > 1:
                x0 = val[:, 1:-1]
                z0 = val[:, [1, -1]]
            else:
                x0 = val[1:-1]
                z0 = val[[1, -1]]
        else:
            if len(val.shape) > 1:
                x0 = np.hstack((x0, val[:, 1:-1]))
                z0 = np.hstack((z0, val[:, [1, -1]]))
            else:
                x0 = np.hstack((x0, val[1:-1]))
                z0 = np.hstack((z0, val[[1, -1]]))
    return x0, z0


def dict_to_x(input_dict, model_order):
    x0 = None
    if model_order is None:
        key_names = list(input_dict.keys())
    else:
        key_names = ['pr000']
        for i in range(1, model_order):
            key_names.extend([f'pr{i:0=3d}', f'pi{i:0=3d}'])
    for key in key_names:
        val = input_dict[key]
        if x0 is None:
            if len(val.shape) > 1:
                x0 = val[:, :]
            else:
                x0 = val[:]
        else:
            if len(val.shape) > 1:
                x0 = np.hstack((x0, val[:, :]))
            else:
                x0 = np.hstack((x0, val[:]))
    return x0


def get_dir_base(filename):
    dirname = os.path.dirname(filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    return dirname, basename


def generate_interior_points(array1, interior_array):
    diffs = np.diff(array1)
    interior_points = (array1[:-1]
                       + interior_array.reshape(-1, 1)
                       @ diffs.reshape(-1, 1).T).flatten('f')
    return np.hstack(interior_points)


def get_constrained_pts(pts, n_interior):
    interior_pts = colloc(n_interior)[0]
    constrained_pts = generate_interior_points(pts, interior_pts)
    constrained_pts = np.hstack([pts[0],constrained_pts, pts[-1]])
    return constrained_pts
