# [depends] split_and_dance_ex3_gsd.gsd
# [makes] movie
import argparse
from lib import utils
import os
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from lib.utils import color_brewer
import seaborn as sns
from lib.hoomd_utils import gpu_quaternion_to_euler_angle_vectorized2
from lib.utils import get_array_module, array_module, asnumpy
from split_and_dance_ex3_gsd import SetPoint
from types import SimpleNamespace
from scipy.integrate import simpson

xp = array_module('numpy')


def main(make_pdf_, show_):
    dirname, basename = utils.get_dir_base(__file__)
    basename = basename.replace('_movie', '')
    srcname = os.path.join(
        dirname,
        "build/"
        f"{basename}_gsd.gsd")
    frames = gsd.hoomd.open(
        name=srcname, mode='rb')
    sp_obj = SetPoint()

    plt.close('all')
    fig, ax = plt.subplots(ncols=3, figsize=(6, 4), num=2, clear=True)

    sp_color_ind = 4
    color_palette = [color_brewer.rgb_to_hex(e) for e in color_brewer.Set1[6]]
    my_cmap = sns.diverging_palette(145, 300, l=25, s=100, as_cmap=True)
    legend_elements = [
        Line2D([0], [0], color=color_palette[sp_color_ind], label='Target',
               linestyle="solid"),
        Line2D([0], [0], color='blue', label='MPC',
               linestyle="solid"),
        ]
    leg = fig.legend(handles=legend_elements, bbox_to_anchor=(0.666, 0.95),
                     loc="center", ncol=5)

    legend_elements1 = [
        mpatches.Patch(color=my_cmap(1000), label=r'$\rightarrow$'),
        mpatches.Patch(color=my_cmap(-1000), label=r'$\leftarrow$'),
        ]
    leg1 = ax[0].legend(handles=legend_elements1, bbox_to_anchor=(0.5, 1.1),
                        loc="center", ncol=5,
                        columnspacing=0.2)


    line1 = ax[1].plot([], [], color='blue', )[0]
    line2 = ax[2].plot([], [], color='blue', )[0]
    ax[2].axhline(0, color='k', linestyle='dashed')


    time_text = ax[0].text(0.5, 1.05, '', transform=ax[0].transAxes, va='top',
                           ha='center', usetex=False)
    xlim = (-5, 5)
    ax[1].set_xlim(xlim)
    ax[2].set_xlim(xlim)
    ylim = (0, 0.73)
    ax[1].set_ylim(ylim)
    pad = 0.2
    ylim = (-3 - pad, 3 + pad)
    ax[2].set_ylim(ylim)
    ax[1].set_xlabel('$x$')
    ax[2].set_xlabel('$x$')
    ax[1].set_ylabel(r'Number density $n$')
    ax[2].set_ylabel(r'$x$-pointing torque ${\omega}_{x}$')


    ctrl_keys = ['pr000']
    for i in range(1, 5):
        ctrl_keys.extend([f'pr{i:0=3d}', f'pi{i:0=3d}'])
    n_state_pts = 41
    state_pts, state_dz, state_dz2, state_ig = utils.colloc(n_state_pts - 2,
                                                            left=True,
                                                            right=True,
                                                            plate_gap=8,
                                                            shifted=False)
    state_pts_all = state_pts.flatten()
    state_pts = state_pts.flatten()[1:-1]

    r = (frames[0].particles.diameter[0] / 2)

    w, h, _, _, _, _ = frames[0].configuration.box
    ax[0].set_ylim((-w / 2), w / 2)
    ax[0].set_xlim(xlim)

    ax[0].set_xlabel('$x$')
    ax[0].axes.get_yaxis().set_visible(False)

    patches = []
    frame = frames[0]

    pcls = ax[0].scatter([],
                         [],
                         c=[],
                         cmap=my_cmap,
                         s=4)
    bd_vl1 = ax[0].axvline(0, color=color_palette[sp_color_ind], label='Target',
                           linestyle='dotted')
    bd_vl2 = ax[0].axvline(0, color=color_palette[sp_color_ind], label='Target',
                           linestyle='dotted')
    field_vl1 = ax[1].axvline(0, ymin=0, ymax=0.88,
                              color=color_palette[sp_color_ind], label='Target',
                              linestyle='dotted')
    field_vl2 = ax[1].axvline(0, ymin=0, ymax=0.88,
                              color=color_palette[sp_color_ind], label='Target',
                              linestyle='dotted')
    i1_text = ax[1].text(0.25, 0.93, 'hi',
                         transform=ax[1].transAxes,
                         va='top', color='blue',
                         ha='center', usetex=False)
    i2_text = ax[1].text(0.75, 0.93, 'hi2', transform=ax[1].transAxes, va='top',
                         color='blue', ha='center', usetex=False)
    i1_targ = ax[1].text(0.25, 0.99, '',
                         transform=ax[1].transAxes,
                         va='top', color=color_palette[sp_color_ind],
                         ha='center', usetex=False)
    i2_targ = ax[1].text(0.75, 0.99, '', transform=ax[1].transAxes, va='top',
                         color=color_palette[sp_color_ind], ha='center',
                         usetex=False)

    def sp_fcn(time):
        control_area = 6
        wr001_max = 5
        max_speed = (1 / xp.tanh(wr001_max) - 1 / wr001_max)
        oscillations_start = 15
        scaled_speed = 0.9 * max_speed
        period = control_area / 2 / scaled_speed
        n_periods = 1.25
        oscillations_stop = period * 2 * xp.pi * n_periods + oscillations_start
        if oscillations_start < time < oscillations_stop:
            rel_t = time - oscillations_start - period * xp.pi / 2
            pos_sp = control_area / 2 * xp.sin(rel_t / period)
        else:
            pos_sp = 0
        return pos_sp

    def animate(i):
        frame = frames[:][i]
        xdata1 = frame.log['env_zpts']
        ydata1 = frame.log['pr000']
        udata1 = xp.polyval(frame.log['wr001_coeffs'], xdata1)
        lines = []
        line1.set_data(xdata1, ydata1)
        line2.set_data(xdata1, udata1)

        lines.append(line1)
        lines.append(line2)

        pcls.set_offsets([[e[1], e[0]] for e in frame.particles.position])
        pcls.set_array(asnumpy(xp.sin(gpu_quaternion_to_euler_angle_vectorized2(
            xp.array(frame.particles.orientation[:, 0]),
            xp.array(frame.particles.orientation[:, 1]),
            xp.array(frame.particles.orientation[:, 2]),
            xp.array(frame.particles.orientation[:, 3]), xp)[
                                          -1] * xp.pi / 180)))
        lines.append(pcls)

        time = (frame.configuration.step * frame.log['dt'][0]
                - frames[0].configuration.step * frame.log['dt'][0])
        sps = sp_obj.sp_fcn(time)
        bd_vl1.set_xdata([sps[0],
                          sps[0]])
        bd_vl2.set_xdata([sps[1],
                          sps[1]])
        field_vl1.set_xdata([sps[0],
                             sps[0]])
        field_vl2.set_xdata([sps[1],
                             sps[1]])

        l_half_inds = xp.argwhere(xdata1 <= 0)
        r_half_inds = xp.argwhere(xdata1 >= 0)
        li = simpson(ydata1[l_half_inds].flatten(),
                     xdata1[l_half_inds].flatten())
        ri = simpson(ydata1[r_half_inds].flatten(),
                     xdata1[r_half_inds].flatten())

        i1_text.set_text(r"{0:2.0f}%".format(100 * li))
        i2_text.set_text(r"{0:2.0f}%".format(100 * ri))

        i1_targ.set_text(r"{0:2.0f}%".format(100 * sps[3] / (sps[2] + sps[3])))
        i2_targ.set_text(r"{0:2.0f}%".format(100 * sps[2] / (sps[2] + sps[3])))
        time_text.set_text("t={0:4.1f}$\\tau_R$".format(time))
        return lines

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)

    if show_:
            
            plt.ion()
            anim = FuncAnimation(fig, animate,
                                frames=len(frames[::1]) - 2, interval=50, blit=False)
            return anim
    else:
        if make_pdf_:
            animate(0)
            utils.lazy_pdf([fig], __file__, dpi=200)
        else:
            base_name = f"{os.path.basename(__file__).replace('.py', '.mp4')}"
            utils.domovie(fig, base_name,
                        animate,
                        len(frames[::1]) - 2, ax,
                        incremental=False,
                        fps=10, dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("make_pdf", help="make a pdf or not", default=0)
    parser.add_argument("show", help="show animation", default=0, nargs='?')
    options = parser.parse_args()
    make_pdf = bool(int(options.make_pdf))
    show = bool(int(options.show))
    ani = main(make_pdf_=make_pdf, show_=show)
    if show:
        plt.show()