# [depends] heur_sol_ex1_gsd.gsd
# [makes] movie
import argparse
from lib import utils
import os
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.utils import color_brewer
import seaborn as sns
from sp_chase_ex2_gsd import SetPoint
import matplotlib.patches as mpatches
from lib.hoomd_utils import gpu_quaternion_to_euler_angle_vectorized2
from lib.utils import get_array_module, array_module, asnumpy
from matplotlib.animation import FuncAnimation, writers
import numpy as np
xp = array_module(np)


def main(make_pdf_, show_):
    dirname, basename = utils.get_dir_base(__file__)
    basename = basename.replace('_movie', '')
    srcname = os.path.join(
        dirname,
        f"build/heur_sol_ex1_gsd.gsd"
    )
    frames = gsd.hoomd.open(
        name=srcname, mode='rb')[::10][:510]
    plt.close('all')
    fig, ax = plt.subplots(ncols=3, figsize=(6, 4), num=2, clear=True)
    plate_gap = 10
    sp_obj = SetPoint()
    sp_fcn = sp_obj.sp_fcn

    sp_color_ind = 4
    color_palette = [color_brewer.rgb_to_hex(e) for e in color_brewer.Set1[6]]
    my_cmap = sns.diverging_palette(145, 300, l=25, s=100, as_cmap=True)
    legend_elements1 = [mpatches.Patch(color=my_cmap(1000),
                                       label=r'$\rightarrow$'),
                        mpatches.Patch(color=my_cmap(-1000),
                                       label=r'$\leftarrow$'),
                        ]
    leg1 = ax[0].legend(handles=legend_elements1, bbox_to_anchor=(0.5, 1.1),
                        loc="center", ncol=5,
                        columnspacing=0.2)
    legend_elements = [Line2D([0], [0], color=color_palette[sp_color_ind],
                              label='Target', linestyle="solid"),
                       Line2D([0], [0], color='blue',
                              label='HEU', linestyle="solid"),
                       ]
    leg = fig.legend(handles=legend_elements, bbox_to_anchor=(0.66666, 0.95)
                     , loc="center", ncol=5)

    bd_vl = ax[0].axvline(0, color=color_palette[sp_color_ind],
                          label='Target')
    field_vl = ax[1].axvline(0, color=color_palette[sp_color_ind],
                             label='Target')

    line1 = ax[1].plot([], [], color='blue', )[0]
    line2 = ax[2].plot([], [], color='blue', )[0]
    ax[2].axhline(0, color='k', linestyle='dashed')

    time_text = ax[0].text(0.5, 1.05, '', transform=ax[0].transAxes,
                           va='top', ha='center', usetex=False)
    
    ax[0].text(0., 1.05, r'$\mathbf{(a)}$', transform=ax[0].transAxes,
               va='top', ha='center', usetex=False)
    ax[1].text(0., 1.05, r'$\mathbf{(b)}$', transform=ax[1].transAxes,
               va='top', ha='center', usetex=False)
    ax[2].text(0., 1.05, r'$\mathbf{(c)}$', transform=ax[2].transAxes,
                           va='top', ha='center', usetex=False)

    xlim = (-plate_gap/2, plate_gap/2)
    ax[1].set_xlim(xlim)
    ax[2].set_xlim(xlim)
    ylim = (0, 0.85)
    ax[1].set_ylim(ylim)
    pad = 0.2
    ylim = (-5 - pad, 5 + pad)
    ax[2].set_ylim(ylim)
    ax[1].set_xlabel('$x$')
    ax[2].set_xlabel('$x$')
    ax[1].set_ylabel(r'Number Density $n$')
    ax[2].set_ylabel(r'$x$-torque ${\omega}_x$')
    # hi = utils.polinterp(sim_info['z'].flatten()[1:-1], frames[0].log['env_zpts'])

    ctrl_keys = ['pr000']
    for i in range(1, 5):
        ctrl_keys.extend([f'pr{i:0=3d}', f'pi{i:0=3d}'])
    n_state_pts = 41
    state_pts, state_dz, state_dz2, state_ig = utils.colloc(
        n_state_pts - 2,
        left=True,
        right=True,
        plate_gap=plate_gap,
        shifted=False)
    state_pts_all = state_pts.flatten()
    state_pts = state_pts.flatten()[1:-1]

    r = (frames[0].particles.diameter[0] / 2)
    # x_sp_correction = frames[0].log['x_sp'][0]
    w, h, _, _, _, _ = frames[0].configuration.box
    ax[0].set_ylim((-w / 2), w / 2)
    ax[0].set_xlim(-plate_gap/2, plate_gap/2)
    ax[0].set_xlabel('$x$')
    ax[0].axes.get_yaxis().set_visible(False)


    patches = []
    frame = frames[0]

    pcls = ax[0].scatter([],
                         [],
                         c=[],
                         cmap=my_cmap,
                         s=4)

    def animate(i):
        frame = frames[:][i]
        xdata1 = frame.log['env_zpts']
        ydata1 = frame.log['pr000']
        time = (frame.configuration.step * frame.log['dt'][0]
                - frames[0].configuration.step * frame.log['dt'][0])
        sp = sp_fcn(time)
        udata1 = data1 = np.where(xdata1 < sp,
                                  sp_obj.w001_max,
                                  np.where(xdata1 > sp,
                                           -    sp_obj.w001_max, 0)) 
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
        
        bd_vl.set_xdata([sp,
                         sp])
        lines.append(bd_vl)
        field_vl.set_xdata([sp,
                            sp])
        lines.append(field_vl)

        time_text.set_text("$t=${0:4.1f}$\\tau_R$".format(time))
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
                          fps=20, dpi=200)


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
