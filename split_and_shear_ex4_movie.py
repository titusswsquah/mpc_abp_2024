# [depends] split_and_shear_ex4_gsd.gsd
# [makes] movie
import argparse
from lib import utils
import os
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.utils import color_brewer
import seaborn as sns
from split_and_shear_ex4_gsd import SetPoint
import matplotlib.patches as mpatches
from lib.hoomd_utils import gpu_quaternion_to_euler_angle_vectorized2
from lib.utils import get_array_module, array_module, asnumpy
from matplotlib.animation import FuncAnimation

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
    sp_obj = SetPoint(10)

    plt.close('all')

    fig = plt.figure(figsize=(6, 4), num=2, clear=True)
    gs = fig.add_gridspec(2, 3)

    ax = [fig.add_subplot(gs[0:, 0:1]),
          fig.add_subplot(gs[0:1, 1:2]),
          fig.add_subplot(gs[0:1, 2:3]),
          fig.add_subplot(gs[1:2, 1:2]),
          fig.add_subplot(gs[1:2, 2:3]),
          ]

    mpc_color_palette = [color_brewer.rgb_to_hex(e) for e in
                         color_brewer.Blues[3]]
    heu_color_palette = [color_brewer.rgb_to_hex(e) for e in
                         color_brewer.Oranges[3]]
    sp_color_ind = 4
    color_palette = [color_brewer.rgb_to_hex(e) for e in color_brewer.Set1[6]]
    my_cmap = sns.diverging_palette(250, 30, l=45, s=100, as_cmap=True)
    legend_elements1 = [
        mpatches.Patch(color=my_cmap(-1000), label=r'$\downarrow$'),
        mpatches.Patch(color=my_cmap(1000), label=r'$\uparrow$'), ]
    leg1 = ax[0].legend(handles=legend_elements1, bbox_to_anchor=(0.5, 1.1),
                        loc="center", ncol=5,
                        columnspacing=0.2)
    legend_elements = [
        Line2D([0], [0], color=color_palette[sp_color_ind], label='Target',
               linestyle="dotted"),
        Line2D([0], [0], color=mpc_color_palette[2], label='MPC',
               linestyle="solid"),
        ]
    leg = fig.legend(handles=legend_elements, bbox_to_anchor=(0.6666, 0.95),
                     loc="center", ncol=5)


    bd_p0_line = ax[1].plot([], [], color=mpc_color_palette[2], )[0]
    bd_p1x_line = ax[2].plot([], [], color=mpc_color_palette[2], )[0]
    bd_w1z_line = ax[3].plot([], [], color=mpc_color_palette[2], )[0]
    bd_w1x_line = ax[4].plot([], [], color=mpc_color_palette[2], )[0]

    fi_p0_line = \
    ax[1].plot([], [], color=mpc_color_palette[1], linestyle='dashed')[0]
    fi_p1x_line = \
    ax[2].plot([], [], color=mpc_color_palette[1], linestyle='dashed')[0]
    fi_w1z_line = \
    ax[3].plot([], [], color=mpc_color_palette[1], linestyle='dashed')[0]
    fi_w1x_line = \
    ax[4].plot([], [], color=mpc_color_palette[1], linestyle='dashed')[0]

    time_text = ax[0].text(0.5, 1.05, '', transform=ax[0].transAxes, va='top',
                           ha='center', usetex=False)
    xlim = (-5, 5)
    for i in range(1, 5):
        ax[i].set_xlim(xlim)
    ax[1].set_ylim((0, 0.73))
    ax[2].set_ylim((-0.5, 0.5))
    pad = 0.2
    ylim = (-3 - pad, 3 + pad)
    ax[3].set_ylim(ylim)
    ax[4].set_ylim(ylim)
    ax[3].set_xlabel('$x$')
    ax[4].set_xlabel('$x$')
    ax[1].sharex(ax[3])
    ax[2].sharex(ax[4])
    ax[1].set_ylabel(r'Number density $n$')
    ax[2].set_ylabel(r'$y$-velocity $m_y/n$')
    ax[3].set_ylabel(r'$x$-torque $\omega_x$')
    ax[4].set_ylabel(r'$y$-torque $\omega_y$')

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
    field_vl1 = ax[1].axvline(0, color=color_palette[sp_color_ind],
                              label='Target', linestyle='dotted')
    field_vl2 = ax[1].axvline(0, color=color_palette[sp_color_ind],
                              label='Target', linestyle='dotted')
    field_sp3 = \
    ax[2].plot([], [], color=color_palette[sp_color_ind], label='Target',
               linestyle='dotted')[0]
    def animate(i):
        frame = frames[:][i]
        xdata1 = frame.log['env_zpts']
        ydata1 = frame.log['pr000']
        udata1 = xp.polyval(frame.log['wr001_coeffs'], xdata1)
        lines = []
        bd_p0_line.set_data(xdata1, ydata1)
        bd_p1x_line.set_data(xdata1, frame.log['pr001'] / frame.log['pr000'])
        bd_w1z_line.set_data(xdata1, udata1)
        bd_w1x_line.set_data(xdata1,
                             -xp.polyval(frame.log['wi001_coeffs'], xdata1))

        lines.append(bd_p0_line)
        lines.append(bd_p1x_line)
        lines.append(bd_w1z_line)
        lines.append(bd_w1x_line)
        lines.append(fi_p0_line)
        lines.append(fi_p1x_line)
        lines.append(fi_w1z_line)
        lines.append(fi_w1x_line)

        pcls.set_offsets([[e[1], e[0]] for e in frame.particles.position])
        pcls.set_array(asnumpy(xp.cos(gpu_quaternion_to_euler_angle_vectorized2(
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
        field_sp3.set_data(xdata1, sps[2] * sp_obj.desired_mx_fcn(xdata1))
        lines.append(field_sp3)

        time_text.set_text("t={0:4.1f}$\\tau_R$".format(time))
        return lines

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
        
    if show_:
        plt.ion()
        anim = FuncAnimation(fig, animate,
                             frames=len(frames[::1]), interval=50, blit=False)
        return anim
    else:
        if make_pdf_:
            animate(0)
            utils.lazy_pdf([fig], __file__, dpi=200)
        else:
            base_name = f"{os.path.basename(__file__).replace('.py', '.mp4')}"
            utils.domovie(fig, base_name,
                          animate,
                          len(frames[::1]), ax,
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
