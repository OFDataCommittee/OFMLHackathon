import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
def animate_reconstruction(reconstruction, title, n_frames):
    vmin, vmax = reconstruction.min(), reconstruction.max()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=640)
    plt.subplots_adjust(bottom=0.2, top=0.85, left=0.1, right=0.95)
    def animate(i):
        print("\r", f"frame {i:03d}", end="")
        ax.clear()
        tri = ax.tricontourf(x, y, reconstruction[:, i], levels=60, cmap="seismic", vmin=vmin*1.0/0.98, vmax=vmax*0.98)
        add_stl_patch(ax, scale=1.0/CHORD)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/c$")
        ax.set_ylabel(r"$y/c$")
        ax.set_title(title)
        ax.set_xlim(-0.2, 2.0)
        ax.set_ylim(-0.3, 1.0)
    return FuncAnimation(fig, animate,  frames=n_frames, repeat=True)
writer = FFMpegWriter(fps=15, bitrate=1800)
anim = animate_reconstruction(time_series, r"$\bar{{f}}\approx{:2.2f}$, $a$".format(f_mean), n_frames)
plt.close()
anim.save(f"flow.mp4", writer=writer)