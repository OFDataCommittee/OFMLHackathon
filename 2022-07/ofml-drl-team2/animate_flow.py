# installation of flowtorch:
# pip3 install git+https://github.com/FlowModelingControl/flowtorch.git@aweiner

from os.path import join
from flowtorch.data import FOAMDataloader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse

def parseArguments():

    ag = argparse.ArgumentParser()
    ag.add_argument('directory',
                    help='Case directory of OF results.')
    ag.add_argument('-o','--output', required=False, default='magU_controlled.mp4',
                    help='Filename of video file.')
    ag.add_argument('-v', '--value', required=False, default='U',
                    help='Value that is displayed in the video.')
    args = ag.parse_args()
    return args

def main(args):

    # path to OpenFOAM simulation
    of_simulation = args.directory
    # for reconstructed fields, set distributed=False
    loader = FOAMDataloader(of_simulation, distributed=True)
    # get available write times, skip 0
    times = loader.write_times[1:]
    # vertices
    vertices = loader.vertices
    x = vertices[:, 0]
    y = vertices[:, 1]
    # load U - shape: n_points x 3 x n_times
    value = args.value
    U = loader.load_snapshot(value, times)  
    # animate mag(U) over time
    def animate_mag(U, title, n_frames):
        magU = U.norm(dim=1)
        vmin, vmax = magU.min(), magU.max()
        fig, ax = plt.subplots(figsize=(7, 4), dpi=640)
        plt.subplots_adjust(bottom=0.2, top=0.85, left=0.1, right=0.95)
        def animate(i):
            print("\r", f"frame {i:03d}", end="")
            ax.clear()
            tri = ax.tricontourf(x, y, magU[:, i], levels=60, cmap="seismic", vmin=vmin*1.0/0.98, vmax=vmax*0.98)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(title)
            ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color='k'))
        return FuncAnimation(fig, animate,  frames=n_frames, repeat=True)


    # gives a nice animation for snapshots written every 0.05s
    writer = FFMpegWriter(fps=15, bitrate=1800)
    anim = animate_mag(U, r"$||\mathbf{u}||$", U.shape[-1])
    plt.close()
    anim.save(args.output, writer=writer)

if __name__ == '__main__':
    args = parseArguments()
    main(args)
