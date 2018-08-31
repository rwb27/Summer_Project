
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


def plot_txy(dset, microns_per_pixel=2.16):
    matplotlib.rcParams.update({'font.size': 8})

    t = dset[:, 0]
    x = dset[:, 1] * microns_per_pixel
    y = dset[:, 2] * microns_per_pixel

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, x, "r-")
    ax2 = ax[0].twinx()
    ax2.plot(t, y, "b-")
    
    # Make the scale the same for X and Y (so it's obvious which is moving)
    xmin, xmax = ax[0].get_ylim()
    ymin, ymax = ax2.get_ylim()
    r = max(xmax - xmin, ymax - ymin)
    ax[0].set_ylim((xmax + xmin)/2 - r/2, (xmax + xmin)/2 + r/2)
    ax2.set_ylim((ymax + ymin)/2 - r/2, (ymax + ymin)/2 + r/2)
    
    # plot the XY motion, make the limits equal (because it looks nice)
    ax[1].plot(x, y, ".-")
    ax[1].set_aspect(1)
    xmin, xmax = ax[1].get_xlim()
    ymin, ymax = ax[1].get_ylim()
    r = max(xmax - xmin, ymax - ymin)
    ax[1].set_xlim((xmax + xmin)/2 - r/2, (xmax + xmin)/2 + r/2)
    ax[1].set_ylim((ymax + ymin)/2 - r/2, (ymax + ymin)/2 + r/2)

    ax[0].set_xlabel('Time [$\mathrm{s}$]')
    ax[0].set_ylabel('X Position [$\mathrm{\mu m}$]')
    ax2.set_ylabel('Y Position [$\mathrm{\mu m}$]')
    ax[1].set_xlabel('X Position [$\mathrm{\mu m}$]')
    ax[1].set_ylabel('Y Position [$\mathrm{\mu m}$]')

    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    df = h5py.File("linear_motion.h5", mode = "r")
    experiment = df.values()[-1]
    with PdfPages("linear_all.pdf") as pdf:
        for group in experiment.values():
            if "sequence" in group.name:
                fig, ax = plot_txy(group['camera_motion'])
                pdf.savefig(fig)
                plt.close(fig)
    df.close()
