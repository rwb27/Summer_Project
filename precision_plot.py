
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    print ("Loading data...")

    df = h5py.File("precision.hdf5", mode = "r")
    with PdfPages("precision_all.pdf") as pdf:
        for group in df.values():
            dset = group.values()[-1]

            matplotlib.rcParams.update({'font.size': 8})

            t = dset[:, 0]
            x = dset[:, 1] * 2.16
            y = dset[:, 2] * 2.16

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
            
            ax[1].plot(x, y, ".-")

            ax[0].set_xlabel('Time [$\mathrm{s}$]')
            ax[0].set_ylabel('X Position [$\mathrm{\mu m}$]')
            ax2.set_ylabel('Y Position [$\mathrm{\mu m}$]')
            ax[1].set_xlabel('X Position [$\mathrm{\mu m}$]')
            ax[1].set_ylabel('Y Position [$\mathrm{\mu m}$]')
            ax[1].set_aspect(1)
            
            fig.suptitle(group.name)

            plt.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)
