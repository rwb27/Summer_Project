
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib

if __name__ == "__main__":
    print ("Loading data...")

    df = h5py.File("precision.hdf5", mode = "r")
    group = df.values()[-1]
    dset = group.values()[-1]

    matplotlib.rcParams.update({'font.size': 8})

    t = dset[:, 0]
    x = dset[:, 1] * 2.16
    y = dset[:, 2] * 2.16

    fig, ax = plt.subplots(1, 2)

    ax[0].plot(t, x, "r-")
    ax2 = ax[0].twinx()
    ax2.plot(t, y, "b-")
    ax[1].plot(x, y, ".-")

    ax[0].set_xlabel('Time [$\mathrm{s}$]')
    ax[0].set_ylabel('X Position [$\mathrm{\mu m}$]')
    ax2.set_ylabel('Y Position [$\mathrm{\mu m}$]')
    ax[1].set_xlabel('X Position [$\mathrm{\mu m}$]')
    ax[1].set_ylabel('Y Position [$\mathrm{\mu m}$]')
    ax[1].set_aspect(1)

    plt.tight_layout()
    plt.savefig("precision.pdf", bbox_inches='tight', dpi=180)

    plt.show()
