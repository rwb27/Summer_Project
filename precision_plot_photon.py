
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    print ("Loading data...")

    # Load the data from the HDF5 file
    df = h5py.File("precision.hdf5", mode = "r")
    group = df['data015']
    dset = group.values()[-1]
    rr = slice(0,500)
    t = dset[rr, 0]
    x = dset[rr, 1] * 2.16 * 1000
    x -= np.mean(x)
    y = dset[rr, 2] * 2.16 * 1000
    y -= np.mean(y)

    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 2, figsize=(6,3))

    ax[0].plot(t, x, "r-")
    ax[0].plot(t, y, "b-")
    
    ax[1].plot(x, y, ".-")

    ax[0].set_xlabel('Time [$\mathrm{s}$]')
    ax[0].set_ylabel('Displacement [$\mathrm{nm}$]')
    ax[1].set_xlabel('X Position [$\mathrm{nm}$]')
    ax[1].set_ylabel('Y Position [$\mathrm{nm}$]')
    ax[1].set_aspect(1)
    
    plt.tight_layout()

    fig.savefig("{}_photon.pdf".format(group.name[1:]))
    plt.close(fig)
    df.close()
