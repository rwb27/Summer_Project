
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib

def printProgressBar(iteration, total, length = 10):
    percent = 100.0 * iteration / total
    filledLength = int(length * iteration // total)
    bar = '*' * filledLength + '-' * (length - filledLength)
    print('Progress: |%s| %d%% Completed' % (bar, percent), end = '\r')
    if iteration == total: 
        print()

if __name__ == "__main__":
    print ("Loading data...")

    N_frames = 500
    #need to be consistant between drift.py and drift_plot.py

    df = h5py.File("drift.hdf5", mode = "r")
    group = df.values()[-1]
    data = np.zeros([N_frames * len(group), 3])
    for i in range(len(group)):
        dset = group["data%05d" % i]
        for j in range(N_frames):
            data[N_frames * i + j, :] = dset[j, :]
        printProgressBar(i, len(group))
    print("")

    matplotlib.rcParams.update({'font.size': 8})

    t = data[:, 0]
    x = data[:, 1] * 2.16
    y = data[:, 2] * 2.16

    fig, ax = plt.subplots(1, 1)

    ax.plot(t, x, "r-")
    ax2 = ax.twinx()
    ax2.plot(t, y, "b-")

    ax.set_xlabel('Time [$\mathrm{s}$]')
    ax.set_ylabel('X Position [$\mathrm{\mu m}$]')
    ax2.set_ylabel('Y Position [$\mathrm{\mu m}$]')
    plt.savefig("drift.pdf", bbox_inches='tight', dpi=180)

    plt.show()
