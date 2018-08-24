
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy.linalg
import matplotlib

if __name__ == "__main__":
    print ("Loading data...")

    df = h5py.File("orthogonality.hdf5", mode = "r")
    group1 = df["data_stage000"]
    group2 = df["data_cam000"]
    group3 = df["data_steps000"]
    group4 = df["data_distance000"]
    dset1 = group1["data_stage00000"]
    dset2 = group2["data_cam00000"]
    dset3 = group3["data_steps00000"]
    dset4 = group4["data_distance00000"]

    n = len(dset1)

    pixel_shifts = np.zeros([n, 2])
    process = np.zeros([n, 3])
    location_shifts = np.zeros([n, 2])

    for i in range(n):
        pixel_shifts[i, :] = dset2[i, :] - np.mean(dset2, axis = 0) #avoids translation
        process[i, :] = dset1[i, :] - np.mean(dset1, axis = 0)
        location_shifts[i, 0] = process[i, 0]
        location_shifts[i, 1] = process[i, 2]

    A, res, rank, s = np.linalg.lstsq(pixel_shifts, location_shifts)
    #A is the least squares solution pixcel_shifts*A = location_shifts
    #res is the sums of residuals location_shifts - pixcel_shifts*A
    #rank is rank of matrix pixcel_shifts
    #s is singular values of pixcel_shifts
    print(A)

    #unit vectors
    x = np.array([1, 0]) 
    y = np.array([0, 1])

    #dot products of A with x and y unit vectors to find x and y components of A
    A_x = np.dot(A, x)
    A_y = np.dot(A, y)

    #uses standard dot product formula to find angle between A_x and A_y
    dotproduct = np.dot(A_x, A_y)
    cosa = dotproduct / (np.linalg.norm(A_x) * np.linalg.norm(A_y))
    angle = np.arccos(cosa)
    angle = angle * 180 / np.pi
    print (angle)

    matplotlib.rcParams.update({'font.size': 8})

    fig1, ax1 = plt.subplots(1, 1)

    ax1.plot(location_shifts[:, 0], location_shifts[:, 1], "+")
    estimated_positions = np.dot(pixel_shifts, A)
    #dot product pixcel_shifts . A
    ax1.plot(estimated_positions[:, 0], estimated_positions[:, 1], "o")
    ax1.set_aspect(1)

    ax1.set_xlabel('X Position [$\mathrm{\mu m}$]')
    ax1.set_ylabel('Y Position [$\mathrm{\mu m}$]')
    plt.savefig("orthogonality.pdf", bbox_inches='tight', dpi=180)

    fig2, ax2 = plt.subplots(1, 1)
    ax3 = ax2.twinx()

    ax2.plot(dset3[0:9], dset4[0:9] / dset3[0:9] * 2.16, "r-")
    ax3.plot(dset3[9:18], dset4[9:18] / dset3[9:18] * 2.16, "b-")

    ax2.set_xlabel('Steps Moved')
    ax2.set_ylabel('Step Size in X [$\mathrm{\mu m}$]')
    ax3.set_ylabel('Step Size in Y [$\mathrm{\mu m}$]')
    plt.savefig("step_size.pdf", bbox_inches='tight', dpi=180)

    plt.show()
