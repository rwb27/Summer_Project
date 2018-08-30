
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib

if __name__ == "__main__":
    print ("Loading data...")

    df = h5py.File("repeat.hdf5", mode = "r")
    group = df.values()[-1]
    n = len(group)

    dist = np.zeros(n)
    mean_error = np.zeros(n)
    for i in range(n):
        dset = group["distance%03d" % i] #distances
        m = len(dset)
        diff = np.zeros([m, 2])
        move = np.zeros([m, 3])
        for j in range(m):
            data = dset["move%03d" % j] #moves
            init_c = data["init_cam_position"]
            final_c = data["final_cam_position"]
            init_s = data["init_stage_position"]
            moved_s = data["moved_stage_position"]
            diff[j, 0] = final_c[0, 1] - init_c[0, 1]
            diff[j, 1] = final_c[0, 2] - init_c[0, 2]
            move[j, :] = moved_s[:] - init_s[:]
        abs_move = np.sqrt(np.sum(move**2, axis = 0))
        error = np.sqrt(np.sum(diff**2, axis = 0))
        dist[i] = np.mean(abs_move, axis = 0)
        mean_error[i] = np.mean(error, axis = 0)

        fig, ax = plt.subplots(1, 1)
        ax.plot(diff[:, 0] * 2.16, diff[:, 1] * 2.16, "+")
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        plt.xlabel('X Position [$\mathrm{\mu m}$]', horizontalalignment = 'right', x = 1.0)
        plt.ylabel('Y Position [$\mathrm{\mu m}$]', horizontalalignment = 'right', y = 1.0)
        
        plt.savefig("repeat_%d.pdf" % i, bbox_inches='tight', dpi=180)

    fig2, ax2 = plt.subplots(1, 1)

    ax2.semilogx(dist[:] * 0.01, mean_error[:] * 2.16, "r-")

    ax2.set_xlabel('Move Distance [$\mathrm{\mu m}$]')
    ax2.set_ylabel('Error [$\mathrm{\mu m}$]')
    plt.savefig("repeatability.pdf", bbox_inches='tight', dpi=180)

    plt.show()
