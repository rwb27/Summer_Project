from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py

def plot_precision_dataset(dset):
    """Given a dataset of t,x,y points, plot x vs t, y vs t, y vs x."""
    f, ax = plt.subplots(1,3)
    ax[0].plot(dset[:,0], dset[:,1])
    ax[1].plot(dset[:,0], dset[:,2])
    ax[2].plot(dset[:,1], dset[:,2],'.-')
    ax[2].set_aspect(1)
    return f

if __name__ == "__main__":
    df = h5py.File("precision.hdf5", mode="r")
    print("Groups in the datafile: {}".format(df.keys()))
    group = df.values()[-1]
    print("Datasets in the datafile: {}".format(group.keys()))
    dset = group.values()[-1]
    print("Plotting dataset {}".format(dset.name))
    
    plot_precision_dataset(dset)
    plt.show()
    