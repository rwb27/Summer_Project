
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy.interpolate import interp1d


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
    
def compare_txy(camera_txy, stage_txy, microns_per_pixel=2.16):
    matplotlib.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots(1, 2)
    for i, direction in enumerate(['X', 'Y']):
        stage_t = stage_txy[:,0]
        stage_p = stage_txy[:,i+1]
        stage_pos = interp1d(stage_t, stage_p, 
                             kind="linear", bounds_error=False, 
                             fill_value=(stage_p[0], stage_p[-1]))
                             
        camera_t = camera_txy[:,0]
        camera_p = camera_txy[:,i+1]
        ax[i].plot(stage_pos(camera_t), camera_p)
        ax[i].set_xlabel("Stage {}/steps".format(direction))
        ax[i].set_ylabel("Camera {}/microns".format(direction))
        coeffs = np.polyfit(stage_pos(camera_t), camera_p, 1)
        trendline = np.poly1d(coeffs)
        ax[i].plot(stage_p, trendline(stage_p))
        ax[i].text(0, 0,'gradient {:.03}um/step'.format(coeffs[0]), 
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax[i].transAxes)
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    df = h5py.File("linear_motion.h5", mode = "r")
    pixels_per_micron = 2.16
    for experiment in df.values():
        pdf_filename = "linear_all{}.pdf".format(experiment.name.replace('/','_'))
        try:
            with PdfPages(pdf_filename) as pdf:
                for group in experiment.values():
                    if "sequence" in group.name:
                        fig, ax = plot_txy(group['camera_motion'], pixels_per_micron)
                        pdf.savefig(fig)
                        plt.close(fig)
                        fig, ax = compare_txy(group['camera_motion'], group['stage_moves'], pixels_per_micron)
                        pdf.savefig(fig)
                        plt.close(fig)
        except KeyError:
            print("giving up on {}, probably old-format data.".format(experiment.name))
            if os.path.exists(pdf_filename):
                print("removing failed PDF of plots")
                os.remove(pdf_filename)
    df.close()
