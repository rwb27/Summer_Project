
from __future__ import print_function
import io
import sys
import os
import time
import argparse
import numpy as np
import picamera
from builtins import input
from readchar import readchar, readkey
from openflexure_stage import OpenFlexureStage
from openflexure_microscope import load_microscope
from openflexure_microscope.microscope import picamera_supports_lens_shading
import scipy
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from contextlib import contextmanager, closing
import data_file
import cv2
from camera_stuff import find_template
#import h5py
import threading
import queue
import random

def measure_txy(start_t, ms, templ8):
    data = np.zeros((1, 3))
    data[0, 0] = time.time() - start_t
    frame = ms.rgb_image().astype(np.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data[0, 1:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
    return data

def random_point(move_dist):
    angle = random.randrange(0, 360) * np.pi / 180
    vector = np.array([move_dist * np.cos(angle), 0, move_dist * np.sin(angle)])
    vector = np.rint(vector)
    return vector

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz", dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = "repeat.hdf5")) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        backlash = 256

        camera.resolution=(640,480)
        stage.backlash = backlash

        n_moves = 10

        camera.start_preview(resolution=(640,480))
        
        initial_stage_position = stage.position

        stage.move_rel([-backlash, -backlash, -backlash])
        stage.move_rel([backlash, backlash, backlash])
        
        experiment_group = df.new_group("repeatability", "Repeatability measurements for different distances in each group")

        for dist in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            data_gr = df.new_group("distance", "Repeatability measurements, moving the stage away and back again", parent=experiment_group)
            data_gr.attrs['move_distance'] = dist

            image = ms.rgb_image().astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean = np.mean(image)
            templ8 = (image - mean)[100:-100, 100:-100]

            start_t = time.time()

            for j in range(n_moves):
                move_group = df.new_group("move", "One move away and back again", parent = data_gr)
                move_group['init_stage_position'] = stage.position
                move_group['init_cam_position'] = measure_txy(start_t, ms, templ8)
                move_vect = random_point(dist)
                stage.move_rel(move_vect)
                time.sleep(1)
                move_group['moved_stage_position'] = stage.position
                move_group['moved_cam_position'] = measure_txy(start_t, ms, templ8)
                stage.move_rel(np.negative(move_vect))
                time.sleep(1)
                move_group['final_stage_position'] = stage.position
                move_group['final_cam_position'] = measure_txy(start_t, ms, templ8)
                stage.move_abs(initial_stage_position)

        camera.stop_preview()
        print("Done")
