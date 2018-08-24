
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

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz", dummy_stage = False) as ms, \
         closing(data_file.Datafile(filename = "orthogonality.hdf5")) as df:
        
        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        side_length = 10000
        points = 10
        backlash = 256

        camera.resolution=(640,480)
        stage.backlash = backlash

        stage_pos = df.new_group("data_stage", "orthogonality")
        cam_pos = df.new_group("data_cam", "orthogonality")
        steps = df.new_group("data_steps", "step_size")
        distance = df.new_group("data_distance", "step_size")
        data_stage = np.zeros((2 * points, 3))
        data_cam = np.zeros((2 * points, 2))
        data_steps = np.zeros(2 * (points - 1))
        data_distance = np.zeros(2 * (points - 1))

        camera.start_preview(resolution=(640,480))

        stage.move_rel([-backlash, -backlash, -backlash])
        stage.move_rel([backlash, backlash, backlash])

        image = ms.rgb_image().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image)
        templ8 = (image - mean)[100:-100, 100:-100]

        initial_stage_position = stage.position

        for i in range(points):
            stage.move_rel([side_length / (points - 1), 0, 0])
            data_stage[i, 0:] = stage.position
            frame = ms.rgb_image().astype(np.float32)
            data_cam[i, 0:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
            if i > 0:
                data_steps[i - 1] = i * side_length / (points - 1)
                data_distance[i - 1] = np.sqrt((data_cam[i, 0] - data_cam[0, 0])**2 + (data_cam[i, 1] - data_cam[0, 1])**2)
        for j in range(points):
            stage.move_rel([0, 0, side_length / (points - 1)])
            i = j + points
            data_stage[i, 0:] = stage.position
            frame = ms.rgb_image().astype(np.float32)
            data_cam[i, 0:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
            if j > 0:
                data_steps[i - 2] = j * side_length / (points - 1)
                data_distance[i - 2] = np.sqrt((data_cam[i, 0] - data_cam[points, 0])**2 + (data_cam[i, 1] - data_cam[points, 1])**2)

        df.add_data(data_stage, stage_pos, "data_stage")
        df.add_data(data_cam, cam_pos, "data_cam")
        df.add_data(data_steps, steps, "data_steps")
        df.add_data(data_distance, distance, "data_distance")

        stage.move_abs(initial_stage_position)
        camera.stop_preview()
        print("Done")
