
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
from scipy import ndimage
import matplotlib.pyplot as plt
from contextlib import contextmanager, closing
import data_file
import cv2
from camera_stuff import find_template
#import h5py
import threading
import queue

def image_capture(start_t, event, ms, q):
    while event.is_set():
        frame = ms.rgb_image().astype(np.float32)
        capture_t = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q.put(frame)
        tim = capture_t - start_t
        q.put(tim)
        print('Number of itms in the queue: {}'.format(q.qsize()))
        time.sleep(0.2)

if __name__ == "__main__":

    with load_microscope("microscope_settings.npz") as ms, \
         closing(data_file.Datafile(filename = "drift.hdf5")) as df:

        assert picamera_supports_lens_shading(), "You need the updated picamera module with lens shading!"

        camera = ms.camera
        stage = ms.stage

        camera.resolution=(640,480)

        cam_pos = df.new_group("data", "drift")

        N_frames = 500
        #need to be consistant between drift.py and drift_plot.py

        camera.start_preview(resolution=(640,480))

        image = ms.rgb_image().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image)
        templ8 = (image - mean)[100:-100, 100:-100]
        cam_pos['template'] = templ8
        cam_pos['initial_image'] = image

        q = queue.Queue()
        event = threading.Event()

        start_t = time.time()
        t = threading.Thread(target = image_capture, args = (start_t, event, ms, q), name = 'thread1')
        event.set()
        t.start()

        try:
            while event.is_set():
                if not q.empty():
                    data = np.zeros((N_frames, 3))
                    for i in range(N_frames):
                        frame = q.get()
                        tim = q.get()
                        data[i, 0] = tim
                        data[i, 1:], corr = find_template(templ8, frame - np.mean(frame), return_corr = True, fraction=0.5)
                    df.add_data(data, cam_pos, "data")
                    imgfile_location_1 = "/home/pi/summer/drift/frames/drift_%s.jpg" % time.strftime("%02Y.%02m.%02d_%02H:%02M:%02S")
                    imgfile_location_2 = "/home/pi/summer/drift/frames/corr_%s.jpg" % time.strftime("%02Y.%02m.%02d_%02H:%02M:%02S")
                    cv2.imwrite(imgfile_location_1, frame)
                    cv2.imwrite(imgfile_location_2, corr * 255.0 / np.max(corr))
                else:
                    time.sleep(0.5)
                print("Looping")
            print("Done")
        except KeyboardInterrupt:
            event.clear()
            t.join()
            camera.stop_preview()
            print ("Got a keyboard interrupt, stopping")
            