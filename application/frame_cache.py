import gc
import time

import cv2
import numpy as np


class FrameCache(object):
    def __init__(self, length, fps, img_size=(224, 224)):
        self.frames = np.zeros((length, *img_size, 3))
        self.seq = 0
        self.time = time.time()
        self.fps = fps

    def add_frame(self, frame, seq):
        if seq < self.seq:
            print("skipping old frame", seq)
            return
        if time.time() - self.time < 1 / self.fps:
            return
        self.seq = seq
        self.time = time.time()
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        gc.collect()
        self.frames[-1, :, :, :] = frame
        # print(np.mean(self.frames))
        # detect(f)

    def write(self):
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (224, 224))
        for i in range(self.frames.shape[0]):
            f = self.frames[i, :, :, :]
            out.write(f.astype(np.uint8))
        out.release()
