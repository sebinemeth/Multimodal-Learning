import gc

import numpy as np


class FrameCache(object):
    def __init__(self, length, img_size=(224, 224)):
        self.frames = np.zeros((*img_size, 3, length))
        self.seq = 0

    def add_frame(self, f, seq):
        if seq < self.seq:
            print("skipping old frame", seq)
            return
        self.seq = seq
        self.frames = np.roll(self.frames, shift=1, axis=3)
        gc.collect()
        self.frames[:, :, :, 0] = f
        # print(np.mean(self.frames))
        # detect(f)

    def write(self):
        """
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (224, 224))
        for i in range(32):
            f = self.frames[:, :, :, i]
            out.write(f.astype(np.uint8))
        out.release()
        """
        pass
