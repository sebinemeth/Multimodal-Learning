import time

import cv2


class WebcamStreamer(object):
    def __init__(self):
        pass

    def resize_rect(self, frame, size):
        h, w, _ = frame.shape
        min_dim = min(h, w)
        ratio = size / min_dim
        frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
        return frame[int(h * ratio / 2 - size / 2):int(h * ratio / 2 + size / 2),
               int(w * ratio / 2 - size / 2):int(w * ratio / 2 + size / 2), :]

    def run(self, action, sleep=0.1, show=True):
        video = cv2.VideoCapture(0)
        t = time.time()
        seq = 0

        while True:
            seq += 1
            ret, frame = video.read()
            frame = self.resize_rect(frame, 224)
            if show:
                cv2.imshow('frame', cv2.resize(frame, (512, 512)))

            action(frame, seq, time.time() - t)

            t = time.time()
            time.sleep(sleep)
            if cv2.waitKey(1) == 27:
                break
        if show:
            cv2.destroyAllWindows()
