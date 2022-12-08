import queue
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

    def run(self, action, sleep=0, result_queue=None, show=True):
        video = cv2.VideoCapture(0)
        t = time.time()
        seq = 0
        result = None
        classes = [10, 14, 20, 22, 24]

        while True:
            seq += 1
            ret, frame = video.read()
            frame = self.resize_rect(frame, 224)

            if result_queue is not None:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    pass

            if show:
                big_frame = cv2.resize(frame, (512, 512))
                if result is not None:
                    bar_width = 512//len(result)
                    for i in range(len(result)):
                        big_frame = cv2.rectangle(big_frame,
                                                  (i * bar_width, 512 - int(result[i] * 256)),
                                                  ((i+1) * bar_width, 511),
                                                  (int(result[i] * 255), 0, 0), -1)
                        big_frame = cv2.putText(big_frame,
                                                f"{result[i]:.2f}",
                                                (i * bar_width + 20, 512 - int(result[i] * 256) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        big_frame = cv2.putText(big_frame,
                                                f"cls {classes[i]}",
                                                (i * bar_width + 20, 512 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('frame', big_frame)

            action(frame, seq, time.time() - t)

            t = time.time()
            time.sleep(sleep)
            if cv2.waitKey(1) == 27:
                break
        if show:
            cv2.destroyAllWindows()
