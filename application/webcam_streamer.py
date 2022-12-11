import queue
import time

import cv2


class WebcamStreamer(object):
    def __init__(self):
        pass

    def run(self, action, sleep=0, result_queue=None, show=True):
        video = cv2.VideoCapture(0)
        t = time.time()
        seq = 0
        det = 0
        result = None
        r = None
        classes = [10, 14, 20, 22, 24]

        while True:
            seq += 1
            ret, frame = video.read()

            if result_queue is not None:
                try:
                    det, r = result_queue.get_nowait()
                    if r is not None:
                        result = r
                except queue.Empty:
                    pass

            if show:
                big_frame = cv2.resize(frame, (512, 512))
                big_frame = cv2.rectangle(big_frame,
                                          (0, 0),
                                          (512 - int(512 * (1 - det)), 23),
                                          (128, 0, 0) if r is None else (0, 0, 128), -1)
                big_frame = cv2.putText(big_frame,
                                        f"{det:.2f}",
                                        (10, 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
                if result is not None:
                    bar_width = 512 // len(result)
                    for i in range(len(result)):
                        big_frame = cv2.rectangle(big_frame,
                                                  (i * bar_width, 512 - int(result[i] * 256)),
                                                  ((i + 1) * bar_width, 511),
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

            action(cv2.resize(frame, (224, 224)), seq, time.time() - t)

            t = time.time()
            time.sleep(sleep)
            if cv2.waitKey(1) == 27:
                break
        if show:
            cv2.destroyAllWindows()
