import argparse
import base64
import queue
import time

import cv2
import eventlet
import numpy as np
import socketio
import multiprocessing as mp

from application.model import model_processor
from application.webcam_streamer import WebcamStreamer
from application.frame_cache import FrameCache

cache = FrameCache(32, fps=16)
frame_queue = mp.Queue(maxsize=1)
result_queue = mp.Queue(maxsize=20)
stop_event = mp.Event()

"""sio = socketio.Server()
app = socketio.WSGIApp(sio)

@sio.event
def connect(sid, environ):
    print('connect ', sid)


@sio.event
def frame_event(sid, data):
    b64 = data["base64"]
    fr = np.frombuffer(base64.b64decode(b64), np.uint8)
    fr = cv2.imdecode(fr, cv2.IMREAD_COLOR)
    cache.add_frame(fr, data["seq"])
    t = time.time() - data["time"]
    return t


@sio.event
def disconnect(sid):
    print('disconnect ', sid)
    cache.write()

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ServerApp',
        description='Runs the network on camera feed')
    parser.add_argument('-l', '--lonely',
                        help='Runs on local machine without client',
                        action='store_true')
    args = parser.parse_args()

    if args.lonely:
        webcam = WebcamStreamer()
        processor = mp.Process(target=model_processor, args=(frame_queue, result_queue, stop_event))
        processor.start()

        def process_frame(frame, seq, t):
            # print(seq, t)
            cache.add_frame(frame, seq)
            try:
                frame_queue.put_nowait(cache.frames)
            except queue.Full:
                pass

        webcam.run(
            action=process_frame,
            result_queue=result_queue,
            show=True
        )
        cache.write()
        stop_event.set()
        processor.join()

        print(" - end of script")

        # x = threading.Thread(target=thread_function, args=(1,))
        # x.start()
    else:
        eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
