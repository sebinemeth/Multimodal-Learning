import argparse
import base64
import time

import cv2
import eventlet
import numpy as np
import socketio

from application.model import Model
from application.webcam_streamer import WebcamStreamer
from application.frame_cache import FrameCache

sio = socketio.Server()
app = socketio.WSGIApp(sio)

cache = FrameCache(32)
model = Model()


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

cnt = 0
def process_frame(frame, seq, t):
    global cnt
    print(seq, t)
    cache.add_frame(frame, seq)
    if cnt % 32 == 0:
        model(cache.frames)
        input()
    cnt += 1


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
        webcam.run(action=process_frame, sleep=0.05, show=True)
        cache.write()
    else:
        eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
