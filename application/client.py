import base64
import time
import socketio
import cv2

from webcam_streamer import WebcamStreamer


def handle_response(*args):
    print("Response", *args)
    pass


def process_frame(sio_client, frame, seq, capture_period):
    process_time = time.time()
    b64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode("ascii")
    sio_client.emit(
        event='frame_event',
        data={
            'base64': b64,
            'time': process_time,
            'seq': seq,
        },
        callback=handle_response
    )


if __name__ == '__main__':
    sio = socketio.Client()
    sio.connect('http://localhost:5000')

    webcam = WebcamStreamer()
    webcam.run(
        action=lambda f, s, t: process_frame(sio, f, s, t),
        sleep=0.1,
        show=True,
    )
    sio.disconnect()
