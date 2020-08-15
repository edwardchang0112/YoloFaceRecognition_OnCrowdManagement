#!flask/bin/python
from flask import Flask, request, render_template, Response, url_for, redirect, stream_with_context
from gevent.pywsgi import WSGIServer
import pickle
from Face_detectionAndRecognition import VideoCapture
import cv2
import requests

app = Flask(__name__)


@app.route('/thermalanalysis')
def Base_getdata():
    return render_template('Base.html', name='0', num='0')


def gen(camera):
    while True:
        frame, face_num, face_names = camera.get_frame()
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        data = {"face_num": face_num, "face_names": face_names, "body_temperature": 'PASS!'}
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route('/thermalanalysis/video_feed')
def video_feed():
    VC = VideoCapture()
    return Response(gen(VC), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thermalanalysis/VideoStream', methods=['GET'])
def VideoStream():
    return render_template('VideoStream.html')

if __name__ == '__main__':
    app.debug = True
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
