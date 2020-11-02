from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import time
import numpy as np
import threading

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins=[])

settings = {
    'fs': 3e5,  # sample rate
    'nFreq': 1e2,  # number of frequency values to send
    'fScale': 'log',  # frequency spacing, linear or log
    'window': .01,  # length of window in seconds
    'overlap': .5,  # fractional overlap
}

sendSize = int(settings['nFreq'] / settings['window'] / settings['overlap'])
print(sendSize)


def sendFakeFFT():
  # send 1 second of (fake) data
  while True:
    time.sleep(1)
    print('emitting data', sendSize)
    socketio.emit('fft', {'s': np.random.randint(
        low=0, high=255, size=sendSize).tolist()})


t = threading.Thread(target=sendFakeFFT)
t.start()


@socketio.on('connect')
def connected():
  print('made connection')
  emit('settings', settings)


@socketio.on('disconnect')
def disconnected():
  print('disconnected client')


@socketio.on('test')
def print_msg(json):
  print(str(json))


@app.route('/')
def index():
  return render_template('index.html')


if __name__ == '__main__':
  socketio.run(app, host='127.0.0.1', port=3001,
               debug=True, use_reloader=False)
