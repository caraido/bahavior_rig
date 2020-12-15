# this demo serves a react app that streams white noise data encoded in jpeg

# in practice we will want to stream hevc data, which requires a different approach due to interframe compression
# we can do this using the libde265.js javascript decoder, modified for websockets (https://github.com/strukturag/libde265.js/pull/10/files/4d29e7bf3a1de850b8c3810610d16bd14698de0d)
# frames will be transmitted by flask in chunks via a websocket and attached to a canvas in js

from flask import Response, Flask, render_template, request
from flask_socketio import SocketIO, Namespace, emit
from numpy import random
from cv2 import imencode
import time

async_mode = 'threading'


class VideoStreamSocketHandler(Namespace):
  def on_connect(self):
    print('socket opened')

  def on_start(self, params):
    print('params: ', params)

    chunk_size = params['chunk_size']
    period = 1 / params['fps']

    f = open('./static/spreedmovie.hevc', 'rb')

    while True:
      data = f.read(chunk_size)
      if not data or len(data) != chunk_size:
        print('Done!')
        emit('frames', {'flush': True})  # unclear...
        f.close()
        break
      emit('frames', {'data': data})  # mode = binary?
      time.sleep(period)

  def on_disconnect(self):
    print('Socket closed')


# create the server
app = Flask(__name__)
socketio = SocketIO(app, async_mode=async_mode)

socketio.on_namespace(VideoStreamSocketHandler('/video0'))
# define the index route


@app.route('/')
def index():
  return render_template('index.html', flask_token="Hello world", async_mode=async_mode)


@app.route('/hevcdemo')
def hevcdemo():
  return render_template('socket_libde265.html')


@socketio.on('connect')
def connected():
  print('connection established')


@socketio.on('msg')
def misc(message):
  print(message)


@app.route('/api/', methods=['POST'])
def start_recording():
  print(request.json['test'])
  # lets the browser know that we received its request
  return Response(status=200)


# host the server on the local machine
if __name__ == '__main__':
  socketio.run(app, host='127.0.0.1', port=3001,
               debug=True, use_reloader=False)
