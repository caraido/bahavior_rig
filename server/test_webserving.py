from flask import Response, Flask, render_template
from numpy import random
from cv2 import imencode

#in place of recording, generate some white noise using numpy
def randomData():
  while True:
    (flag, encodedImage) = imencode('.jpg', 256*random.rand(300,200)) #encode the data using jpg format
    if not flag:
      continue

    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n') #convert to a bytearray for serving

#create the server
app = Flask(__name__)

#define the index route
@app.route('/')
def index():
  return render_template('index.html', flask_token="Hello world")

#individual cameras will be available at /cameraN
@app.route(f'/camera<int:camera_id>')
def show_data(camera_id):
  return Response(randomData(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

#allow user to interact with the recording through a simple API
@app.route('/api/', methods = ['POST'])
def start_recording():
  print('got record request')
  return render_template('index.html', flask_token = "Hello world")

#host the server on the local machine
if __name__ == '__main__':
  app.run(host='127.0.0.1',port='3001',debug=True, use_reloader=False)