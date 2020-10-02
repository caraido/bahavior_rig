from flask import Response, Flask, render_template
from numpy import random
from cv2 import imencode

def randomData():
  while True:
    #rand(300,200)
    (flag, encodedImage) = imencode('.jpg', 256*random.rand(300,200))
    if not flag:
      continue

    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

app = Flask(__name__)
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/camera1')
def camera1():
  return Response(randomData(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/camera2')
def camera1():
  return Response(randomData(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/camera3')
def camera1():
  return Response(randomData(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/camera4')
def camera1():
  return Response(randomData(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
  app.run(host='127.0.0.1',port='3001',debug=True, use_reloader=False)