import PySpin
from PIL import Image
from io import BytesIO
import numpy as np
from flask import Response, Flask
# import time
# print(time.time())

app = Flask(__name__)

sys = PySpin.System.GetInstance()
camlist = sys.GetCameras()
cam = camlist.GetByIndex(0)

cam.Init()
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

# allows setting of the framerate, for test only
cam.AcquisitionFrameRateEnable.SetValue(True)
cam.AcquisitionFrameRate.SetValue(15)

byte_io = BytesIO()


def gen(camera):
  '''
  Takes the next available image, converts it to a bitmap (no compression), and sends to display
  In practice we want to do this a bit differently
    ~ one process should gather the frames and store them in memory, possibly also encoding
    ~ another process should take the most recent frame and send it to display if available
  '''
  while True:

    im = camera.GetNextImage()
    byte_io.seek(0)  # go to the beginning of the buffer
    Image.fromarray(np.reshape(
        im.GetData(), (1024, 1280))).save(byte_io, 'bmp')

    # print(im.GetTimeStamp())
    im.Release()

    # print(time.time())
    yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + byte_io.getvalue() + b'\r\n')


@app.route('/video0')
def video0():
  print(time.time())
  cam.BeginAcquisition()
  return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=3001, debug=True, use_reloader=False)
