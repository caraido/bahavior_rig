import PySpin
import ffmpeg
import numpy as np
import socket

# config = {'c:v': 'libx265'}


def make_encoding_stream():

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(('127.0.0.1', 5000))
  sock.listen(1)

  # now we should attach a process to the socket to accept connects
  # while True:
  #   conn = sock.accept()

  # when the connection is made we want to start sending frame data in a predefined bitlength

  return sock, (
      ffmpeg
      .input('tcp://127.0.0.1:5000', format='rawvideo', pix_fmt='gray', s='1280x1024')
      .output('pipe:', format='ismv', vcodec='libx265', pix_fmt='yuv420p')
      .run_async(pipe_stdout=True)
  )


def send_frame(camera, sock):

  im = (
      camera
      .GetNextImage()
      .GetData()
      .astype('uint8')
      .tobytes()
  )

  sock.send(im)


if __name__ == "__main__":
  f = open('test_output.mp4', 'wb')

  sys = PySpin.System.GetInstance()
  camlist = sys.GetCameras()
  cam = camlist.GetByIndex(0)

  cam.Init()
  cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

  cam.BeginAcquisition()

  outstream, instream = make_encoding_stream()

  for i in range(0, 100):
    send_frame(cam, outstream)

  outdata = outstream.stdout.read(1280*1024*3)

  f.close()

  cam.EndAcquisition()
  cam.DeInit()
