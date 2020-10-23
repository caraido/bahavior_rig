import PySpin
import ffmpeg
import numpy as np


class Camera:
  def __init__(self, camlist, index):
    self._spincam = camlist.GetByIndex(index)
    self._spincam.Init()
    # here we will eventually want to enable hardware triggering
    # for now we'll just hardcode the framerate at 30
    self._spincam.AcquisitionFrameRateEnable.SetValue(True)
    self._spincam.AcquisitionFrameRate.SetValue(30)

    self._running = False
    self._saving = False
    self._displaying = False

    self.file = None
    self.route = ''

  def start(self, filepath=None, route=None):
    if filepath:
      self._saving = True

      # we will assume hevc for now
      # will also assume 30fps
      self.file = ffmpeg \
          .input('pipe:', format='rawvideo', pix_fmt='gray', s='1280x1024') \
          .output(filepath, vcodec='libx265') \
          .overwrite_output() \
          .run_async(pipe_stdin=True)
      # self.file = cv2.VideoWriter(
      # filepath, cv2.VideoWriter_fourcc(*'hvc1'), 30, (1024, 1280), False)

    if route:
      self.route = route
      self._displaying = True

    if not self._running:
      self._running = True
      self._spincam.BeginAcquisition()

  def stop(self):
    if self._running:
      if self._saving:
        self._saving = False
        # self.file.release()
        self.file.stdin.close()
        self.file.wait()
        del self.file
        self.file = None

      self._running = False
      self._displaying = False
      self.route = ''
      self._spincam.EndAcquisition()

  def capture(self):
    im = self._spincam.GetNextImage()
    # parse to make sure that image is complete....

    # assume image size for now
    frame = np.reshape(im.GetData(), (1024, 1280))

    if self._saving:
      self.save(frame)

    if self._displaying:
      self.display(frame)

    im.Release()

  def save(self, frame):
    self.file.stdin.write(frame.tobytes())

  def display(self, frame):
    pass

  def get_camera_property(self):
    pass

  def calibration(self):
    pass

  def save_intrinsic(self):
    pass

  def save_extrinsic(self):
    pass

  def __del__(self):
    self.stop()
    self._spincam.DeInit()
    del self._spincam


class CameraGroup:
  def __init__(self):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.cameras = [Camera(self._camlist, i)
                    for i in range(self._camlist.GetSize())]

  def __del__(self):
    for cam in self.cameras:
      del cam
    self._camlist.Clear()
    self._system.ReleaseInstance()


if __name__ == '__main__':
  cg = CameraGroup()
  for i, cam in enumerate(cg.cameras):
    cam.start(filepath=f'testing{i:02d}.mov')

  for j in range(100):
    for i, cam in enumerate(cg.cameras):
      cam.capture()

  for i, cam in enumerate(cg.cameras):
    cam.stop()
