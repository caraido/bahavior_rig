import PySpin


class Camera:
  def __init__(self, spincam):
    self._spincam = spincam
    self._spincam.Init()
    # here we will eventually want to enable hardware triggering

    self.running = False

  def start(self):
    self.running = True
    self._spincam.BeginAcquisition()

  def stop(self):
    self.running = False
    self._spincam.EndAcquisition()

  def get_camera_property(self):
    pass

  def calibration(self):
    pass

  def acquire_image(self):
    pass

  def save_image(self):
    pass

  def save_video(self):
    pass

  def save_intrinsic(self):
    pass

  def save_extrinsic(self):
    pass

  def __del__(self):
    if self.running:
      self.stop()
    self._spincam.DeInit()
    del self._spincam


class CameraGroup:
  def __init__(self):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.cameras = [Camera(camlist.GetByIndex(i))
                    for i in range(self._camlist.GetSize())]

  def __del__(self):
    for cam in self.cameras:
      del cam
    self._camlist.Clear()
    self._system.ReleaseInstance()
