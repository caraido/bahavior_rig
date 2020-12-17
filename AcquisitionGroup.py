
import PySpin
import os

import threading
import Camera
import Nidaq


class AcquisitionGroup:
  def __init__(self, frame_rate=30, audio_settings=None):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    self.cameras = [Camera.Camera(self._camlist, i, frame_rate)
                    for i in range(self.nCameras)]
    self.nidaq = Nidaq.Nidaq(frame_rate, audio_settings)

    self._dlc_runners = []
    self._runners = []
    self.filepaths = None

  def __call__(self,filepaths):
    self.filepaths=filepaths

  def start(self, filepaths=None, isDisplayed=True):
    if not self.filepaths:
      self.filepaths = [None] * (self.nCameras + 1)
    if not isDisplayed:
      isDisplayed = [False] * (self.nCameras + 1)

    if not isinstance(isDisplayed, list) or len(isDisplayed) == 1:
      isDisplayed = [isDisplayed] * (self.nCameras + 1)

    print('detected %d cameras' % self.nCameras)

    for cam, fp, disp in zip(self.cameras, self.filepaths[: -1], isDisplayed[: -1]):
      cam.start(filepath=fp, display=disp)
      print('starting camera ' + cam.device_serial_number)

    # once the camera BeginAcquisition methods are called, we can start triggering
    self.nidaq.start(filepath=self.filepaths[-1], display=isDisplayed[-1])
    print('starting nidaq')

  def run(self):
    # begin gathering samples
    if not self._runners:  # if self._runners == []
      for i, cam in enumerate(self.cameras):
        self._runners.append(threading.Thread(target=cam.run))
        self._runners[i].start()
      self._runners.append(threading.Thread(target=self.nidaq.run))
      self._runners[-1].start()

    else:
      for i, cam in enumerate(self.cameras):
        if not self._runners[i].is_alive():
          self._runners[i] = threading.Thread(target=cam.run)
          self._runners[i].start()

      if not self._runners[-1].is_alive():
        self._runners[-1] = threading.Thread(target=self.nidaq.run)
        self._runners[-1].start()

  def stop(self):
    for cam in self.cameras:
      cam.stop()
    self.nidaq.stop()  # make sure cameras are stopped before stopping triggers

    os.remove('C:\\Users\\SchwartzLab\\Desktop\\unwanted.tdms')
    os.remove('C:\\Users\\SchwartzLab\\Desktop\\unwanted.tdms_index')

  def __del__(self):
    for cam in self.cameras:
      del cam
    self._camlist.Clear()
    self._system.ReleaseInstance()
    del self.nidaq


if __name__ == '__main__':
  ag = AcquisitionGroup()
  ag.start()
  ag.run()
