import PySpin
import os

import threading
from Camera import Camera
from Nidaq import Nidaq

import ProcessingGroup as pg

model_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\Alec_second_try-Devon-2020-12-07\exported-models\DLC_Alec_second_try_resnet_50_iteration-0_shuffle-1'
class AcquisitionGroup:
  def __init__(self, frame_rate=30, audio_settings=None):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    self.cameras = [Camera(self._camlist, i, frame_rate)
                    for i in range(self.nCameras)]
    self.nidaq = Nidaq(frame_rate, audio_settings)
    self.children = self.cameras + [self.nidaq]
    self.nChildren = self.nCameras + 1

    self._processors = [None] * self.nChildren
    self._runners = [None] * self.nChildren
    self.filepaths = None

    self.pg = pg.ProcessingGroup()

  def start(self, filepaths=None, isDisplayed=True):
    self.filepaths = filepaths
    if not self.filepaths:
      self.filepaths = [None] * self.nChildren
    if not isDisplayed:
      isDisplayed = [False] * self.nChildren

    if not isinstance(isDisplayed, list) or len(isDisplayed) == 1:
      isDisplayed = [isDisplayed] * self.nChildren

    print('detected %d cameras' % self.nCameras)

    for child, fp, disp in zip(self.children, self.filepaths[: -1], isDisplayed[: -1]):
      child.start(filepath=fp, display=disp)
      # print('starting camera ' + child.device_serial_number)

    # once the camera BeginAcquisition methods are called, we can start triggering
    # self.nidaq.start(filepath=self.filepaths[-1], display=isDisplayed[-1])
    # print('starting nidaq')

  def run(self):
    # begin gathering samples
    # if not self._runners:  # if self._runners == []
    #   for i, child in enumerate(self.children):
    #     self._runners.append(threading.Thread(target=child.run))
    #     self._runners[i].start()

      # self._runners.append(threading.Thread(target=self.nidaq.run))
      # self._runners[-1].start()

    # else:
    for i, child in enumerate(self.children):
      if self._runners[i] is None or not self._runners[i].is_alive():
        self._runners[i] = threading.Thread(target=child.run)
        self._runners[i].start()

      # if not self._runners[-1].is_alive():
      #   self._runners[-1] = threading.Thread(target=self.nidaq.run)
      #   self._runners[-1].start()

  def process(self, i, options):
    # if it's recording, process() shouldn't be run.
    if self.filepaths is None:
      if self._processors[i] is None or not self._processors[i].is_alive():
        self.children[i].processing = options
        self._processors[i] = threading.Thread(
            target=self.children[i].run_processing)
        self._processors[i].start()

  def stop(self):
    # for cam in self.cameras:
    #   cam.stop()
    # self.nidaq.stop()  # make sure cameras are stopped before stopping triggers
    for child in self.children:
      child.stop()
    del self
    self._processors = [None] * self.nChildren

    if self.filepaths is not None:
      rootpath = os.path.split(self.filepaths[0])[:-2]
      self.pg(rootpath,model_path)
      self.post_analysis = threading.Thread(
        target = self.pg.post_process)
      try:
        self.post_analysis.start()
      except Warning("Post analysis failed. Have to do it manually.")
    # ProcessGroup takeover?

    # TODO: should be able to remove this
    os.remove('C:\\Users\\SchwartzLab\\Desktop\\unwanted.tdms')
    os.remove('C:\\Users\\SchwartzLab\\Desktop\\unwanted.tdms_index')

  def __del__(self):
    for child in self.children:
      del child
    self._camlist.Clear()
    self._system.ReleaseInstance()
    # del self.nidaq


if __name__ == '__main__':
  from main import audio_settings
  default_model_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\Alec_second_try-Devon-2020-12-07\exported-models'
  filepaths = r'C:\Users\SchwartzLab\Desktop'
  ag = AcquisitionGroup(audio_settings=audio_settings)
  # preview
  ag.start()
  ag.run()

  # calibration/dlc
  ag.process(0, {'mode': 'DLC', 'modelpath': default_model_path}) #'DLC'/'extrinsic'/'intrinsic'
  ag.stop() # saving calibration stuff

  # record
  ag.start(filepaths=[filepaths,filepaths,filepaths,filepaths])
  ag.run()

  ag.process(0,{'mode': 'DLC', 'modelpath': default_model_path}) # this shouldn't work when there's file path

  ag.stop() # after this there should be post processing?
