import PySpin
import os

import threading
from Camera import Camera
from Nidaq import Nidaq

import ProcessingGroup as pg
# import RigStatus


class AcquisitionGroup:
  # def __init__(self, frame_rate=30, audio_settings=None):
  def __init__(self, status, hostname='localhost'):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    self.cameras = [Camera(self._camlist, i, status['frame rate'].current, (hostname, status[f'camera {i}'].current['port'].current))
                    for i in range(self.nCameras)]
    self.nidaq = Nidaq(status['frame rate'].current, status['sample frequency'].current,
                       status['read rate'].current, status['spectrogram'].current, hostname)
    self.children = self.cameras + [self.nidaq]
    self.nChildren = self.nCameras + 1

    self._processors = [None] * self.nChildren
    self._runners = [None] * self.nChildren
    self._displayers = [None] * self.nChildren
    self.filepaths = None

    self.started = False
    self.processing = False
    self.running = False

    self.pg = pg.ProcessingGroup()

    print('done setting up ag. is camera 3 running? ', self.cameras[3].running)

  def start(self, filepaths=None, isDisplayed=True):
    if self.started:
      print('already started. Filepath/display unchanged.')
      return

    self.filepaths = filepaths
    if not self.filepaths:
      self.filepaths = [None] * self.nChildren
    if not isDisplayed:
      isDisplayed = [False] * self.nChildren

    if not isinstance(isDisplayed, list) or len(isDisplayed) == 1:
      # TODO: does this work? what if isDisplayed = [True]?
      # wouldn't we then get [[True], [True], [True], ...] ?
      isDisplayed = [isDisplayed] * self.nChildren

    print('detected %d cameras' % self.nCameras)

    for child, fp, disp in zip(self.children, self.filepaths[: -1], isDisplayed[: -1]):
      child.start(filepath=fp, display=disp)
      print('started camera ' + child.device_serial_number)

    # once the camera BeginAcquisition methods are called, we can start triggering
    self.nidaq.start(filepath=self.filepaths[-1], display=isDisplayed[-1])
    print('started nidaq')

    self.started = True

  def run(self):
    print('called ag.run')
    # begin gathering samples
    # if not self._runners:  # if self._runners == []
    #   for i, child in enumerate(self.children):
    #     self._runners.append(threading.Thread(target=child.run))
    #     self._runners[i].start()

    # self._runners.append(threading.Thread(target=self.nidaq.run))
    # self._runners[-1].start()

    # else:
    for i, child in enumerate(self.children):
      print('child ', i)
      if self._runners[i] is None or not self._runners[i].is_alive():
        print('was not alive')
        self._runners[i] = threading.Thread(target=child.run)
        print('made thread object, starting')
        self._runners[i].start()
        print('done starting')
      if self._displayers[i] is None or not self._displayers[i].is_alive():
        self._displayers[i] = threading.Thread(target=child.display)
        self._displayers[i].start()
    self.running = True
    #
    #       # if not self._runners[-1].is_alive():
    #       #   self._runners[-1] = threading.Thread(target=self.nidaq.run)
    #   self._runners[-1].start()
    print('finished ag.run')

  def process(self, i, options):
    # if it's recording, process() shouldn't be run. except dlc
    if not any(self.filepaths) or options['mode'] == 'DLC':
      if self._processors[i] is None or not self._processors[i].is_alive():
        self.children[i].processing = options
        self._processors[i] = threading.Thread(
            target=self.children[i].run_processing)
        self._processors[i].start()
    self.processing = True

  def stop(self):
    # for cam in self.cameras:
    #   cam.stop()
    # self.nidaq.stop()  # make sure cameras are stopped before stopping triggers
    for child in self.children:
      child.stop()
    print('stopped children')
    for child in self.children:
      print('waiting for next child')
      child.wait_for()
    #del self.children
    # self._processors = [None] * self.nChildren

    self.processing = False
    self.running = False
    self.started = False

    if self.filepaths is not None and any(self.filepaths):
      rootpath = os.path.split(self.filepaths[0])[0]
      self.pg(rootpath)
      self.post_analysis = threading.Thread(
          target=self.pg.post_process)
      try:
        self.post_analysis.start()
      except:
        Warning("Post analysis failed. Have to do it manually.")
    # ProcessGroup takeover?

  def __del__(self):
    del self.children
    self._camlist.Clear()
    self._system.ReleaseInstance()
    # del self.nidaq


if __name__ == '__main__':
  from utils.audio_settings import audio_settings
  import utils.path_operation_utils as pop
  default_model_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\Alec_second_try-Devon-2020-12-07\exported-models\DLC_Alec_second_try_resnet_50_iteration-0_shuffle-1'
  filepaths = r'D:'
  ag = AcquisitionGroup(audio_settings=audio_settings)
  # preview
  ag.start()
  ag.run()

  ag.stop()

  # dlc
  # ag.start()
  # ag.run()
  # ag.process(0, {'mode': 'DLC', 'modelpath': default_model_path}) #'DLC'/'extrinsic'/'intrinsic'
  # ag.cameras[0].display()
  # ag.stop() # saving calibration stuff

  # calibration
  ag.start()
  ag.run()
  ag.process(1, {'mode': 'extrinsic'})
  ag.cameras[1].display()
  ag.stop()

  ag.start()
  ag.run()
  ag.process(0, {'mode': 'intrinsic'})
  ag.process(0, {'mode': 'extrinsic'})  # this shouldn't work
  ag.stop()

  camera_list = []
  for i in range(ag.nCameras):
    camera_list.append(ag.cameras[i].device_serial_number)
  path = 'behavior_data_temp'
  name = 'alec_testing'

  paths = pop.reformat_filepath(path, name, camera_list)

  # record
  ag.start(filepaths=paths)
  ag.run()
  #ag.process(0,{'mode': 'intrinsic'})
  # this should work when there's file path
  ag.process(0, {'mode': 'DLC', 'modelpath': default_model_path})

  ag.stop()  # with post processing
