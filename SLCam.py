import cv2
import PySpin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ffmpeg
import utils.calibration_utils as cau
import os
import toml
import threading
from io import BytesIO
from PIL import Image
import nidaqmx


class CharucoBoard:
  def __init__(self, grid_num=3, marker_size=0.8):
    self.grid_num = grid_num
    self.marker_size = marker_size
    self.default_dictionary = cv2.aruco.DICT_4X4_50  # default
    self.seed = 0
    self.dictionary = cv2.aruco.getPredefinedDictionary(
        self.default_dictionary)

  @property
  def board(self):
    this_board = cv2.aruco.CharucoBoard_create(self.grid_num,
                                               self.grid_num,
                                               1,
                                               self.marker_size,
                                               self.dictionary)
    return this_board

  @property
  def marker_size(self):
    return self._marker_size

  @marker_size.setter
  def marker_size(self, value):
    if value <= 0 or value >= 1:
      raise ValueError("this value can only be set between 0 ~ 1!")
    else:
      self._marker_size = value

  def save_board(self, img_size=1000):
    if self.default_dictionary == 0:
      file_name = 'charuco_board_grid#_' + str(self.grid_num) + '_marker_size_' + str(
          self.marker_size) + '_default.png'
      img = self.board.draw((img_size, img_size))
      cv2.imwrite('./multimedia/board/' + file_name, img)
    else:
      pass

  def print_board(self):
    img = self.board.draw((1000, 1000))
    plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
    plt.axis("off")
    plt.show()


class Calib:
  def __init__(self):
    self.allCorners = []
    self.allIds = []
    self.decimator = 0
    self.board = CharucoBoard().board
    self.max_size = cau.get_expected_corners(self.board)
    self.save_path = './config/config_intrinsic_'

  def reset(self):
    del self.allIds
    del self.allCorners
    del self.decimator
    self.allCorners = []
    self.allIds = []
    self.decimator = 0

  def save_config(self, camera_serial_number, width, height):
    save_path = self.save_path + camera_serial_number + '.toml'
    if os.path.exists(save_path):
      print('\n config file already exists.')
    else:
      param = cau.quick_calibrate(self.allCorners,
                                  self.allIds,
                                  self.board,
                                  width,
                                  height)
      param['camera_serial_number'] = camera_serial_number
      with open(save_path, 'w') as f:
        toml.dump(param, f)


class Camera:

  def __init__(self, camlist, index):
    self._spincam = camlist.GetByIndex(index)
    self._spincam.Init()

    # hardware triggering
    self._spincam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    # trigger has to be off to change source
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    self._spincam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_On)

    self.device_serial_number, self.height, self.width = self.get_camera_property()
    self.calib = Calib()

    self._running = False
    self._running_lock = threading.Lock()

    self._saving = False
    self.file = None

    self._displaying = False
    self.frame = None
    self.frame_count = 0
    self._frame_lock = threading.Lock()
    self._frame_bytes = BytesIO()

    self._calibrating = False

  def start(self, filepath=None, display=False):
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
      #     filepath, cv2.VideoWriter_fourcc(*'hvc1'), 30, (1024, 1280), False)
    else:
      self._saving = False

    self.frame_count = 0
    if display:
      self._displaying = True
    else:
      self._displaying = False

    with self._running_lock:
      if not self._running:
        self._running = True
        self._spincam.BeginAcquisition()

  def stop(self):
    with self._running_lock:
      if self._running:
        if self._saving:
          self._saving = False
          # self.file.release()
          self.file.stdin.close()
          self.file.wait()
          del self.file
          self.file = None

        cv2.destroyAllWindows()
        self._running = False
        self._displaying = False

        self._spincam.EndAcquisition()
        self._spincam.DeInit()

  def capture(self):
    im = self._spincam.GetNextImage()
    # parse to make sure that image is complete....
    if im.IsIncomplete():
      status = im.GetImageStatus()
      im.Release()
      raise Exception(f"Image incomplete with image status {status} ...")

    frame = np.reshape(im.GetData(), (self.height, self.width))
    if self._saving:
      self.save(frame)

    # press "c" key to turn on or off calibration mode
    if cv2.waitKey(1) & 0xFF == ord('c'):
      self.calibration_switch()

    if self._calibrating:
      self.calibration(frame)

    if self._displaying:
      # acquire lock on frame
      with self._frame_lock:
        self.frame = frame
        self.frame_count += 1
      # self.display()

    im.Release()

  def save(self, frame):
    self.file.stdin.write(frame.tobytes())

  def get_camera_property(self):
    nodemap_tldevice = self._spincam.GetTLDeviceNodeMap()
    device_serial_number = PySpin.CStringPtr(
        nodemap_tldevice.GetNode('DeviceSerialNumber')).GetValue()
    nodemap = self._spincam.GetNodeMap()
    height = PySpin.CIntegerPtr(nodemap.GetNode('Height')).GetValue()
    width = PySpin.CIntegerPtr(nodemap.GetNode('Width')).GetValue()
    return device_serial_number, height, width

  def calibration_switch(self):
    if not self._calibrating:
      print('turning on calibration mode')
      self._calibrating = True
      self.calib.reset()
    else:
      print('turning off calibration mode')
      self._calibrating = False
      self.calib.save_config(self.device_serial_number,
                             self.width,
                             self.height)

  def calibration(self, frame):
    # write something on the frame
    text = 'Calibration Mode On'
    cv2.putText(frame, text, (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

    # key step: detect markers
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 700
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5

    # get corners and refine them in openCV
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        frame, self.calib.board.dictionary, parameters=params)
    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        cv2.aruco.refineDetectedMarkers(frame, self.calib.board, corners, ids,
                                        rejectedImgPoints, parameters=params)

    # interpolate corners and draw corners
    if len(detectedCorners) > 0:
      rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
          detectedCorners, detectedIds, frame, self.calib.board)
      if detectedCorners is not None and 2 <= len(
              detectedCorners) <= self.calib.max_size and self.calib.decimator % 3 == 0:
        self.calib.allCorners.append(detectedCorners)
        self.calib.allIds.append(detectedIds)
      cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)
    self.calib.decimator += 1

    return frame

  def display(self):
    # cv2.imshow('frame', self.frame)
    if self._displaying:
      with self._frame_lock:
        frame_count = self.frame_count  # get the starting number of frames
      while self._displaying:
        with self._frame_lock:
          if self.frame_count > frame_count:  # display the frame if it's new ~ might run into issues here?
            frame_count = self.frame_count
            self._frame_bytes.seek(0)  # go to the beginning of the buffer
            Image.fromarray(self.frame).save(self._frame_bytes, 'bmp')
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + self._frame_bytes.getvalue() + b'\r\n')

  def run(self):
    while True:
      with self._running_lock:
        if self._running:
          self.capture()
        else:
          return

  def __del__(self):
    self.stop()


class Nidaq:
  def __init__(self):
    self.audio = None
    self.trigger = None
    self._audio_reader = None
    self.data = None

    self.sample_rate = int(3e5)
    self.trigger_freq = 30
    self.duty_cycle = .01
    self.read_rate = 10  # minimum recommended by NI
    self._read_size = self.sample_rate // self.read_rate
    self.read_count = 0

    self._running = False
    self._running_lock = threading.Lock()
    self._displaying = False
    self._data_lock = threading.Lock()
    self._saving = False

  def start(self, filepath=None, display=False):
    with self._running_lock:
      if not self._running:
        # audio task
        self.audio = nidaqmx.Task()
        self.audio.ai_channels.add_ai_voltage_chan(
            "Dev1/ai1"
        )  # this channel measures the audio signal
        self.audio.timing.cfg_samp_clk_timing(
            self.sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
        )
        self.audio.control(
            nidaqmx.constants.TaskMode.TASK_COMMIT
        )  # transition the task to the committed state so it's ready to start

        self.read_count = 0

        if display:
          self._displaying = True
          self._audio_reader = nidaqmx.stream_readers.AnalogSingleChannelReader(
              self.audio.in_stream)
          self._read_size = self.sample_rate // self.read_rate

          self.data = np.ndarray(shape=(1, self._read_size))

          log_mode = nidaqmx.constants.LoggingMode.LOG_AND_READ

        else:
          self._displaying = False
          self._audio_reader = None
          self._read_size = None

          self.data = None

          log_mode = nidaqmx.constants.LoggingMode.LOG

        if filepath:
          self._saving = True
          self.audio.in_stream.configure_logging(
              filepath, logging_mode=log_mode)  # see nptdms
        else:
          self._saving = False
          self.audio.in_stream.configure_logging(
              '', logging_mode=nidaqmx.constants.LoggingMode.OFF)  # not sure if this works

        # trigger task
        self.trigger = nidaqmx.Task()
        self.trigger.co_channels.add_co_pulse_chan_freq(
            "Dev1/ctr0", freq=self.trigger_freq, duty_cycle=self.duty_cycle
        )
        self.trigger.triggers.start_trigger.cfg_dig_edge_start_trig(
            "/Dev1/ai/StartTrigger"
        )  # start the video trigger with the audio channel
        self.trigger.timing.cfg_implicit_timing(
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
        )  # configure the trigger to repeat until the task is stopped
        self.trigger.control(
            nidaqmx.constants.TaskMode.TASK_COMMIT
        )

        # begin acquisition
        self.trigger.start()
        self.audio.start()

        self._running = True

  def capture(self):
    if self._displaying:
      # we will save the samples to self.data
      with self._data_lock:
        self._audio_reader.read_many_sample(
            self.data,
            number_of_samples_per_channel=self._read_size
        )
        self.read_count += 1
    else:
      # not sure... if we're logging, then we do nothing
      # if not logging, will we get an error if we do nothing?
      pass

  def display(self, socket):
    if self._displaying:
      with self._data_lock:
        read_count = self.read_count
      while self._displaying:
        with self._data_lock:

          if self.read_count > read_count:

            read_count = self.read_count
            # generate the fft, using numpy?
            spectrogram = []  # from

            # pass the most recent data to any connected browser
            socket.emit('fft', {'s': spectrogram})
            yield

  def run(self):
    while True:
      with self._running_lock:
        if self._running:
          self.capture()
        else:
          return

  def stop(self):
    with self._running_lock:
      if self._running:
        self.audio.close()
        self.trigger.close()
        self._running = False
        self._displaying = False
        self._saving = False

  def __del__(self):
    self.stop()


class AcquisitionGroup:
  def __init__(self):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    self.cameras = [Camera(self._camlist, i)
                    for i in range(self.nCameras)]
    self.nidaq = Nidaq()

    self._runners = [threading.Thread(
        target=cam.run) for cam in self.cameras] + [threading.Thread(target=self.nidaq.run)]

  def start(self, filepaths=None, isDisplayed=None):
    if not filepaths:
      filepaths = [None] * (self.nCameras + 1)
    if not isDisplayed:
      isDisplayed = [False] * (self.nCameras + 1)

    for cam, fp, disp in zip(self.cameras, filepaths[:-1], isDisplayed[:-1]):
      cam.start(filepath=fp, display=disp)

    # once the camera BeginAcquisition methods are called, we can start triggering
    self.nidaq.start(filepath=filepaths[-1], display=isDisplayed[-1])

  def run(self):
    # begin gathering samples
    for runner in self._runners:
      runner.start()

  def stop(self):
    self.nidaq.stop()
    for cam in self.cameras:
      cam.stop()

  def __del__(self):
    for cam in self.cameras:
      cam.stop()
      del cam
    self._camlist.Clear()
    self._system.ReleaseInstance()
    del self.nidaq


# if __name__ == '__main__':
#     board = CharucoBoard()
#     board.save_board()
#     cg = CameraGroup()
#
#     for i, cam in enumerate(cg.cameras):
#         # cam.start(filepath=f'testing{i:02d}.mov')
#         cam.start(display=True)
#
#     for j in range(100):
#         for i, cam in enumerate(cg.cameras):
#             cam.capture()
#     for i, cam in enumerate(cg.cameras):
#         cam.stop()
#
#     del cg
#     for i, cam in enumerate(cg.cameras):
#        cam.stop()
