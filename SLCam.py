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
import time

BUFFER_TIME = .005  # time in seconds allowed for overhead


class CharucoBoard:
  def __init__(self, x, y, marker_size=0.8):
    self.x = x
    self.y = y
    self.marker_size = marker_size
    self.default_dictionary = cv2.aruco.DICT_4X4_50  # default
    self.seed = 0
    self.dictionary = cv2.aruco.getPredefinedDictionary(
        self.default_dictionary)

  @property
  def board(self):
    this_board = cv2.aruco.CharucoBoard_create(self.x,
                                               self.y,
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
      file_name = 'charuco_board_shape_' + str(self.x) + 'x' + str(self.y) + '_marker_size_' + str(
          self.marker_size) + '_default.png'
      img = self.board.draw((img_size, img_size))
      result = cv2.imwrite('./multimedia/board/' + file_name, img)
      if result:
        print('save board successfully! Name: ' + file_name)
      else:
        raise Exception('save board failed! Name: '+file_name)

  def print_board(self):
    img = self.board.draw((1000, 1000))
    plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
    plt.axis("off")
    plt.show()


class Calib:
  def __init__(self, calib_type):
    self._get_type(calib_type)

    self.allCorners = []
    self.allIds = []
    self.decimator = 0
    self.config = None

    self.board = CharucoBoard(x=6, y=2).board

    self.max_size = cau.get_expected_corners(self.board)
    self.save_path = './config/config_'+self.type+'_'
    self.load_path = './config/'

  def _get_type(self, calib_type):
    if calib_type == 'intrinsic' or calib_type == 'extrinsic':
      self.type = calib_type
    else:
      raise ValueError("type can only be intrinsic or extrinsic!")

  def reset(self):
    del self.allIds, self.allCorners, self.decimator, self.config
    self.allCorners = []
    self.allIds = []
    self.decimator = 0
    self.config = None

  def load_config(self, camera_serial_number):
    if not os.path.exists(self.load_path):
      os.mkdir(self.load_path)
      raise Warning("config directory doesn't exist. creating one...")

    items = os.listdir(self.load_path)
    for item in items:
      if camera_serial_number in item and self.type in item:
        path = self.load_path+'config_' + self.type + \
            '_' + camera_serial_number + '.toml'
        with open(path, 'r') as f:
          # there only should be only one calib file for each camera
          self.config = toml.load(f)

  def save_config(self, camera_serial_number, width, height):
    save_path = self.save_path + camera_serial_number + '.toml'
    if os.path.exists(save_path):
      print('\n config file already exists.')
    else:
      if self.type == "intrinsic":
        param = cau.quick_calibrate(self.allCorners,
                                    self.allIds,
                                    self.board,
                                    width,
                                    height)
        param['camera_serial_number'] = camera_serial_number
        with open(save_path, 'w') as f:
          toml.dump(param, f)
        print('intrinsic calibration configuration saved!')
      else:
        param = {'corners': self.allCorners,
                 'ids': self.allIds, 'CI': 5,
                 'camera_serial_number': camera_serial_number}
        with open(save_path, 'w') as f:
          toml.dump(param, f)
        print('extrinsic calibration configuration saved!')


class Camera:

  def __init__(self, camlist, index, frame_rate):
    self._spincam = camlist.GetByIndex(index)
    self._spincam.Init()

    # hardware triggering
    self._spincam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    # trigger has to be off to change source
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    self._spincam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_On)

    self.frame_rate = frame_rate

    self.device_serial_number, self.height, self.width = self.get_camera_property()
    self.in_calib = Calib('intrinsic')
    self.ex_calib = Calib('extrinsic')

    self._running = False
    self._running_lock = threading.Lock()

    self._saving = False
    self.file = None

    self._displaying = False
    self.frame = None
    self.frame_count = 0
    self._frame_lock = threading.Lock()
    self._frame_bytes = BytesIO()

    self._in_calibrating = False
    self._ex_calibrating = False

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

    # press "i" or "e" key to turn on or off calibration mode
    if cv2.waitKey(1) & 0xFF == ord('c'):
      self.extrinsic_calibration_switch()
    if cv2.waitKey(1) & 0xFF == ord('i'):
      self.intrinsic_calibration_switch()

    # check calibration status
    if self._in_calibrating and self._ex_calibrating:
      raise Warning('Only one type of calibration can be turned on!')

    # intrinsic calibration
    if self._in_calibrating and not self._ex_calibrating:
      self.intrinsic_calibration(frame)

    # extrinsic calibration
    if self._ex_calibrating and not self._in_calibrating:
      self.extrinsic_calibration(frame)

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

  def intrinsic_calibration_switch(self):
    if not self._in_calibrating:
      print('turning on intrinsic calibration mode')
      self._in_calibrating = True
      self.in_calib.reset()
    else:
      print('turning off intrinsic calibration mode')
      self._in_calibrating = False
      self.in_calib.save_config(self.device_serial_number,
                                self.width,
                                self.height)

  def extrinsic_calibration_switch(self):
    if not self._ex_calibrating:
      print('turning on extrinsic calibration mode')
      self._ex_calibrating = True
      self.ex_calib.reset()
      self.ex_calib.load_config(self.device_serial_number)
    else:
      print('turning off extrinsic calibration mode')
      self._ex_calibrating = False
      self.ex_calib.save_config(self.device_serial_number,
                                self.width,
                                self.height)

  def intrinsic_calibration(self, frame):
    # write something on the frame
    text = 'Intrinsic calibration mode On'
    cv2.putText(frame, text, (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 125), 2)

    # key step: detect markers
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 700
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5

    # get corners and refine them in openCV
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        frame, self.in_calib.board.dictionary, parameters=params)
    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        cv2.aruco.refineDetectedMarkers(frame, self.in_calib.board, corners, ids,
                                        rejectedImgPoints, parameters=params)

    # interpolate corners and draw corners
    if len(detectedCorners) > 0:
      rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
          detectedCorners, detectedIds, frame, self.in_calib.board)
      if detectedCorners is not None and 2 <= len(
              detectedCorners) <= self.in_calib.max_size and self.in_calib.decimator % 3 == 0:
        self.in_calib.allCorners.append(detectedCorners)
        self.in_calib.allIds.append(detectedIds)
      cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)
    self.in_calib.decimator += 1

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

  def extrinsic_calibration(self, frame):
    if self.ex_calib.config is None:
      text = 'No configuration file found. Performing initial extrinsic calibration... '
      cv2.putText(frame, text, (50, 50),
                  cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
      # key step: detect markers
      params = cau.get_calib_param()

      # get corners and refine them in openCV
      corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, self.ex_calib.board.dictionary, parameters=params)

      cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)
      self.ex_calib.allCorners = corners
      self.ex_calib.allIds = ids
    else:
      text = 'Found configuration file for this camera. Calibrating...'
      cv2.putText(frame, text, (50, 50),
                  cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      truecorners = self.ex_calib.config['corners']
      trueids = self.ex_calib.config['ids']
      CI = self.ex_calib.config['CI']  # pixels

      # key step: detect markers
      params = cau.get_calib_param()
      corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, self.ex_calib.board.dictionary, parameters=params)
      cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

      # check if aligned:

      for id, corner in zip(ids, corners):
        color = cau.check_aligned(id, corner, trueids, truecorners, CI)
        cv2.rectangle(frame, truecorners[i]
                      [0], truecorners[i][[2]], color, 5)

  def run(self):
    last = 0
    # we will wait a bit less than the interval between frames
    interval = (1 / self.read_rate) - BUFFER_TIME
    while True:
      time.sleep(max(last + interval - time.time(), 0))
      with self._running_lock:
        if self._running:
          last = time.time()
          self.capture()
        else:
          return

  def __del__(self):
    self.stop()


class Nidaq:
  def __init__(self, frame_rate, audio_rate):
    self.audio = None
    self.trigger = None
    self._audio_reader = None
    self.data = None

    self.sample_rate = audio_rate
    self.trigger_freq = frame_rate
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
    '''
    Calculate the spectrogram of the data and send to connected browsers.
    There are many ways to approach this, in particular by using wavelets or by using
    overlapping FFTs. For now just trying non-overlapping FFTs ~ the simplest approach.
    '''
    if self._displaying:
      with self._data_lock:
        read_count = self.read_count
      while self._displaying:
        with self._data_lock:

          if self.read_count > read_count:
            # note that we're not guaranteed to be gathering sequential reads...

            read_count = self.read_count
            # generate the fft, using numpy?
            spectrogram = np.fft.fftshift(np.fft(self.data))

            # pass the most recent data to any connected browser
            socket.emit('fft', {'s': spectrogram})

  def run(self):
    last = 0
    # we will wait a bit less than the interval between frames
    interval = (1 / self.read_rate) - BUFFER_TIME
    while True:
      time.sleep(max(last + interval - time.time(), 0))
      with self._running_lock:
        if self._running:
          last = time.time()
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
  def __init__(self, frame_rate=30, audio_rate=int(3e5)):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    self.cameras = [Camera(self._camlist, i, frame_rate)
                    for i in range(self.nCameras)]
    self.nidaq = Nidaq(frame_rate, audio_rate)

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
#      board = CharucoBoard(6,2)
#      #board.save_board(2000)
#      cg = CameraGroup()
#
#      for i, cam in enumerate(cg.cameras):
#          # cam.start(filepath=f'testing{i:02d}.mov')
#          cam.start(display=True)
#
#      for j in range(500):
#          for i, cam in enumerate(cg.cameras):
#              cam.capture()
#      for i, cam in enumerate(cg.cameras):
#          cam.stop()
#
#      del cg
#     for i, cam in enumerate(cg.cameras):
#        cam.stop()
