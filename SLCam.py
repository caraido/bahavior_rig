import cv2
import PySpin
import numpy as np
from scipy import signal, interpolate
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
from nidaqmx.stream_readers import AnalogSingleChannelReader as AnalogReader
import time
import pandas as pd

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

  @property
  def params(self):
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 700
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5
    return params

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
          try:
            self.config['ids'] = cau.reformat(self.config['ids'])
            self.config['corners'] = cau.reformat(self.config['corners'])
            markers = pd.DataFrame({'truecorners': list(self.config['corners'])},
                                                  index=list(self.config['ids']))
            self.config['markers'] = markers
          except ValueError:
            print("there's nothing in the configuration file called ids! Please check.")


  def save_config(self, camera_serial_number, width, height):
    save_path = self.save_path + camera_serial_number + '.toml'
    if os.path.exists(save_path):
      print('Configuration file already exists.')
    else:
      if self.type == "intrinsic":
        param = cau.quick_calibrate(self.allCorners,
                                    self.allIds,
                                    self.board,
                                    width,
                                    height)
        param['camera_serial_number'] = camera_serial_number
        if len(param)>1:
          with open(save_path, 'w') as f:
            toml.dump(param, f)
          print('intrinsic calibration configuration saved!')
        else:
          print("intrinsic calibration configuration NOT saved due to lack of markers.")
      else:
        if self.allIds is not None and not len(self.allIds)<self.max_size+1:
          param = {'corners': np.array(self.allCorners),
                   'ids': np.array(self.allIds), 'CI': 5,
                   'camera_serial_number': camera_serial_number}
          with open(save_path, 'w') as f:
            toml.dump(param, f, encoder=toml.TomlNumpyEncoder())
            print('extrinsic calibration configuration saved!')
        else:
          # TODO: should be a pop up window/show up on the screen
          raise Exception("failed to record all Ids! can't save configuration. Please calibrate again.")


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
      # '1280x1024' can be replaced by self.width and self.height
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

  def display_switch_on(self):
    if not self._displaying:
      self._displaying=True
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
        # cv2.destroyAllWindows() #this appears to be causing errors?
        self._running = False
        self._displaying = False

        self._spincam.EndAcquisition()
        self._spincam.DeInit()

        print(f'stopped camera {self.device_serial_number}')

  def capture(self):
    im = self._spincam.GetNextImage(100)

    # parse to make sure that image is complete....
    if im.IsIncomplete():
      status = im.GetImageStatus()
      im.Release()
      raise Exception(f"Image incomplete with image status {status} ...")

    frame = np.reshape(im.GetData(), (self.height, self.width))
    if self._saving:
      self.save(frame)

    # press "i" or "e" key to turn on or off calibration mode
    # if cv2.waitKey(1) & 0xFF == ord('c'):
    #   self.extrinsic_calibration_switch()
    # if cv2.waitKey(1) & 0xFF == ord('i'):
    #   self.intrinsic_calibration_switch()

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
        # TODO: this part should not be commented?
        self.display()

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
    if self._displaying:
      if not self._in_calibrating:
        print('turning ON intrinsic calibration mode')
        self._in_calibrating = True
        self.in_calib.reset()
      else:
        print('turning OFF intrinsic calibration mode')
        self._in_calibrating = False
        self.in_calib.save_config(self.device_serial_number,
                                  self.width,
                                  self.height)
    else:
      print("not displaying. Can't perform intrinsic calibration")

  def extrinsic_calibration_switch(self):
    if self._displaying:
      if not self._ex_calibrating:
        print('turning ON extrinsic calibration mode')
        self._ex_calibrating = True
        self.ex_calib.reset()
        self.ex_calib.load_config(self.device_serial_number)
      else:
        print('turning OFF extrinsic calibration mode')
        self._ex_calibrating = False
        self.ex_calib.save_config(self.device_serial_number,
                                  self.width,
                                  self.height)
      return self.display()
    else:
      # TODO: return a string to display on webpage
      return "not displaying. Can't perform extrinsic calibration"

  def intrinsic_calibration(self, frame):
    # write something on the frame
    text = 'Intrinsic calibration mode On'
    cv2.putText(frame, text, (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 125), 2)

    # get corners and refine them in openCV for every 3 frames
    if self.in_calib.decimator % 3 == 0:
      corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, self.in_calib.board.dictionary, parameters=self.in_calib.params)
      detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
          cv2.aruco.refineDetectedMarkers(frame, self.in_calib.board, corners, ids,
                                          rejectedImgPoints, parameters=self.in_calib.params)

      # interpolate corners and draw corners
      if len(detectedCorners) > 0:
        rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
            detectedCorners, detectedIds, frame, self.in_calib.board)
        if detectedCorners is not None and 2 <= len(
                detectedCorners) <= self.in_calib.max_size:
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
    # if there isn't configuration on the screen, save corners and ids
    if self.ex_calib.config is None:
      text = 'No configuration file found. Performing initial extrinsic calibration... '
      cv2.putText(frame, text, (50, 50),
                  cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      # calibrate every 3 frames
      if self.ex_calib.decimator % 3 == 0:
        # get parameters
        params = self.ex_calib.params

        # detect corners
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            frame, self.ex_calib.board.dictionary, parameters=params)
        if ids is not None:
          # draw corners on the screen
          cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

          if len(ids) >= len(self.ex_calib.allIds):
            self.ex_calib.allCorners = corners
            self.ex_calib.allIds = ids
    else:
      text = 'Found configuration file for this camera. Calibrating...'
      cv2.putText(frame, text, (50, 50),
                  cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      if True: #self.ex_calib.decimator % 3 == 0:
        truecorners = self.ex_calib.config['corners']  # float numbers
        trueids = self.ex_calib.config['ids']  # int numbers
        CI = self.ex_calib.config['CI']  # int pixels
        markers = self.ex_calib.config['markers']

        # key step: detect markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            frame, self.ex_calib.board.dictionary, parameters=self.ex_calib.params)

        # make sure there are ids and markers and the number of ids is no less than 3
        if cau.check_ids(ids) and cau.check_corners(corners):

          # check if aligned:
          aligns, colors = cau.get_align_color(ids,corners, trueids,truecorners,CI)

          markers['aligns'] = pd.Series(aligns,index=list(map(str,cau.reformat(ids))))
          markers['colors'] = pd.Series(colors,index=list(map(str,cau.reformat(ids))))

          # any way to make it more concise?
          for tid,truecorner in zip(trueids,truecorners):
            real_color = int(markers['colors'][tid]) if pd.notna(markers['colors'][tid]) else 200
            point1 = tuple(np.array(truecorner[0],np.int))
            point2 = tuple(np.array(truecorner[1],np.int))
            point3 = tuple(np.array(truecorner[2],np.int))
            point4 = tuple(np.array(truecorner[3],np.int))
            cv2.line(frame,point1,point2,color=real_color, thickness=CI*2)
            cv2.line(frame, point2, point3, color=real_color, thickness=CI*2)
            cv2.line(frame, point3, point4, color=real_color, thickness=CI*2)
            cv2.line(frame, point4, point1, color=real_color, thickness=CI*2)
          # draw the detected markers on top of the true markers.
          cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

          if all(aligns):
            text = 'Enough corners aligned! Ready to go'
            cv2.putText(frame, text, (500, 1000), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
          else:
            text = "Missing ids or corners!"
            cv2.putText(frame, text, (500, 1000), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
      self.ex_calib.decimator += 1

  def run(self):
    flag = False
    last = 0
    # we will wait a bit less than the interval between frames
    interval = (1 / self.frame_rate) - BUFFER_TIME
    while True:
      pause_time = last + interval - time.time()
      if pause_time > 0:
        time.sleep(pause_time)
      with self._running_lock:
        if self._running:
          last = time.time()
          self.capture()
        else:
          flag = True
      if flag:
        return



  def __del__(self):
    self.stop()


class Nidaq:
  def __init__(self, frame_rate, audio_settings):
    self.audio = None
    self.trigger = None
    self._audio_reader = None
    self.data = None
    self._nBuffers = 10

    self.sample_rate = int(audio_settings['fs'])
    self.trigger_freq = frame_rate
    self.duty_cycle = .01
    self.read_rate = audio_settings['readRate']  # in Hz, depends on PC buffer size...
    self._read_size = self.sample_rate // self.read_rate
    self.read_count = 0

    self._running = False
    self._running_lock = threading.Lock()
    self._displaying = False
    self._data_lock = threading.Lock()
    self._saving = False

    # for display
    self._nfft = int(audio_settings['nFreq'])
    self._window = int(audio_settings['window'] * self.sample_rate)
    self._overlap = int(audio_settings['overlap'] * self._window)
    self._nx = int(np.floor(self.sample_rate-self._overlap)/(self._window-self._overlap))

    # number of calculated timepoints
    self._xq = np.linspace(0, 1, num=self._nx)

    # number of frequency points
    self._yq = np.linspace(0, int(self.sample_rate/2), num=int(self._window/2 + 1))

    # we will use scip.interpolate to convert yq to zq
    if audio_settings['fScale']=='linear':
      self._zq = np.linspace(int(audio_settings['fMin']), int(audio_settings['fMax']), num=int(audio_settings['nFreq']))
    else:
      self._zq = np.logspace(int(np.log10(audio_settings['fMin'])), int(np.log10(audio_settings['fMax'])), num=int(audio_settings['nFreq']))

    self._freq_correct = audio_settings['correction']

    self._frame_bytes = BytesIO()

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
        self.audio.in_stream.input_buf_size = self.sample_rate * \
            60  # buffer on PC in seconds

        self.audio.control(
            nidaqmx.constants.TaskMode.TASK_COMMIT
        )  # transition the task to the committed state so it's ready to start

        self.read_count = 0

        if display:
          self._displaying = True
          self._audio_reader = AnalogReader(
              self.audio.in_stream)
          self._read_size = self.sample_rate // self.read_rate

          self.data = [np.ndarray(shape=(self._read_size))
                       for i in range(self._nBuffers)]

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
              filepath,
              logging_mode=log_mode,
              operation=nidaqmx.constants.LoggingOperation.CREATE_OR_REPLACE)  # see nptdms
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

  def display_switch_on(self):
    if not self._displaying:
      self._displaying=True
      self._audio_reader = AnalogReader(
        self.audio.in_stream)
      self._read_size = self.sample_rate // self.read_rate

      self.data = [np.ndarray(shape=(self._read_size))
                   for i in range(self._nBuffers)]

      log_mode = nidaqmx.constants.LoggingMode.LOG_AND_READ


  def capture(self, read_count):
    if self._displaying:
      # we will save the samples to self.data

      self._audio_reader.read_many_sample(
          # modulo implements a circular buffer
          self.data[self.read_count % self._nBuffers],
          number_of_samples_per_channel=self._read_size
      )

      with self._data_lock:
        self.read_count += 1

    else:
      # not sure... if we're logging, then we do nothing
      # if not logging, will we get an error if we do nothing?
      pass

  def display(self):
    '''
    Calculate the spectrogram of the data and send to connected browsers.
    There are many ways to approach this, in particular by using wavelets or by using
    overlapping FFTs. For now just trying non-overlapping FFTs ~ the simplest approach.
    '''

    flag = False
    if self._displaying:
      with self._data_lock:
        read_count = max(self.read_count - 1, 0)

      while self._displaying:

        last = 0
        # we will wait a bit less than the interval between frames
        interval = (1 / self.read_rate) - BUFFER_TIME

        pause_time = last + interval - time.time()
        if pause_time > 0:
          time.sleep(pause_time)

        with self._data_lock:
          if self.read_count > read_count:
            # note that we're not guaranteed to be gathering sequential reads...

            read_count = self.read_count
            flag = True

        if flag:
          # we ought to extend the sampled data range so that the tails of the spectrogram are accurate with the desired overlap
          # but as the number of windows increases, this probably becomes minor

          _, _, spectrogram = signal.spectrogram(self.data[(
              self.read_count-1) % self._nBuffers], self.sample_rate, nperseg=self._window, noverlap=self._overlap)

          # print(self._xq.shape, self._yq.shape, spectrogram.shape, self._zq.shape)
          respect = interpolate.RectBivariateSpline(self._yq, self._xq, spectrogram)(self._zq, self._xq)

          if self._freq_correct == True:
            respect *= self._zq[:,np.newaxis]
            #corrects for 1/f noise by multiplying with f

          thisMin = np.amin(respect, axis=(0,1))
          respect -= thisMin

          thisMax = np.amax(respect, axis=(0,1))

          respect /= thisMax #normalized to [0,1]

          respect = mpl.cm.viridis(respect) * 255 #colormap

          self._frame_bytes.seek(0)  # go to the beginning of the buffer
          Image.fromarray(respect.astype(np.uint8)).save(self._frame_bytes, 'bmp')
          yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + self._frame_bytes.getvalue() + b'\r\n'
          # socket.emit('fft', {'s': respect.flatten('F').tolist()})

          flag = False

  def run(self):
    # last = 0
    # we will wait a bit less than the interval between frames
    # interval = (1 / self.read_rate) - BUFFER_TIME
    # print(f'nidaq interval = {interval}')
    with self._data_lock:
      read_count = self.read_count

    while True:
      # pause_time = last + interval - time.time()
      # if pause_time > 0:
      #   time.sleep(pause_time)
      # this allows other threads to capture lock in interim
      time.sleep(BUFFER_TIME)
      with self._running_lock:
        if self._running:
          last = time.time()
          self.capture(read_count)
          read_count += 1
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
        print('stopped nidaq')

  def __del__(self):
    self.stop()


class AcquisitionGroup:
  def __init__(self, frame_rate=30, audio_settings=None):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    self.cameras = [Camera(self._camlist, i, frame_rate)
                    for i in range(self.nCameras)]
    self.nidaq = Nidaq(frame_rate, audio_settings)

    self._runners = []

  def start(self, filepaths=None, isDisplayed=None):
    if not filepaths:
      filepaths = [None] * (self.nCameras + 1)
    if not isDisplayed:
      isDisplayed = [False] * (self.nCameras + 1)

    for cam, fp, disp in zip(self.cameras, filepaths[: -1], isDisplayed[: -1]):
      cam.start(filepath=fp, display=disp)
      print('starting camera '+ cam.device_serial_number)

    # once the camera BeginAcquisition methods are called, we can start triggering
    self.nidaq.start(filepath=filepaths[-1], display=isDisplayed[-1])
    print('starting nidaq')


  def run(self):
    # begin gathering samples
    if not self._runners:
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

  def __del__(self):
    for cam in self.cameras:
      del cam
    self._camlist.Clear()
    self._system.ReleaseInstance()
    del self.nidaq


if __name__ == '__main__':
  ag = AcquisitionGroup()
  ag.start(isDisplayed=[True, False])
  ag.run()

# if __name__ == '__main__':
#      board = CharucoBoard(6,2)
#      #board.save_board(2000)
#      cg = AcquisitionGroup()
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
#     #for i, cam in enumerate(cg.cameras):
#     #   cam.stop()
