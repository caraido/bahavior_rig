import cv2
import PySpin
import numpy as np
from scipy import signal, interpolate
from scipy import io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import ffmpeg
import utils.calibration_utils as cau
import utils.image_draw_utils as idu
import utils.extrinsic as extrinsic
import os
import toml
import threading
from io import BytesIO
from PIL import Image
import nidaqmx
from nidaqmx.stream_readers import AnalogSingleChannelReader as AnalogReader
import time
import pandas as pd
from dlclive import DLCLive, Processor
from audio_processing import read_audio
import pyaudio
import nptdms
import scipy.io.wavfile as wavfile
import pickle as pk

BUFFER_TIME = .005  # time in seconds allowed for overhead
CHUNK=1024
default_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig'

class CharucoBoard:
  def __init__(self, x, y, marker_size=0.8,type=None):
    self.x = x
    self.y = y
    self.marker_size = marker_size
    #self.default_dictionary = cv2.aruco.DICT_4X4_50  # default
    self.default_dictionary = type
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

  @property
  def default_dictionary(self):
    return self._default_dictionary

  @default_dictionary.setter
  def default_dictionary(self, type):
    if type is None:
      self._default_dictionary = cv2.aruco.DICT_4X4_50
    elif type == 'intrinsic' or type == 'extrinsic_3d':
      self._default_dictionary = cv2.aruco.DICT_5X5_50
    elif type == 'extrinsic':
      self._default_dictionary = cv2.aruco.DICT_4X4_50
    else:
      raise ValueError('wrong type')

  def save_board(self, img_size=1000):
    file_name = 'charuco_board_shape_%dx%d_marker_size_%d_default_%d.png' % (
    self.x, self.y, self.marker_size, self.default_dictionary)
    img = self.board.draw((img_size, img_size))
    result = cv2.imwrite('./multimedia/board/' + file_name, img)
    if result:
      print('save board successfully! Name: ' + file_name)
    else:
      raise Exception('save board failed! Name: ' + file_name)


  def print_board(self):
    img = self.board.draw((1000, 1000))
    plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
    plt.axis("off")
    plt.show()


class Calib:
  def __init__(self, calib_type):
    self._get_type(calib_type)

    self.root_config_path = None

    self.allCorners = []
    self.allIds = []
    # for 3d calibration
    self.rvec=[]
    self.tvec=[]
    self.success=[]
    self.rejectedImgPoints = []
    self.decimator = 0
    self.config = None

    self.charuco_board = CharucoBoard(x=self.x, y=self.y, type=self.type)
    self.board = self.charuco_board.board

    self.max_size = cau.get_expected_corners(self.board)
    self.load_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'

  def _get_type(self, calib_type):
    if calib_type == 'extrinsic':
      self.type = calib_type
      self.x = 6
      self.y = 2
    elif calib_type == 'intrinsic':
      self.type = calib_type
      self.x = 4
      self.y = 5
    elif calib_type == 'extrinsic_3d':
      self.type = calib_type
      self.x = 3
      self.y = 5
    else:
      raise ValueError("type can only be intrinsic or extrinsic!")

  def reset(self):
    del self.allIds, self.allCorners, self.decimator, self.config
    self.allCorners = []
    self.allIds = []
    self.rvec = []
    self.tvec = []
    self.success = []
    self.rejectedImgPoints=[]
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

  @property
  def root_config_path(self):
    return self._root_config_path

  @root_config_path.setter
  def root_config_path(self, path):
    if path is not None:
      if os.path.exists(path):
        try:
          os.mkdir(os.path.join(path, 'config'))  # is it needed?
        except:
          pass
        self._root_config_path = os.path.join(path, 'config')
      else:
        raise FileExistsError("root file folder doens't exist!")
    else:
      self._root_config_path = None

  # load configuration only for extrinsic calibration
  def load_ex_config(self, camera_serial_number):
    items = os.listdir(self.load_path)
    for item in items:
      if camera_serial_number in item and self.type in item:
        path = os.path.join(self.load_path, 'config_%s_%s.toml' % (self.type, camera_serial_number))
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
            print("Missing ids/corners/markers in the configuration file. Please check.")

  # load intrinsic configuration only for 3d calibration
  def load_in_config(self, camera_serial_number):
    items = os.listdir(self.load_path)
    for item in items:
      if camera_serial_number in item and 'intrinsic' in item:
        path = os.path.join(self.load_path, 'config_%s_%s.toml' % (self.type, camera_serial_number))
        with open(path, 'r') as f:
          # there only should be only one calib file for each camera
          self.config = toml.load(f)

  # check the existence of intrinsic calibration configuration for 3D calibration
  def check_intrinsics(self,camera_serial_number):
    items = os.listdir(self.load_path)
    top=None
    side = None

    for i in items:
      if camera_serial_number in i and 'intrinsic' in i:
        side = os.path.join(self.load_path, i)
      if '17391304' in i and 'intrinsic' in i:
        top = os.path.join(self.load_path,i)
    if top and side:
      return top,side
    else:
      print('missing intrinsic calibration file for either top camera or side camera ')


  def save_config(self, camera_serial_number, width, height):
    save_path = os.path.join(self.root_config_path, 'config_%s_%s.toml' % (self.type, camera_serial_number))
    save_copy_path = os.path.join(self.load_path,'config_%s_%s.toml' % (self.type, camera_serial_number))  # overwrite

    if os.path.exists(save_path):
      return 'Configuration file already exists.'
    else:
      if self.type == "intrinsic":
        # time consuming
        param = cau.quick_calibrate(self.allCorners,
                                self.allIds,
                                self.board,
                                width,
                                height)
        param['camera_serial_number'] = camera_serial_number
        param['date'] = time.strftime("%Y-%m-%d-_%H:%M:%S", time.localtime())
        if len(param) > 2:
          with open(save_path, 'w') as f:
            toml.dump(param, f)
          # save a copy to the configuration folder. Overwrite the previous one
          with open(save_copy_path, 'w') as f:
            toml.dump(param, f)

          return "intrinsic calibration configuration saved!"
        else:
          return "intrinsic calibration configuration NOT saved due to lack of markers."
      elif self.type=='extrinsic':
        if self.allIds is not None and not len(self.allIds) < self.max_size + 1:
          param = {'corners': np.array(self.allCorners),
                   'ids': np.array(self.allIds), 'CI': 5,
                   'camera_serial_number': camera_serial_number,
                   'date': time.strftime("%Y-%m-%d-_%H:%M:%S", time.localtime())}
          with open(save_path, 'w') as f:
            toml.dump(param, f, encoder=toml.TomlNumpyEncoder())
          with open(save_copy_path, 'w') as f:
            toml.dump(param, f, encoder=toml.TomlNumpyEncoder())

          return 'extrinsic calibration configuration saved!'
        else:
          return "failed to record all Ids! Can't save configuration. Please calibrate again."

  def save_config_3d(self,camera_serial_number,param,error):
    save_path = os.path.join(self.root_config_path, 'config_%s_%s.toml' % (self.type, camera_serial_number))
    save_copy_path = os.path.join(self.load_path, 'config_%s_%s.toml' % (self.type, camera_serial_number))  # overwrite

    if os.path.exists(save_path):
      return 'Configuration file already exists.'
    else:
      if self.type=='extrinsic_3d':

        param['error']=error
        param['camera_serial_number']=camera_serial_number
        with open(save_path, 'w') as f:
          toml.dump(param, f, encoder=toml.TomlNumpyEncoder())
        with open(save_copy_path, 'w') as f:
          toml.dump(param, f, encoder=toml.TomlNumpyEncoder())



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
    self.ex_3d_calib=Calib('extrinsic_3d')

    self._start = False

    self._running = False
    self._running_lock = threading.Lock()

    self._saving = False
    self.file = None
    self.filepath=None

    self._displaying = False
    self.frame = None
    self.frame_count = 0
    self._frame_lock = threading.Lock()
    self._frame_bytes = BytesIO()

    self._in_calibrating = False
    self._ex_calibrating = False
    self._ex_3d_calibrating=False
    self.intrinsic_file=None

    self._dlc = False
    self._save_dlc = False
    self._dlc_count= None
    self.dlc_proc = None
    self.dlc_live = None


  def start(self, filepath=None, display=False):
    self.filepath=filepath
    if filepath:
      rootpath = os.path.split(filepath)[0]
      self.in_calib.root_config_path=rootpath
      self.ex_calib.root_config_path=rootpath
      self.ex_3d_calib.root_config_path=rootpath
      #self._saving = True

      # we will assume hevc for now
      # will also assume 30fps
      # '1280x1024' can be replaced by self.width and self.height
      self.file = ffmpeg \
          .input('pipe:', format='rawvideo', pix_fmt='gray', s='1280x1024') \
          .output(filepath, vcodec='libx265') \
          .overwrite_output() \
          .run_async(pipe_stdin=True)

    #else:
    #  self._saving = False

    self.frame_count = 0

    if display:
      self._displaying = True
    else:
      self._displaying = False

    with self._running_lock:
      if not self._running:
        self._running = True
        self._spincam.BeginAcquisition()

  def saving_switch_on(self):
    if not self._saving:
      if self.file is not None:
        self._saving = True
      else:
        raise FileNotFoundError("file path is not found!")
    else:
      self._saving = False
      self.file.stdin.close()
      self.file.wait()
      del self.file
      self.file = None

  def dlc_switch(self,model_path=None):
    if not self._dlc:
      self.dlc_proc = Processor()
      if model_path:
        # TODO: displays should be False
        self.dlc_live = DLCLive(model_path=model_path,processor=self.dlc_proc,display=False,resize=0.6)
        self._dlc = True
        self._dlc_count = 1
    else:
      self.dlc_live.close()
      self._dlc = False
      self._dlc_count = None

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
        self.dlc_switch()

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
      text = 'recording...'
      cv2.putText(frame, text, (700,50),
                  cv2.FONT_HERSHEY_PLAIN, 4.0, 0, 2)

    # check calibration status
    check1=self._in_calibrating and self._ex_calibrating
    check2=self._in_calibrating and self._ex_3d_calibrating
    check3=self._ex_calibrating and self._ex_3d_calibrating
    if check1 or check2 or check3:
      raise Warning('Only one type of calibration can be turned on!')

    # intrinsic calibration
    if self._in_calibrating and not self._ex_calibrating and not self._ex_3d_calibrating:
      self.intrinsic_calibration(frame)

    # extrinsic calibration
    if self._ex_calibrating and not self._in_calibrating and not self._ex_3d_calibrating:
      self.extrinsic_calibration(frame)

    # 3d extrinsic calibration
    if self._ex_3d_calibrating and not self._in_calibrating and not self._ex_calibrating:
      self.extrinsic_3d_calibration(frame)

    if self._dlc:
      if self._dlc_count:
        self.dlc_live.init_inference(frame)
        self._dlc_count = None
      if self.frame_count%3 == 0:
        self.dlc_live.get_pose(frame)
        pose=self.dlc_live.pose
        idu.draw_dots(frame,pose)

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
        self.ex_calib.load_ex_config(self.device_serial_number)
      else:
        print('turning OFF extrinsic calibration mode')
        self._ex_calibrating = False
        self.ex_calib.save_config(self.device_serial_number,
                                  self.width,
                                  self.height)

  def display(self):
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


  def intrinsic_calibration(self, frame):
    # write something on the frame
    text = 'Intrinsic calibration mode On'
    cv2.putText(frame, text, (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 125), 2)

    # get corners and refine them in openCV for every 3 frames
    if self.in_calib.decimator % 3 == 0:  # TODO: move 3 to a constant at top of file
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


  def extrinsic_calibration(self, frame):
    # if there isn't configuration on the screen, save corners and ids
    if self.ex_calib.config is None:
      text = 'No configuration file found. Performing initial extrinsic calibration... '
      cv2.putText(frame, text, (50, 50),
                  cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      # calibrate every 3 frames
      if self.ex_calib.decimator % 3 == 0: #TODO: move to constant at top of file
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

      if True:  # self.ex_calib.decimator % 3 == 0:
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
          aligns, colors = cau.get_align_color(
              ids, corners, trueids, truecorners, CI)

          markers['aligns'] = pd.Series(
              aligns, index=list(map(str, cau.reformat(ids))))
          markers['colors'] = pd.Series(
              colors, index=list(map(str, cau.reformat(ids))))

          # any way to make it more concise?
          for tid, truecorner in zip(trueids, truecorners):
            real_color = int(markers['colors'][tid]) if pd.notna(
                markers['colors'][tid]) else 200
            point1 = tuple(np.array(truecorner[0], np.int))
            point2 = tuple(np.array(truecorner[1], np.int))
            point3 = tuple(np.array(truecorner[2], np.int))
            point4 = tuple(np.array(truecorner[3], np.int))
            cv2.line(frame, point1, point2, color=real_color, thickness=CI*2)
            cv2.line(frame, point2, point3, color=real_color, thickness=CI*2)
            cv2.line(frame, point3, point4, color=real_color, thickness=CI*2)
            cv2.line(frame, point4, point1, color=real_color, thickness=CI*2)
          # draw the detected markers on top of the true markers.
          cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

          if all(aligns):
            text = 'Enough corners aligned! Ready to go'
            cv2.putText(frame, text, (500, 1000),
                        cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
          else:
            text = "Missing ids or corners!"
            cv2.putText(frame, text, (500, 1000),
                        cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
      self.ex_calib.decimator += 1


  def extrinsic_3d_calibration(self,frame):
    if self.ex_3d_calib.config is None:
      text = 'Performing 3D extrinsic calibration... '
      cv2.putText(frame, text, (50, 50),
                  cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      # detect corners
      corners,ids=extrinsic.detect_aruco_2(frame, intrinsics=self.ex_3d_calib.config,params=self.ex_3d_calib.params,board=self.ex_3d_calib.board)

      #success, result=extrinsic.estimate_pose_aruco(frame,self.intrinsic_file,self.ex_3d_calib.board)

      self.ex_3d_calib.allCorners.append(corners)
      self.ex_3d_calib.allIds.append(ids)
      #self.ex_3d_calib.tvec.append(tvec)
      #self.ex_3d_calib.rvec.append(rvec)
      #self.ex_3d_calib.success.append(success)


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
    # in Hz, depends on PC buffer size...
    self.read_rate = audio_settings['readRate']
    self._read_size = self.sample_rate // self.read_rate
    self.read_count = 0

    self._running = False
    self._running_lock = threading.Lock()
    self._displaying = False
    self._data_lock = threading.Lock()
    self._saving = False
    self.filepath = None
    self.log_mode = nidaqmx.constants.LoggingMode.LOG_AND_READ

    ## for display
    self._nfft = int(audio_settings['nFreq'])
    self._window = int(audio_settings['window'] * self.sample_rate)
    self._overlap = int(audio_settings['overlap'] * self._window)
    self._nx = int(np.floor(self.sample_rate-self._overlap) /
                   (self._window-self._overlap))

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

  def start(self, filepath=None, display=True):
    with self._running_lock:
      if not self._running:
        # audio task
        self.audio = nidaqmx.Task()
        self.audio.ai_channels.add_ai_voltage_chan(
            "Dev1/ai1"
        )  # this channel measures the audio signal

        # self.audio.ai_channels.ai_microphone_sensitivity=100 # doesn't know if it works
        self.audio.ai_channels['Dev1/ai1'].ai_gain= 10000
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

          self.log_mode = nidaqmx.constants.LoggingMode.LOG_AND_READ
        else:
          self._displaying = False
          self._audio_reader = None
          self._read_size = None

          self.data = None

          self.log_mode = nidaqmx.constants.LoggingMode.LOG

        self.audio.in_stream.configure_logging(
          'C:\\Users\\SchwartzLab\\Desktop\\unwanted.tdms',
          logging_mode=self.log_mode,
          operation=nidaqmx.constants.LoggingOperation.CREATE_OR_REPLACE)  # see nptdms
        '''
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
        '''
        self._saving=False
        self.filepath=filepath

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

  def saving_switch_on(self):
    if not self._saving:
      self._saving=True
      if self.filepath:
        self.audio.in_stream.start_new_file(self.filepath)
      else:
        # TODO: will set a default path to save
        raise FileNotFoundError("file path is not found!")
    #else:
    #  self._saving=False
    #  self.audio.in_stream.logging_mode = nidaqmx.constants.LoggingMode.OFF

  def display_switch_on(self):
    if not self._displaying:
      self._displaying=True
      self._audio_reader = AnalogReader(
        self.audio.in_stream)
      self._read_size = self.sample_rate // self.read_rate

      self.data = [np.ndarray(shape=(self._read_size))
                   for i in range(self._nBuffers)]

  def capture(self,read_count):
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
      try:
        os.remove('C:\\Users\\SchwartzLab\\Desktop\\unwanted.tdms_index')
        os.remove('C:\\Users\\SchwartzLab\\Desktop\\unwanted.tdms')
      finally:
        # save .wav
        audio, _ = read_audio(self.filepath)
        wavfile.write(self.filepath[:-4] + 'wav', self.sample_rate, audio)
        #sio.savemat(self.filepath[:-4] + 'mat', {'audio': audio, 'sample_rate': self.sample_rate})
        print('save nidaq')

  def __del__(self):
    self.stop()

class Mic:
  def __init__(self,audio_settings):
    self.audio=None
    self.data=None
    self.index=None
    self.stream=None
    self._nBuffers=10

    self.sample_rate = int(audio_settings['fs'])
    self.duty_cycle=.01

    self.read_rate=audio_settings['readRate']
    self._read_size=self.sample_rate//self.read_rate
    self.read_count = 0
    self._running_lock=threading.Lock()

    self._running=False
    self._displaying=False
    self._data_lock=threading.Lock()
    self._saving=False
    self.filepath=None

    self.group_name = 'Dev1/ai2'
    self.channel_name = 'channel_0'
    self.saving_switch=False

    self.channels=1
    self.format = pyaudio.paFloat32

    ## for display
    self._nfft = int(audio_settings['nFreq'])
    self._window = int(audio_settings['window'] * self.sample_rate)
    self._overlap = int(audio_settings['overlap'] * self._window)
    self._nx = int(np.floor(self.sample_rate - self._overlap) /
                   (self._window - self._overlap))


    # number of calculated timepoints
    self._xq = np.linspace(0, 1, num=self._nx)

    # number of frequency points
    self._yq = np.linspace(0, int(self.sample_rate / 2), num=int(self._window / 2 + 1))

    # we will use scip.interpolate to convert yq to zq
    if audio_settings['fScale'] == 'linear':
      self._zq = np.linspace(int(audio_settings['fMin']), int(audio_settings['fMax']), num=int(audio_settings['nFreq']))
    else:
      self._zq = np.logspace(int(np.log10(audio_settings['fMin'])), int(np.log10(audio_settings['fMax'])),
                             num=int(audio_settings['nFreq']))

    self._freq_correct = audio_settings['correction']

    self._frame_bytes = BytesIO()

  def capture(self,in_data, frame_count, time_info, status):
    self.data = np.fromstring(in_data, dtype=np.float32)
    if self.saving_switch:
      data_chunk = nptdms.ChannelObject(self.group_name,
                                        self.channel_name,
                                        self.data,
                                        properties={})
      if self.filepath is not None:
        with nptdms.TdmsWriter(self.filepath,'a') as writer:
          writer.write_segment([data_chunk])

    with self._data_lock:
      self.read_count+=1
    return (self.data, pyaudio.paContinue)

  def start(self,filepath=None, display=True):
    with self._running_lock:
      if not self._running:
        self.audio=pyaudio.PyAudio()
        for devices in range(self.audio.get_device_count()):
          info = self.audio.get_device_info_by_index(devices)
          if 'UltraMic' in info['name']:
            self.index = devices
        if self.index is not None:
          self.stream=self.audio.open(format=self.format,
                          channels=self.channels,
                          input_device_index=self.index,
                          frames_per_buffer=self._read_size,
                          rate=self.sample_rate,
                          stream_callback=self.capture,
                          input=True
                          )
        else:
          raise ModuleNotFoundError("can't detect the desired mic!")

        if display:
          self._displaying=True
        else:
          self._displaying=False
          self.data=None

        self._saving=False
        self.filepath=filepath
        self._running=True


  def run(self):
    self.data = np.empty((self._read_size), dtype="float32")
    self.stream.start_stream()
    while self.stream.is_active():
      time.sleep(BUFFER_TIME)
      with self._running_lock:
        if self._running:
          with self._data_lock:
            self.read_count+=1
        else:
          self.stream.stop_stream()
          return

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
          size=(self.read_count-1) % self._nBuffers
          _, _, spectrogram = signal.spectrogram(self.data, self.sample_rate, nperseg=self._window, noverlap=self._overlap)

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

          flag = False

  def stop(self):
    with self._running_lock:
      if self._running:
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self._running = False
        self._displaying = False
        self._saving = False

      audio,_=read_audio(self.filepath)
      #sio.savemat(self.filepath[:-4]+'mat',{'audio': audio, 'sample_rate': self.sample_rate})
      wavfile.write(self.filepath[:-4]+'wav',self.sample_rate,audio)
      print('save USB mic')


  def _del__(self):
    self.stop()


class AcquisitionGroup:
  def __init__(self, frame_rate=30, audio_settings=None):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    self.cameras = [Camera(self._camlist, i, frame_rate)
                    for i in range(self.nCameras)]
    self.nidaq = Nidaq(frame_rate, audio_settings)
    self.mic = Mic(audio_settings=audio_settings)

    self._runners = []
    self.filepaths = None

  def start(self, isDisplayed=True):
    if not self.filepaths:
      self.filepaths = [None] * (self.nCameras + 2)
    if not isDisplayed:
      isDisplayed = [False] * (self.nCameras + 2)

    if not isinstance(isDisplayed, list) or len(isDisplayed) == 1:
      isDisplayed = [isDisplayed] * (self.nCameras + 2)

    print('detected %d cameras' % self.nCameras)

    for cam, fp, disp in zip(self.cameras, self.filepaths[: -2], isDisplayed[: -2]):
      cam.start(filepath=fp, display=disp)
      print('starting camera ' + cam.device_serial_number)

    # once the camera BeginAcquisition methods are called, we can start triggering
    self.nidaq.start(filepath=self.filepaths[-2],display=isDisplayed[-2])
    print('starting nidaq')

    self.mic.start(filepath=self.filepaths[-1], display=isDisplayed[-1])
    print('starting ultramic')

  def ex_3d_calibration_switch(self, side, top='17391304'):
    side_cam=None
    top_cam=None
    for cam in self.cameras:
      if cam.device_serial_number==side:
        side_cam=cam
      if cam.device_serial_number==top:
        top_cam=cam

    if side_cam and top_cam and side_cam.ex_3d_calib.check_intrinsics(side):
      top_intrinsic, side_intrinsic = side_cam.ex_3d_calib.check_intrinsics(side)
      top_cam.intrinsic_file = toml.load(top_intrinsic)
      side_cam.intrinsic_file = toml.load(side_intrinsic)
      intrinsic_file=[top_cam.intrinsic_file,side_cam.intrinsic_file]
      if not side_cam._ex_3d_calibrating and not top_cam._ex_3d_calibrating:
        print('turning ON 3D extrinsic calibration mode')
        side_cam._ex_3d_calibrating = True
        top_cam._ex_3d_calibrating = True
      else:
        side_cam._ex_3d_calibrating = False
        top_cam._ex_3d_calibrating = False
        cam_indices=[0,1]
        ids=[top_cam.ex_3d_calib.allIds,side_cam.ex_3d_calib.allIds]
        corner = [top_cam.ex_3d_calib.allCorners, side_cam.ex_3d_calib.allCorners]
        extrinsic_param,error=extrinsic.get_extrinsics_2(cam_indices,
                                                   ids_list=ids,
                                                   corners_list=corner,
                                                   intrinsics_list=intrinsic_file,
                                                   cam_align=0,# top camera
                                                   board=top_cam.ex_3d_calib.board, skip=40)
        side_cam.ex_3d_calib.save_config_3d(side,extrinsic_param,error)
        print('3d camera calibration finished on %s and %s'%(top,side))
        top_cam.ex_3d_calib.reset()

  def run(self):
    # begin gathering samples
    if not self._runners:  # if self._runners == []
      for i, cam in enumerate(self.cameras):
        self._runners.append(threading.Thread(target=cam.run))
        self._runners[i].start()
      self._runners.append(threading.Thread(target=self.nidaq.run))
      self._runners[-1].start()
      self._runners.append(threading.Thread(target=self.mic.run))
      self._runners[-1].start()

    else:
      for i, cam in enumerate(self.cameras):
        if not self._runners[i].is_alive():
          self._runners[i] = threading.Thread(target=cam.run)
          self._runners[i].start()

      if not self._runners[-2].is_alive():
        self._runners[-2] = threading.Thread(target=self.nidaq.run)
        self._runners[-2].start()

      if not self._runners[-1].is_alive():
        self._runners[-1] = threading.Thread(target=self.nidaq.run)
        self._runners[-1].start()

  def stop(self):
    for cam in self.cameras:
      cam.stop()
    self.nidaq.stop()  # make sure cameras are stopped before stopping triggers
    self.mic.stop()
    # save .mat
    #audio = read_audio(self.filepaths[-1])
    #sio.savemat(self.filepaths[-1] + 'audio.mat', {'audio': audio, 'sample_rate': self.nidaq.sample_rate})
    #print('save nidaq')

  def __del__(self):
    for cam in self.cameras:
      del cam
    self._camlist.Clear()
    self._system.ReleaseInstance()
    del self.nidaq
    del self.mic





if __name__ == '__main__':
  #ag = AcquisitionGroup()
  #ag.start()
  #ag.run()

  calib=Calib('intrinsic')
  calib.charuco_board.save_board()

