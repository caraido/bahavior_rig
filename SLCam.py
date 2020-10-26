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
    # here we will eventually want to enable hardware triggering
    # for now we'll just hardcode the framerate at 30
    self._spincam.AcquisitionFrameRateEnable.SetValue(True)
    self._spincam.AcquisitionFrameRate.SetValue(30)

    self.device_serial_number, self.height, self.width = self.get_camera_property()
    self.calib = Calib()

    self._running = False

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
    # filepath, cv2.VideoWriter_fourcc(*'hvc1'), 30, (1024, 1280), False)

    if display:
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

      self._spincam.EndAcquisition()

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
    cv2.putText(frame, text, cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

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
    # cv2.imshow('frame', frame)
    with self._frame_lock:
      frame_count = self.frame_count  # get the starting number of frames
    while True:
      with self._frame_lock:
        if self.frame_count > frame_count:  # display the frame if it's new ~ might run into issues here?
          frame_count = self.frame_count
          self._frame_bytes.seek(0)  # go to the beginning of the buffer
          Image.fromarray(self.frame).save(self._frame_bytes, 'bmp')

          yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + self._frame_bytes.getvalue() + b'\r\n')

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

  def start(self, filepaths, routes):
    for cam, fp, route in zip(self.cameras, filepaths, routes):
      cam.start(filepath=fp, route=r)

  def stop(self):
    for cam in self.cameras:
      cam.stop()

  def __del__(self):
    for cam in self.cameras:
      del cam
    self._camlist.Clear()
    self._system.ReleaseInstance()


if __name__ == '__main__':
  board = CharucoBoard()
  board.save_board()
  cg = CameraGroup()
  for i, cam in enumerate(cg.cameras):
    cam.start(filepath=f'testing{i:02d}.mov')

  for j in range(100):
    for i, cam in enumerate(cg.cameras):
      cam.capture()

  for i, cam in enumerate(cg.cameras):
    cam.stop()
