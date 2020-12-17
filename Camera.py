import cv2
import PySpin
import numpy as np
import ffmpeg
import utils.calibration_utils as cau
from utils.calibration_utils import Calib
import threading
from io import BytesIO
from PIL import Image
import time
import pandas as pd
from dlclive import DLCLive, Processor

BUFFER_TIME = .005  # time in seconds allowed for overhead
FRAME_TIMEOUT = 100  # time in milliseconds to wait for pyspin to retrieve the frame
DLC_RESIZE = 0.6  # resize the frame by this factor for DLC
DLC_UPDATE_EACH = 3  # frame interval for DLC update


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
    self._interval = (1 / self.frame_rate) - BUFFER_TIME

    self.device_serial_number, self.height, self.width = self.get_camera_properties()
    self.size = (self.width, self.height)

    self.in_calib = Calib('intrinsic')
    self.ex_calib = Calib('extrinsic')

    self._running_lock = threading.Lock()
    self._running=False
    self.running = False

    self._file_lock = threading.Lock()
    self._file=None
    self.file = None

    self._frame_lock = threading.Lock()
    self._frame=None
    self.frame = None

    self._in_calibrating = False
    self._ex_calibrating = False

    self._has_runner = False

    self._dlc_lock = threading.Lock()
    self.dlc = False
    # self._save_dlc = False
    # self._dlc_count = None
    # self.dlc_proc = None
    # self.dlc_live = None

  # NOTE: a property accessed by multiple threads may need a lock, if the property guarantees another
  # these include [method | threads | guarantee]:
  #   self.running  | main and runner                   | tells whether spincam is running
  #   self.file     | main and runner                   | tells whether ffmpeg file is open
  #   self.frame    | main, runner, display, calib, dlc | frame can be read/written without collision if exists
  #   self.dlc      | main, dlc                         | tells whether dlc objects exist

  @property
  def running(self):
    with self._running_lock:
      return self._running

  @running.setter
  def running(self, running):
    if isinstance(running, bool):
      if running:
        with self._running_lock:
          # may need to check first that self._running is false, but slower
          self._spincam.BeginAcquisition()
          self._running = True
      else:
        with self._running_lock:
          try:
            self._spincam.EndAcquisition()
          except:
            print("EndAcquisition called before BeginAcquisition")
          self._running = False
    else:
      with self._running_lock:
        if self._running:
          return True, time.time(), running.next()
        else:
          return False, None, None

  @property
  def file(self):
    with self._file_lock:
      return self._file

  @file.setter
  def file(self, filepath):
    if filepath is not None:
      with self._file_lock:
        self._file = ffmpeg \
            .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{self.width}x{self.height}') \
            .output(filepath, vcodec='libx265') \
            .overwrite_output() \
            .run_async(pipe_stdin=True)
    else:
      with self._file_lock:
        if self._file is not None:
          self._file.stdin.close()
          self._file.wait()
          del self.file
          self._file = None

  @property
  def frame(self):
    with self._frame_lock:
      return self._frame

  @property
  def frame_count(self):
    with self._frame_lock:
      return self._frame_count

  @property
  def frame_and_count(self):
    with self._frame_lock:
      return self._frame, self._frame_count

  @frame.setter
  def frame(self, frame):
    if isinstance(frame, bool):
      if frame:
        with self._frame_lock:
          self._frame = np.empty(self.size)
          self._frame_count = 0
      else:
        with self._frame_lock:
          self._frame = None
          self._frame_count = 0
    else:
      with self._frame_lock:
        if self._frame is not None:
          self._frame = frame
          self._frame_count += 1

  @property
  def dlc(self):
    with self._dlc_lock:
      if self._dlc:
        return self._dlc_live
      else:
        return False

  @dlc.setter  # usage: to turn on, call self.dlc = modelpath ; to turn off, call self.dlc = False
  def dlc(self, modelpath):
    if modelpath:
      with self._dlc_lock:
        # TODO: if any of these functions are slow, then call them outside of the lock and reassign them to member variables inside the lock
        # TODO: do we need to check if self._dlc is already true? if so, do we close the existing one and make a new one with the new model path???
        self._dlc_proc = Processor()
        self._dlc_live = DLCLive(
            model_path=modelpath, processor=self._dlc_proc, display=False, resize=DLC_RESIZE)
        self._dlc_first_frame = True
        self._dlc = True
    else:
      with self._dlc_lock:
        self._dlc_proc = None
        try:
          self._dlc_live.close()
        except:
          pass
        self._dlc_first_frame = None
        self._dlc = False

  def start(self, filepath=None, display=False):
    self.file = filepath
    self.frame = display  # we will start storing frames
    self.running = True

  def stop(self):
    self.running = False
    self.file = False
    self.frame = False
    self.dlc = False

  def capture(self):
    frame = np.empty(self.size)  # preallocate, should speed up a bit
    while True:
      # get the image from spinview
      im = self._spincam.GetNextImage(FRAME_TIMEOUT)
      if im.IsIncomplete():
        status = im.GetImageStatus()
        im.Release()
        raise Exception(f"Image incomplete with image status {status} ...")
      # frame = np.reshape(im.GetData(), self.size)
      frame = im.GetNDArray()  # TODO: check that this works!!

      # TODO: move this to self.display()
      # text = 'recording...'
      # cv2.putText(frame, text, (700, 50),
      #             cv2.FONT_HERSHEY_PLAIN, 4.0, 0, 2)

      # TODO: move calibration stuff to own function running on separate process, accessing self.frame
      # check calibration status
      # if self._in_calibrating and self._ex_calibrating:
      #   raise Warning('Only one type of calibration can be turned on!')

      # # intrinsic calibration
      # if self._in_calibrating and not self._ex_calibrating:
      #   self.intrinsic_calibration(frame)

      # # extrinsic calibration
      # if self._ex_calibrating and not self._in_calibrating:
      #   self.extrinsic_calibration(frame)

      # TODO: check this and move to run_dlc() function running on separate process
      # last_count = 0
      # last_frame_time = time.time() - self._interval
      # while True:
      #   self.sleep(last_frame_time)
      #   frame_count, count = self.frame_and_count
      #   if frame is not None and count > last_count:
      #     last_count = frame_count
      #     last_frame_time = time.time()
      #     with self._dlc_lock:
      #       if self._dlc_count: do init and update count
      #       if not frame_count % DLC_UPDATE_EACH:
      #         self.dlc_live.get_pose(frame) #is this synchronous?
      #         self.last_pose = self.dlc_live.pose
      #    elif frame is None or not self.dlc: #double check this logic
      #       return

      # old:
      # if self._dlc_count:
      #   self.dlc_live.init_inference(frame)
      #   self._dlc_count = None
      # if self.frame_count % 3 == 0:
      #   self.dlc_live.get_pose(frame)
      #   pose = self.dlc_live.pose
      #   idu.draw_dots(frame, pose)

      im.Release()
      yield frame

  def save(self, frame):
    self._file.stdin.write(frame.tobytes())

  def get_camera_properties(self):
    nodemap_tldevice = self._spincam.GetTLDeviceNodeMap()
    device_serial_number = PySpin.CStringPtr(
        nodemap_tldevice.GetNode('DeviceSerialNumber')).GetValue()
    nodemap = self._spincam.GetNodeMap()
    height = PySpin.CIntegerPtr(nodemap.GetNode('Height')).GetValue()
    width = PySpin.CIntegerPtr(nodemap.GetNode('Width')).GetValue()
    return device_serial_number, height, width

  def intrinsic_calibration_switch(self):
    # TODO: with self._displaying_lock:
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
      # return self.display()

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

  def display(self):
    frame_bytes = BytesIO()
    last_count = 0
    frame, frame_count = self.frame_and_count
    last_frame_time = time.time()  # - self._interval #TODO: check this
    while frame is not None:
      # TODO: draw on the frame, using e.g. self._pose and self.ex_calib, etc.
      if frame_count > last_count:  # display the frame if it's new
        last_count = frame_count
        last_frame_time = time.time()
        frame_bytes.seek(0)  # go to the beginning of the buffer
        Image.fromarray(frame).save(frame_bytes, 'bmp')
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes.getvalue() + b'\r\n')
      else:
        self.sleep(last_frame_time)
      frame, frame_count = self.frame_and_count

  def extrinsic_calibration(self, frame):
    # if there isn't configuration on the screen, save corners and ids
    if self.ex_calib.config is None:
      text = 'No configuration file found. Performing initial extrinsic calibration... '
      cv2.putText(frame, text, (50, 50),
                  cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      # calibrate every 3 frames
      if self.ex_calib.decimator % 3 == 0:  # TODO: move to constant at top of file
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

  def run(self):  # repeatedly calls self.capture() while self.running
    self._has_runner = True  # register the runner

    last_frame = np.empty(self.size)
    capture = self.capture()  # returns a generator function, can call .next() method
    last_frame_time = time.time() - self._interval
    while True:
      self.sleep(last_frame_time)
      is_running, last_frame_time, last_frame = self.running(capture)
      if not is_running:
        self._has_runner = False
        return

      with self._file_lock:
        if self._file is not None:
          self.save(last_frame)
      self.frame = last_frame  # write the frame to the instance buffer

  def sleep(self, since):
    # current_time - previous_time === time_elapsed
    # self._interval - time_elapsed === pause_time
    pause_time = since + self._interval - time.time()
    if pause_time > 0:
      time.sleep(pause_time)

  def __del__(self):
    self.stop()
    while self._has_runner:
      last_check_time = time.time()
      self.sleep(last_check_time)
    self._spincam.DeInit()
