import cv2
import PySpin
import numpy as np
import ffmpeg
import utils.calibration_utils as cau
from utils.calibration_utils import Calib
import pandas as pd
from dlclive import DLCLive, Processor
from AcquisitionObject import AcquisitionObject
from utils.image_draw_utils import draw_dots


FRAME_TIMEOUT = 100  # time in milliseconds to wait for pyspin to retrieve the frame
DLC_RESIZE = 0.6  # resize the frame by this factor for DLC
DLC_UPDATE_EACH = 3  # frame interval for DLC update


class Camera(AcquisitionObject):

  def __init__(self, camlist, index, frame_rate):
    self._spincam = camlist.GetByIndex(index)
    self._spincam.Init()

    # hardware triggering
    self._spincam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    # trigger has to be off to change source
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    self._spincam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_On)

    self.device_serial_number, self.height, self.width = self.get_camera_properties()

    AcquisitionObject.__init__(self, frame_rate, (self.width, self.height))

    # self._in_calibrating = False
    # self._ex_calibrating = False

  def prepare_processing(self, options):
    process = {}

    if options['mode'] == 'DLC':
      # process['modelpath'] = options
      process['mode'] = 'DLC'
      process['processor'] = Processor()
      process['DLCLive'] = DLCLive(
          model_path=options['modelpath'], processor=process['processor'], display=False, resize=DLC_RESIZE)
      process['frame0'] = True
      return process
    else:  # mode should be 'intrinsic' or 'extrinsic'
      process['mode'] = options['mode']

      # could move this to init if desired
      process['calibrator'] = Calib(options['mode'])

      process['calibrator'].reset()
      if options['mode'] == 'extrinsic':
        process['calibrator'].load_config(self.device_serial_number)

  def end_processing(self, process):
    if process['mode'] == 'DLC':
      process['DLCLive'].close()
      process['frame0'] = False
    else:
      process['calibrator'].save_config(
          self.device_serial_number, self.width, self.height)
      del process['calibrator']  # could move this to close if desired

  def do_process(self, data, data_count, process):
    if process['mode'] == 'DLC':
      if process['frame0']:
        process['DLCLive'].init_inference(frame=data)
        process['frame0'] = False
        pose = process['DLCLive'].get_pose(data)
        return pose, process
      else:
        return process['DLCLive'].get_pose(data), None
    elif process['mode'] == 'intrinsic':
      result = self.intrinsic_calibration(data, process)
    elif process['mode'] == 'extrinsic':
      result = self.extrinsic_calibration(data, process)

    return result, None

  def capture(self, data):
    while True:
      # get the image from spinview
      im = self._spincam.GetNextImage(FRAME_TIMEOUT)
      if im.IsIncomplete():
        status = im.GetImageStatus()
        im.Release()
        raise Exception(f"Image incomplete with image status {status} ...")
      # frame = np.reshape(im.GetData(), self.size)
      data = im.GetNDArray()  # TODO: check that this works!!

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

      im.Release()
      yield data

  def open_file(self, filepath):
    return ffmpeg \
        .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{self.width}x{self.height}') \
        .output(filepath, vcodec='libx265') \
        .overwrite_output() \
        .run_async(pipe_stdin=True)

  def close_file(self, fileObj):
    fileObj.stdin.close()
    fileObj.wait()
    del fileObj

  def save(self, data):
    self._file.stdin.write(data.tobytes())

  def get_camera_properties(self):
    nodemap_tldevice = self._spincam.GetTLDeviceNodeMap()
    device_serial_number = PySpin.CStringPtr(
        nodemap_tldevice.GetNode('DeviceSerialNumber')).GetValue()
    nodemap = self._spincam.GetNodeMap()
    height = PySpin.CIntegerPtr(nodemap.GetNode('Height')).GetValue()
    width = PySpin.CIntegerPtr(nodemap.GetNode('Width')).GetValue()
    return device_serial_number, height, width

  # def intrinsic_calibration_switch(self):
  #   # TODO: with self._displaying_lock:
  #   if self._displaying:
  #     if not self._in_calibrating:
  #       print('turning ON intrinsic calibration mode')
  #       self._in_calibrating = True
  #       process['calibrator'].reset()
  #     else:
  #       print('turning OFF intrinsic calibration mode')
  #       self._in_calibrating = False
  #       process['calibrator'].save_config(self.device_serial_number,
  #                                 self.width,
  #                                 self.height)
  #   else:
  #     print("not displaying. Can't perform intrinsic calibration")

  # def extrinsic_calibration_switch(self):
  #   if self._displaying:
  #     if not self._ex_calibrating:
  #       print('turning ON extrinsic calibration mode')
  #       self._ex_calibrating = True
  #       process['calibrator'].reset()
  #       process['calibrator'].load_config(self.device_serial_number)
  #     else:
  #       print('turning OFF extrinsic calibration mode')
  #       self._ex_calibrating = False
  #       process['calibrator'].save_config(self.device_serial_number,
  #                                 self.width,
  #                                 self.height)
  #     # return self.display()

  def intrinsic_calibration(self, frame, process):
    # write something on the frame
    # text = 'Intrinsic calibration mode On'
    # cv2.putText(frame, text, (50, 50),
    #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 125), 2)

    # get corners and refine them in openCV for every 3 frames
    if process['calibrator'].decimator % 3 == 0:  # TODO: move 3 to a constant at top of file
      # TODO: we probably don't need the calib object to store the decimator... just get the frame count from do_process
      corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, process['calibrator'].board.dictionary, parameters=process['calibrator'].params)
      detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
          cv2.aruco.refineDetectedMarkers(frame, process['calibrator'].board, corners, ids,
                                          rejectedImgPoints, parameters=process['calibrator'].params)

      # interpolate corners and draw corners
      if len(detectedCorners) > 0:
        rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
            detectedCorners, detectedIds, frame, process['calibrator'].board)
        if detectedCorners is not None and 2 <= len(
                detectedCorners) <= process['calibrator'].max_size:
          process['calibrator'].allCorners.append(detectedCorners)
          process['calibrator'].allIds.append(detectedIds)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)
    process['calibrator'].decimator += 1

    return {'corners': corners, 'ids': ids}

  def extrinsic_calibration(self, frame):
    # if there isn't configuration on the screen, save corners and ids
    allAligns = False  # TODO fix the logic here
    if process['calibrator'].config is None:
      # text = 'No configuration file found. Performing initial extrinsic calibration... '
      # cv2.putText(frame, text, (50, 50),
      #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      # calibrate every 3 frames
      if process['calibrator'].decimator % 3 == 0:  # TODO: move to constant at top of file
        # get parameters
        params = process['calibrator'].params

        # detect corners
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            frame, process['calibrator'].board.dictionary, parameters=params)
        if ids is not None:
          # draw corners on the screen
          # cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

          if len(ids) >= len(process['calibrator'].allIds):
            process['calibrator'].allCorners = corners
            process['calibrator'].allIds = ids
    else:
      # text = 'Found configuration file for this camera. Calibrating...'
      # cv2.putText(frame, text, (50, 50),
      #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      if True:  # process['calibrator'].decimator % 3 == 0:
        truecorners = process['calibrator'].config['corners']  # float numbers
        trueids = process['calibrator'].config['ids']  # int numbers
        CI = process['calibrator'].config['CI']  # int pixels
        markers = process['calibrator'].config['markers']

        # key step: detect markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            frame, process['calibrator'].board.dictionary, parameters=process['calibrator'].params)

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
          # cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

          allAligns = all(aligns)
          #   text = 'Enough corners aligned! Ready to go'
          #   cv2.putText(frame, text, (500, 1000),
          #               cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
          # else:
          #   text = "Missing ids or corners!"
          #   cv2.putText(frame, text, (500, 1000),
          #               cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
      process['calibrator'].decimator += 1
    return {'corners': corners, 'ids': ids, 'allAligns': allAligns}

  def predisplay(self, data):
    # TODO: make sure text is not overlapping
    process = self.processing

    if process is not None:
      results = self.results
      if results is not None:
        if process['mode'] == 'DLC':
          draw_dots(data, results)
        else:
          cv2.putText(frame, f'Performing {process['mode']} calibration', (50, 50),
                      cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 125), 2)

          cv2.aruco.drawDetectedMarkers(
              frame, results['corners'], results['ids'], borderColor=225)

          if process['mode'] == 'extrinsic':
            if process['calibrator'].config is None:
              text = 'No configuration file found. Performing initial extrinsic calibration... '
            else:
              text = 'Found configuration file for this camera. Calibrating...'
              cv2.putText(frame, text, (50, 60),
                          cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

            if results['allAligns']):
              text='Enough corners aligned! Ready to go'
            else:
              text="Missing ids or corners!"

            cv2.putText(frame, text, (500, 1000),
                        cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

    if self.file is not None:
      text='recording...'
      cv2.putText(data, text, (700, 50),
                  cv2.FONT_HERSHEY_PLAIN, 4.0, 0, 2)

      return data

  def close(self):
    self._spincam.DeInit()
