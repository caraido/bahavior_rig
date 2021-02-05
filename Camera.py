import cv2
import PySpin
import numpy as np
import ffmpeg
from utils.calibration_utils import Calib
import pandas as pd
from dlclive import DLCLive, Processor
from AcquisitionObject import AcquisitionObject
from utils.image_draw_utils import draw_dots
import os



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

  def start(self, filepath=None, display=False):
    if filepath is None:
      path = os.path.join(self.temp_file,'camera_'+self.device_serial_number)
      if not os.path.exists(path):
        os.mkdir(path)
      self.file=os.path.join(path,'stream.m3u8')
      self._has_filepath=False
    else:
      path = os.path.join(filepath,'camera_'+self.device_serial_number)
      if not os.path.exists(path):
        os.mkdir(path)
      self.file=os.path.join(path,'stream.m3u8')
      self._has_filepath=True

    self.filepath=filepath
    self.data = display
    self.running = True

  # TODO: make sure this step should be in prepare_display or prepare_run
  def prepare_display(self): #TODO: prepare_run?
    self._spincam.BeginAcquisition()

  def end_display(self):
    self._spincam.EndAcquisition()

  def prepare_processing(self, options):
    process = {}

    if options['mode'] == 'DLC':
      # process['modelpath'] = options
      process['mode'] = 'DLC'
      process['processor'] = Processor()
      process['DLCLive'] = DLCLive(
                                    model_path=options['modelpath'],
                                    processor=process['processor'],
                                    display=False,
                                    resize=DLC_RESIZE)
      process['frame0'] = True
      return process
    else:  # mode should be 'intrinsic' or 'extrinsic'
      process['mode'] = options['mode']

      # could move this to init if desired
      process['calibrator'] = Calib(options['mode'])
      return process
      #process['calibrator'].root_config_path= self.file # does this return the file path?

      #process['calibrator'].reset()
      #if options['mode'] == 'extrinsic':
        #process['calibrator'].load_ex_config(self.device_serial_number)

  def end_processing(self, process):
    if process['mode'] == 'DLC':
      process['DLCLive'].close()
      process['frame0'] = False
      status = 'DLC Live turned off'
    else:
      status = process['calibrator'].save_config(
          self.device_serial_number, self.width, self.height)

      del process['calibrator']  # could move this to close if desired
    # TODO:status should be put on the screen!
    return status

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
      result = process['calibrator'].in_calibrate(data,data_count)
      #result = self.intrinsic_calibration(data, process)
      return result, None
    elif process['mode'] == 'extrinsic':
      result = process['calibrator'].ex_calibrate2(data, data_count)
    #  result = self.extrinsic_calibration(data, process)
      return result, None

  def capture(self, data):
    while True:
      # get the image from spinview
      im = self._spincam.GetNextImage(FRAME_TIMEOUT)
      if im.IsIncomplete():
        status = im.GetImageStatus()
        im.Release()
        raise Exception(f"Image incomplete with image status {status} ...")
      data = im.GetNDArray()
      im.Release()
      yield data

  '''
  def open_file(self, filepath):
    return ffmpeg \
        .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{self.width}x{self.height}',framerate=self.run_rate) \
        .output(filepath, vcodec='libx265') \
        .overwrite_output() \
        .run_async(pipe_stdin=True)
  '''

  def open_file(self, filepath):
    # filepath should be somethings like 'video{cam_id}_stream/stream.m3u8'
    split_time = 1.0  # in seconds, duration of each file
    # NOTE: tried split_time = 0.25, 0.5. Seems like video gets choppier and latency worsens
    # probably due to needing to fetch more files
    # optimum seems to be near 1
    # stream starts at a ~4sec delay
    # but tends to catch up to a little over 1sec delay

    file = (ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray', s='1280x1024', framerate=self.run_rate)
            .output(filepath,
                    format='hls', hls_time=split_time,
                    hls_playlist_type='event', hls_flags='omit_endlist',
                    g=int(self.run_rate * split_time), sc_threshold=0, vcodec='h264',
                    tune='zerolatency', preset='ultrafast')
            .overwrite_output()
            # .run_async(pipe_stdin=True)
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdin=True, quiet=True)  # bug~need low logs if quiet
            )

    return file

  def close_file(self, fileObj):
    fileObj.stdin.close()
    fileObj.wait()
    del fileObj

  def end_run(self):
    with open(self.filepath, 'a') as fobj:
      fobj.write('#EXT-X-ENDLIST')

    if self._has_filepath:
      head = os.path.split(os.path.split(self.filepath)[0])[0]

      ffmpeg \
        .input(self.filepath) \
        .output(head + '.mp4', vcodec='copy') \
        .run()
    os.remove(os.path.split(self.filepath)[0])

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


  def predisplay(self, frame):
    #TODO: still get frame as input? but should return some kind of dictionary? or array?
    #TODO: where does this get called from?

    # TODO: make sure text is not overlapping
    process = self.processing

    if process is not None:
      results = self.results
      if results is not None:
        if process['mode'] == 'DLC':
          draw_dots(frame, results)
        else:
          cv2.putText(frame, f"Performing {process['mode']} calibration", (50, 50),
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

            '''
            if results['allAligns']:
              text='Enough corners aligned! Ready to go'
            else:
              text="Missing ids or corners!"
            '''

            if results['allDectected']:
              text = 'Enough corners detected! Ready to go'
            else:
              text = "Not enough corners! Please adjust the camera"

            cv2.putText(frame, text, (500, 1000),
                        cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

    if self.file is not None:
      text='recording...'
      cv2.putText(frame, text, (700, 50),
                  cv2.FONT_HERSHEY_PLAIN, 4.0, 0, 2)

    return frame

#  def end_run(self):
#    if self.file:
#      copy_config(self.file)

  def close(self):
    self._spincam.DeInit()

  def __del__(self):
    self._spincam.DeInit()
