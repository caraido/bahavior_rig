import threading
from io import BytesIO
from PIL import Image
import time
import numpy as np

BUFFER_TIME = .005  # time in seconds allowed for overhead


class AcquisitionObject:
  #############
  # LIFECYCLE METHODS TO BE OVERLOADED, LISTED IN ORDER:
  #############
  def prepare_run(self):
    # any setup that needs to be done before running goes here
    pass

  def open_file(self, fileObj):
    # anything that needs to be done to open a save file for this class
    return fileObj

  def prepare_processing(self, opts):
    # set up the processing and return the process object
    process = {}
    return process

  def capture(self):
    data = self.new_data  # preallocate for speed
    while True:
      # update data via capture
      yield data

  def save(self, data):
    # anything that needs to be done to save a chunk for this class
    pass

  def predisplay(self, data):
    # set up data for displaying, e.g. cv2.puttext or cv2.drawline
    return data

  def end_run(self):
    # any cleanup that needs to be done after running goes here
    pass

  def close_file(self, fileObj):
    # anything that needs to be done to close a save file for this class
    pass

  def end_processing(self, process):
    # tear down the process
    pass

  def close(self):
    # do anything specific to this class before deleting
    pass

  # NOT OVERLOADED

  def __init__(self, rate, data_size):
    self._interval = (1 / rate) - BUFFER_TIME
    self.data_size = data_size

    self._running_lock = threading.Lock()
    self._running = False

    self._file_lock = threading.Lock()
    self._file = None

    self._data_lock = threading.Lock()
    self._data = None

    self._processing_lock = threading.Lock()
    self._processing = False

    self._has_runner = False

  @property
  def running(self):
    with self._running_lock:
      return self._running

  @running.setter
  def running(self, running):
    if running:
      with self._running_lock:
        if not self._running:
          self.prepare_run()
          self._running = True
    else:
      with self._running_lock:
        if self._running:
          self.end_run()
          self._running = False

  @property
  def file(self):
    with self._file_lock:
      return self._file

  @file.setter
  def file(self, file):
    if file is not None:
      with self._file_lock:
        if self._file is not None:
          self.close_file(self._file)

        self._file = self.open_file(file)
    else:
      with self._file_lock:
        if self._file is not None:
          self.close_file(self._file)
          del self._file
          self._file = None

  @property
  def data(self):
    with self._data_lock:
      return self._data

  @property
  def data_count(self):
    with self._data_lock:
      return self._data_count

  @property
  def data_and_count(self):
    with self._data_lock:
      return self._data, self._data_count

  @property
  def new_data(self):
    return np.empty(self.data_size)

  @data.setter
  def data(self, data):
    if isinstance(data, bool):
      if data:
        with self._data_lock:
          self._data = self.new_data
          self._data_count = 0
      else:
        with self._data_lock:
          self._data = None
          self._data_count = 0
    else:
      with self._data_lock:
        self._data = data
        self._data_count += 1

  @property
  def processing(self):
    with self._processing_lock:
      return self._processing

  @processing.setter
  def processing(self, processing):
    if processing is not None:
      with self._processing_lock:
        if self._processing:
          self.end_processing(self._processing)

        self._processing = self.prepare_processing(processing)

    else:
      with self._processing_lock:
        if self._processing:
          self.end_processing(self._processing)
          self._processing = None

  def start(self, filepath=None, display=False):
    self.file = filepath
    self.data = display
    self.running = True

  def stop(self):
    self.file = None
    self.data = False
    self.running = False
    self.processing = None

  def sleep(self, last):
    pause_time = last + self._interval - time.time()
    if pause_time > 0:
      time.sleep(pause_time)

  def display(self):
    frame_bytes = BytesIO()
    last_count = 0

    data, data_count = self.data_and_count
    last_data_time = time.time()

    while data is not None:
      if data_count > last_count:
        last_data_time = time.time()
        data = self.predisplay(data)  # do any additional frame workup

        frame_bytes.seek(0)
        Image.fromarray(data).save(frame_bytes, 'bmp')
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes.getvalue() + b'\r\n')
      else:
        self.sleep(last_data_time)
      data, data_count = self.data_and_count

  def run(self):
    if self._has_runner:
      return  # only 1 runner at a time

    self._has_runner = True
    data = self.new_data
    capture = self.capture()
    data_time = time.time() - self._interval

    while True:
      self.sleep(data_time)

      with self._running_lock:
        # try to capture the next data segment
        if self._running:
          data_time = time.time()
          data = next(capture)
        else:
          self._has_runner = False
          return

      # save the current data
      with self._file_lock:
        if self._file is not None:
          self.save(data)

      # buffer the current data
      self.data = data

  def __del__(self):
    self.stop()
    while self._has_runner:
      check_time = time.time()
      self.sleep(check_time)
    self.close()
