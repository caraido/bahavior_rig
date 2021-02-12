import threading
import socket
import time
import numpy as np
import os
from utils.tcp_utils import initTCP, getConnections, sendData, doShutdown

BUFFER_TIME = .005  # time in seconds allowed for overhead


class AcquisitionObject:
  #############
  # LIFECYCLE METHODS TO BE OVERLOADED, LISTED IN ORDER:
  #############

  def open_file(self, fileObj):
    # anything that needs to be done to open a save file for this class
    # file object is just whatever you want to pass to close_file() later on, but cannot be None
    return fileObj

  def prepare_display(self):
    # any necessary setup for displaying
    pass

  def prepare_run(self):
    # any setup that needs to be done before running goes here
    pass

  def prepare_processing(self, options):
    # set up the processing and return the process object
    process = {}
    return process

  def capture(self, data):
    # data starts as self.new_data()
    while True:
      # update data via capture
      yield data

  def save(self, data):
    # anything that needs to be done to save a chunk for this class
    pass

  def do_process(self, data, data_count, process):
    # generate results from the input data
    results = {}
    # if the process has changed, return the updated object
    # else return none

    return results, None
    # return results, process #also valid

  def predisplay(self, data):
    # set up data for displaying, e.g. cv2.puttext or cv2.drawline
    # data is the raw data from capture, so it may need to be reshaped, etc.
    return data

  def end_run(self):
    # any cleanup that needs to be done after running goes here
    pass

  def end_display(self):
    # any clean up after running for the display
    pass

  def close_file(self, fileObj):
    # anything that needs to be done to close a save file for this class
    # fileObj is just whatever gets passed from save_file()
    pass

  def end_processing(self, process):
    # tear down the process
    pass

  def close(self):
    # do anything specific to this class before deleting
    pass

  # NOT OVERLOADED

  def __init__(self, run_rate, data_size, address):
    # children should call the base init method with the run_rate (in Hz) and the data_size (a tuple for numpy to preallocate)
    self.run_rate = run_rate
    self.data_size = data_size

    self._running_lock = threading.Lock()
    self._running = False

    self._file_lock = threading.Lock()
    self._file = None

    self._data_lock = threading.Lock()
    self._data = None

    self._processing_lock = threading.Lock()
    self._processing = False
    self.process_rate = 0

    self._results_lock = threading.Lock()
    self._results = None

    self._has_runner = False
    self._has_processor = False
    self._has_filepath = False
    self._has_displayer = False

    self.address = address

    self.filepath = None

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
      if self._data is None:
        return None
      else:
        return self._data.copy()

  @property
  def data_count(self):
    with self._data_lock:
      return self._data_count

  @property
  def data_and_count(self):
    with self._data_lock:
      if self._data is None:
        return None, 0
      else:
        return self._data.copy(), self._data_count

  @property
  def new_data(self):
    return np.empty(self.data_size)

  @data.setter
  def data(self, data):
    if isinstance(data, bool):
      if data:
        with self._data_lock:
          if self._data is not None:
            self.end_display()
          self._data = self.new_data
          self.prepare_display()
          self._data_count = 0
      else:
        with self._data_lock:
          if self._data is not None:
            self.end_display()
          self._data = None
          self._data_count = 0
    else:
      with self._data_lock:
        self._data = data
        self._data_count += 1

  @property
  def displaying(self):
    with self._data_lock:
      return self._data is not None

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

  @property
  def results(self):
    with self._results_lock:
      return self._results

  @results.setter
  def results(self, results):
    with self._results_lock:
      self._results = results

  @property
  def run_interval(self):
    return self._run_interval

  @run_interval.setter
  def run_interval(self, interval):
    self._run_interval = interval
    # self._run_rate = 1/(interval + BUFFER_TIME)

  @property
  def run_rate(self):
    return 1/(self._run_interval + BUFFER_TIME)

  @run_rate.setter
  def run_rate(self, run_rate):
    self._run_interval = (1/run_rate) - BUFFER_TIME

  @property
  def process_interval(self):
    return self._process_interval

  @property
  def process_rate(self):
    return 1/(self._process_interval + BUFFER_TIME)

  @process_interval.setter
  def process_interval(self, interval):
    self._process_interval = interval

  @process_rate.setter
  def process_rate(self, process_rate):
    # self._process_interval = (1/process_rate) - BUFFER_TIME # division by zero error
    self._process_interval = (1/(process_rate+0.000001)-BUFFER_TIME)

  @property
  def data_size(self):
    return self._data_size

  @data_size.setter
  def data_size(self, data_size):
    self._data_size = data_size

  def start(self, filepath=None, display=False):
    self.filepath = filepath
    self.file = filepath
    self.data = display
    self.running = True

  def stop(self):
    self.running = False
    self.file = None
    self.data = False
    self.processing = None

  def wait_for(self):
    while self._has_runner or self._has_processor or self._has_displayer:
      check_time = time.time()
      self.sleep(check_time)

  def sleep(self, last):
    pause_time = last + self.run_interval - time.time()
    if pause_time > 0:
      time.sleep(pause_time)

  def display(self):
    if self._has_displayer:
      print('already had displayer. exiting.')
      return  # only 1 runner at a time

    self._has_displayer = True
    print('created displayer')

    last_count = 0
    try:
      data, data_count = self.data_and_count
    except:
      self._has_displayer = False
      print('acquisition object not in display mode. exiting.')
      return

    last_data_time = time.time()

    recipients = []
    sock = initTCP(self.address)

    # getConnections(sock, recipients, block=True)

    while self._data is not None:
      # NOTE: I don't love how when we have no recipients we keep requesting the data anyways
      # that's why I elected to check if self._data is none instead
      # we don't have the thread lock but should be okay for just reading None status?
      print('data was not none')

      if len(recipients) == 0:
        self.sleep(time.time())
        getConnections(sock, recipients, block=False)

      elif data_count > last_count:
        last_data_time = time.time()
        last_count = data_count

        data = self.predisplay(data)  # do any additional frame workup

        getConnections(sock, recipients, block=False)  # check for new clients
        sendData(data.astype(np.uint8).tobytes(), recipients)

        self.sleep(last_data_time)
        data, data_count = self.data_and_count

    print('Data was none. Exiting.')

    doShutdown(sock, recipients)
    self._has_displayer = False

  def run(self):
    print('started child run')
    if self._has_runner:
      return  # only 1 runner at a time

    self._has_runner = True
    data = self.new_data
    capture = self.capture(data)
    data_time = time.time() - self.run_interval

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

      # save the current data to temp
      with self._file_lock:
        if self._file is not None:
          self.save(data)

      # buffer the current data
      self.data = data

  def run_processing(self):
    if self._has_processor:
      return  # only 1 runner at a time

    self._has_processor = True

    results_time = time.time() - self.process_interval
    last_data_count = 0

    while True:
      self.sleep(results_time)
      try:
        data, data_count = self.data_and_count
      except:
        continue

      if data_count == last_data_count:
        results_time = time.time()
        continue

      with self._processing_lock:
        if self._processing:
          results_time = time.time()
          results, process = self.do_process(
              data, data_count, self._processing)
          if process is not None:
            self._processing = process
        else:
          self._has_processor = False
          # TODO: return what?

          return

      # buffer the current data
      self.results = results

  def rmdir(self, path):
    # print('rming from acq obj')
    for root, dirs, files in os.walk(path, topdown=False):
      for name in files:
        os.remove(os.path.join(root, name))
      for name in dirs:
        os.rmdir(os.path.join(root, name))
    os.rmdir(path)

  def __del__(self):
    self.stop()
    self.wait_for()
    self.close()
