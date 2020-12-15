import cv2
import numpy as np
from scipy import signal, interpolate
from scipy import io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import ffmpeg
import utils.calibration_utils as cau
import utils.image_draw_utils as idu
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

BUFFER_TIME = .005  # time in seconds allowed for overhead

class Nidaq:
  def __init__(self, frame_rate, audio_settings):
    self.audio = None
    self.trigger = None
    self._audio_reader = None
    self.data = None
    self._nBuffers = 10 # TODO: move to top of file

    self.sample_rate = int(audio_settings['fs'])
    self.trigger_freq = frame_rate
    self.duty_cycle = .01 #TODO: move to top of file
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

    # for display
    self._nfft = int(audio_settings['nFreq'])
    self._window = int(audio_settings['window'] * self.sample_rate)
    self._overlap = int(audio_settings['overlap'] * self._window)
    self._nx = int(np.floor(self.sample_rate-self._overlap) /
                   (self._window-self._overlap))

    # number of calculated timepoints
    self._xq = np.linspace(0, 1, num=self._nx)

    # number of frequency points
    self._yq = np.linspace(0, int(self.sample_rate/2),
                           num=int(self._window/2 + 1))

    # we will use scipy.interpolate to convert yq to zq
    if audio_settings['fScale'] == 'linear':
      self._zq = np.linspace(int(audio_settings['fMin']), int(
          audio_settings['fMax']), num=int(audio_settings['nFreq']))
    else:
      self._zq = np.logspace(int(np.log10(audio_settings['fMin'])), int(
          np.log10(audio_settings['fMax'])), num=int(audio_settings['nFreq']))

    self._freq_correct = audio_settings['correction']

    self._frame_bytes = BytesIO()

  def start(self, filepath=None, display=True):
    with self._running_lock:
      if not self._running:
        # audio task
        self.audio = nidaqmx.Task()
        self.audio.ai_channels.add_ai_voltage_chan(
            "Dev1/ai1" #TODO: channel name at top of file
        )  # this channel measures the audio signal

        # self.audio.ai_channels.ai_microphone_sensitivity=100 # doesn't know if it works
        self.audio.ai_channels['Dev1/ai1'].ai_gain = 10000 #TODO: does this work? put at top of file
        self.audio.timing.cfg_samp_clk_timing(
            self.sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
        )
        self.audio.in_stream.input_buf_size = self.sample_rate * \
            60  # buffer on PC in seconds TODO: put at top of file

        self.audio.control(
            nidaqmx.constants.TaskMode.TASK_COMMIT
        )  # transition the task to the committed state so it's ready to start

        self.read_count = 0

        if display:
          self._displaying = True
          self._audio_reader = AnalogReader(
              self.audio.in_stream)
          self._read_size = self.sample_rate // self.read_rate

          self.data = [np.ndarray(shape=(self._read_size)) #TODO: should we use np.ndarray or np.empty? make the same as other calls to np.empty
                       for i in range(self._nBuffers)]

          self.log_mode = nidaqmx.constants.LoggingMode.LOG_AND_READ #if filepath is none then this should be .OFF
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
        self._saving = False
        self.filepath = filepath

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
      self._saving = True
      if self.filepath:
        self.audio.in_stream.start_new_file(self.filepath)
      else:
        # TODO: will set a default path to save
        raise FileNotFoundError("file path is not found!")
    # else:
    #  self._saving=False
    #  self.audio.in_stream.logging_mode = nidaqmx.constants.LoggingMode.OFF

  def display_switch_on(self):
    if not self._displaying:
      self._displaying = True
      self._audio_reader = AnalogReader(
          self.audio.in_stream)
      self._read_size = self.sample_rate // self.read_rate

      self.data = [np.ndarray(shape=(self._read_size))
                   for i in range(self._nBuffers)]

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
        read_count = max(self.read_count - 1, 0) #TODO: what the hell is this

      while self._displaying:

        last = 0
        # we will wait a bit less than the interval between frames
        interval = (1 / self.read_rate) - BUFFER_TIME

        pause_time = last + interval - time.time() #TODO: last needs to be reset
        if pause_time > 0:
          time.sleep(pause_time)

        with self._data_lock:
          if self.read_count > read_count:
            # note that we're not guaranteed to be gathering sequential reads...

            read_count = self.read_count
            #increment last to time.time()
            flag = True

        if flag:
          # we ought to extend the sampled data range so that the tails of the spectrogram are accurate with the desired overlap
          # but as the number of windows increases, this probably becomes minor

          _, _, spectrogram = signal.spectrogram(self.data[(
              self.read_count-1) % self._nBuffers], self.sample_rate, nperseg=self._window, noverlap=self._overlap)

          # print(self._xq.shape, self._yq.shape, spectrogram.shape, self._zq.shape)
          respect = interpolate.RectBivariateSpline(
              self._yq, self._xq, spectrogram)(self._zq, self._xq) #TODO: try linear instead of spline, univariate instead of bivariate

          if self._freq_correct == True:
            respect *= self._zq[:, np.newaxis]
            # corrects for 1/f noise by multiplying with f

          thisMin = np.amin(respect, axis=(0, 1))
          respect -= thisMin

          thisMax = np.amax(respect, axis=(0, 1))

          respect /= thisMax  # normalized to [0,1]

          respect = mpl.cm.viridis(respect) * 255  # colormap

          self._frame_bytes.seek(0)  # go to the beginning of the buffer
          Image.fromarray(respect.astype(np.uint8)).save(
              self._frame_bytes, 'bmp')
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
        self.audio.close() #TODO: maybe we want to use nidaqmx.task.stop() ??
        self.trigger.close()
        self._running = False
        self._displaying = False
        self._saving = False
        print('stopped nidaq')

  def __del__(self):
    self.stop()

