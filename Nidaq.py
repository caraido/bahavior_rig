import numpy as np
from scipy import signal, interpolate
import matplotlib as mpl
import nidaqmx
from nidaqmx.stream_readers import AnalogSingleChannelReader as AnalogReader
from AcquisitionObject import AcquisitionObject
import os
import time
from io import BytesIO
import ffmpeg
# import RigStatus

AUDIO_INPUT_CHANNEL = 'Dev1/ai1'
AUDIO_INPUT_GAIN = 1e4
PC_BUFFER_TIME_IN_SECONDS = 60  # buffer before python
DUTY_CYCLE = .01  # the fraction of time with the trigger high
TRIGGER_OUTPUT_CHANNEL = 'Dev1/ctr0'


class Nidaq(AcquisitionObject):
  # def __init__(self, frame_rate, audio_settings):
    # Nidaq(status['frame_rate'].current, status['sample frequency'].current,
    #  status['read rate'].current, status['spectrogram'].current)
  def __init__(self, frame_rate, sample_rate, read_rate, spectrogram_settings):
    self.sample_rate = int(sample_rate)
    self.run_rate = read_rate

    self.parse_settings(spectrogram_settings)

    AcquisitionObject.__init__(
        self, self.run_rate, (int(self.sample_rate // self.run_rate), 1))

    # TODO: verify that we are not violating the task state model: https://zone.ni.com/reference/en-XX/help/370466AH-01/mxcncpts/taskstatemodel/
    # specifically, if we change logging mode, do we need to re-commit the task??

    # set up the audio task
    self.audio_task = nidaqmx.Task()
    self.audio_task.ai_channels.add_ai_voltage_chan(AUDIO_INPUT_CHANNEL)
    # self.audio.ai_channels[AUDIO_INPUT_CHANNEL].ai_gain = int(AUDIO_INPUT_GAIN) #TODO: (how) does this work?
    self.audio_task.timing.cfg_samp_clk_timing(
        self.sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
    )
    self.audio_task.in_stream.input_buf_size = self.sample_rate * PC_BUFFER_TIME_IN_SECONDS
    self.audio_task.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
    self._audio_reader = AnalogReader(self.audio_task.in_stream)

    # set up the trigger task
    self.trigger_freq = frame_rate

    self.trigger_task = nidaqmx.Task()
    self.trigger_task.co_channels.add_co_pulse_chan_freq(
        TRIGGER_OUTPUT_CHANNEL, freq=self.trigger_freq, duty_cycle=DUTY_CYCLE
    )
    self.trigger_task.triggers.start_trigger.cfg_dig_edge_start_trig(
        f"/{AUDIO_INPUT_CHANNEL[:-1]}/StartTrigger"
    )  # start the video trigger with the audio channel
    self.trigger_task.timing.cfg_implicit_timing(
        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
    )  # configure the trigger to repeat until the task is stopped
    self.trigger_task.control(nidaqmx.constants.TaskMode.TASK_COMMIT)

    self._log_mode = [False, False]  # [isLogging, isDisplaying]
    self._filepath = ''

  def parse_settings(self, spectrogram_settings):
    self._nfft = int(spectrogram_settings['frequency resolution'].current)
    self._window = int(
        spectrogram_settings['pixel duration'].current * self.sample_rate)
    self._overlap = int(
        spectrogram_settings['pixel fractional overlap'].current * self._window)

    _, _, spectrogram = signal.spectrogram(
        np.zeros((int(self.sample_rate // self.run_rate),)), self.sample_rate, nperseg=self._window, noverlap=self._overlap)
    self._nx = spectrogram.shape[1]
    # self._nx = int(np.round(np.floor(self.sample_rate-self._overlap) /
    #                         (self._window-self._overlap) / self.run_rate))
    self._xq = np.linspace(0, 1, num=self._nx)
    self._yq = np.linspace(0, int(self.sample_rate/2),
                           num=int(self._window/2 + 1))
    if spectrogram_settings['log scaling']:
      self._zq = np.logspace(int(np.log10(spectrogram_settings['minimum frequency'])), int(
          np.log10(spectrogram_settings['maximum frequency'])), num=int(spectrogram_settings['frequency resolution']))
    else:
      self._zq = np.linspace(int(spectrogram_settings['minimum frequency']), int(
          spectrogram_settings['maximum frequency']), num=int(spectrogram_settings['frequency resolution']))

    self._freq_correct = spectrogram_settings['noise correction']

  def start(self, filepath=None, display=False):
    path = os.path.join(self.temp_filepath, 'spectrogram')
    if not os.path.exists(path):
      os.mkdir(path)
    self.temp_file = os.path.join(path, 'stream.m3u8')

    if filepath is None:
      self._has_filepath = False
    else:
      self._has_filepath = True

    self.file = filepath
    self.data = display
    self.running = True

  def open_file(self, fileObj):
    self._log_mode[0] = True
    self._filepath = fileObj
    return fileObj

  def open_temp_file(self, fileObj):

    # filepath should be somethings like 'video{cam_id}_stream/stream.m3u8'
    split_time = 1.0  # in seconds, duration of each file
    # NOTE: tried split_time = 0.25, 0.5. Seems like video gets choppier and latency worsens
    # probably due to needing to fetch more files
    # optimum seems to be near 1
    # stream starts at a ~4sec delay
    # but tends to catch up to a little over 1sec delay

    # TODO: if _nx or _nfft change, we need to close and reopen file!!
    file = (ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{self._nx}x{self._nfft}', framerate=self.run_rate)
            .output(fileObj,
                    format='hls', hls_time=split_time,
                    hls_playlist_type='event', hls_flags='omit_endlist',
                    g=int(self.run_rate * split_time), sc_threshold=0, vcodec='h264',
                    tune='zerolatency', preset='ultrafast')
            .overwrite_output()
            # .run_async(pipe_stdin=True)
            .global_args('-loglevel', 'error')
            # bug~need low logs if quiet
            .run_async(pipe_stdin=True, quiet=True)
            )
    return file

  def close_temp_file(self, fileObj):
    fileObj.stdin.close()
    fileObj.wait()
    del fileObj

  def prepare_display(self):
    self._log_mode[1] = True

  def prepare_run(self):
    if self._log_mode[0]:
      if self._log_mode[1]:
        log_mode = nidaqmx.constants.LoggingMode.LOG_AND_READ
      else:
        log_mode = nidaqmx.constants.LoggingMode.LOG
    else:
      log_mode = nidaqmx.constants.LoggingMode.OFF

    self.audio_task.in_stream.configure_logging(
        self._filepath,
        logging_mode=log_mode,
        operation=nidaqmx.constants.LoggingOperation.CREATE_OR_REPLACE)  # see nptdms

    self.trigger_task.start()
    self.audio_task.start()
    print('trigger on')
    print('audio on')

  def prepare_processing(self, options):
    # in the future if we use deepsqueak for real-time annotation, we would set up for that here
    pass

  def capture(self, data):
    while True:
      self._audio_reader.read_many_sample(
          data[:, 0],
          number_of_samples_per_channel=self.data_size[0]
      )
      yield data

  def predisplay(self, data):
    '''
    Calculate the spectrogram of the data and send to connected browsers.
    There are many ways to approach this, in particular by using wavelets or by using
    overlapping FFTs. For now just trying non-overlapping FFTs ~ the simplest approach.
    '''

    _, _, spectrogram = signal.spectrogram(
        data[:, 0], self.sample_rate, nperseg=self._window, noverlap=self._overlap)

    # print(self._xq.shape, self._yq.shape, spectrogram.shape, self._zq.shape)
    interpSpect = interpolate.RectBivariateSpline(
        self._yq, self._xq, spectrogram)(self._zq, self._xq)  # TODO: try linear instead of spline, univariate instead of bivariate

    if self._freq_correct:
      interpSpect *= self._zq[:, np.newaxis]
      # corrects for 1/f noise by multiplying with f

    thisMin = np.amin(interpSpect, axis=(0, 1))
    interpSpect -= thisMin

    thisMax = np.amax(interpSpect, axis=(0, 1))

    interpSpect /= thisMax  # normalized to [0,1]

    # interpSpect = mpl.cm.viridis(interpSpect) * 255  # colormap
    interpSpect = interpSpect * 255  # TODO: decide how to handle colormapping?
    return interpSpect.astype(np.uint8)

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

      # save the current data
      with self._file_lock:
        if self._file is not None:
          self.save(data)

      # save the spectrogram to temp
      with self._temp_file_lock:
        if self._temp_file is not None:
          spectrogram = self.predisplay(data)
          self._temp_file.stdin.write(spectrogram.tobytes())

      # buffer the current data
      self.data = data

  def end_run(self):
    self.audio_task.stop()
    self.trigger_task.stop()
    os.remove(os.path.split(self.temp_filepath)[0])

  def end_display(self):
    self._log_mode[1] = True

  def close_file(self, fileObj):
    self._log_mode[0] = False
    self._filepath = ''

  def end_processing(self, process):
    # in the future we would teardown deepsqueak here
    pass
