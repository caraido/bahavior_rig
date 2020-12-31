import numpy as np
from scipy import signal, interpolate
import matplotlib as mpl
import nidaqmx
from nidaqmx.stream_readers import AnalogSingleChannelReader as AnalogReader
from AcquisitionObject import AcquisitionObject

AUDIO_INPUT_CHANNEL = 'Dev1/ai1'
AUDIO_INPUT_GAIN = 1e4
PC_BUFFER_TIME_IN_SECONDS = 60  # buffer before python
DUTY_CYCLE = .01  # the fraction of time with the trigger high
TRIGGER_OUTPUT_CHANNEL = 'Dev1/ctr0'


class Nidaq(AcquisitionObject):
  def __init__(self, frame_rate, audio_settings):
    self.parse_settings(audio_settings)

    AcquisitionObject.__init__(
        self, self.run_rate, (self.sample_rate // self.run_rate, 1))

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
        f"{AUDIO_INPUT_CHANNEL[:-1]}/StartTrigger"
    )  # start the video trigger with the audio channel
    self.trigger_task.timing.cfg_implicit_timing(
        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
    )  # configure the trigger to repeat until the task is stopped
    self.trigger_task.control(nidaqmx.constants.TaskMode.TASK_COMMIT)

    self._log_mode = [False, False]  # [isLogging, isDisplaying]
    self._filepath = ''

  def parse_settings(self, audio_settings):
    self.sample_rate = int(audio_settings['fs'])
    self.run_rate = audio_settings['readRate']
    self._nfft = int(audio_settings['nFreq'])
    self._window = int(audio_settings['window'] * self.sample_rate)
    self._overlap = int(audio_settings['overlap'] * self._window)
    self._nx = int(np.floor(self.sample_rate-self._overlap) /
                   (self._window-self._overlap))
    self._xq = np.linspace(0, 1, num=self._nx)
    self._yq = np.linspace(0, int(self.sample_rate/2),
                           num=int(self._window/2 + 1))
    if audio_settings['fScale'] == 'linear':
      self._zq = np.linspace(int(audio_settings['fMin']), int(
          audio_settings['fMax']), num=int(audio_settings['nFreq']))
    else:
      self._zq = np.logspace(int(np.log10(audio_settings['fMin'])), int(
          np.log10(audio_settings['fMax'])), num=int(audio_settings['nFreq']))

    self._freq_correct = audio_settings['correction']

  def open_file(self, fileObj):
    self._log_mode[0] = True
    self._filepath = fileObj
    return fileObj

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

  def prepare_processing(self, options):
    # in the future if we use deepsqueak for real-time annotation, we would set up for that here
    pass

  def capture(self, data):
    while True:
      self._audio_reader.read_many_sample(
          data,
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
        data, self.sample_rate, nperseg=self._window, noverlap=self._overlap)

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

    interpSpect = mpl.cm.viridis(interpSpect) * 255  # colormap
    return interpSpect.astype(np.uint8)

  def end_run(self):
    self.audio_task.stop()
    self.trigger_task.stop()

  def end_display(self):
    self._log_mode[1] = True

  def close_file(self, fileObj):
    self._log_mode[0] = False
    self._filepath = ''

  def end_processing(self, process):
    # in the future we would teardown deepsqueak here
    pass
