import numpy as np
from AcquisitionObject import AcquisitionObject
import pyaudio
import nptdms
from scipy import signal, interpolate
import scipy.io.wavfile as wavfile
from utils.audio_processing import read_audio
import time

BUFFER_TIME = .005  # time in seconds allowed for overhead
AUDIO_INPUT_CHANNEL = 'Dev1/ai2'
CHANNEL_NAME='channel_0'
N_CHANNELS=1
DUTY_CYCLE = .01  # the fraction of time with the trigger high

class Mic(AcquisitionObject):
	def __init__(self, sample_rate, spectogram_settings,address):
		self.sample_rate=int(sample_rate)
		self.parse_settings(spectogram_settings)
		self.group_name = AUDIO_INPUT_CHANNEL
		self.channel_name = CHANNEL_NAME
		self.channels = N_CHANNELS
		self.format = pyaudio.paFloat32
		self.index = None
		self.stream = None
		self.filepath = None

		AcquisitionObject.__init__(
			self, self.run_rate, int(self.sample_rate // self.run_rate),address)

		# create an instance of PyAudio and find the mic
		self.audio = pyaudio.PyAudio()
		for devices in range(self.audio.get_device_count()):
			info = self.audio.get_device_info_by_index(devices)
			if 'UltraMic' in info['name']:
				self.index = devices

		self.duty_cycle = DUTY_CYCLE


	def parse_settings(self, audio_settings):
		self._nfft = int(audio_settings['frequency resolution'].current)
		self._window = int(audio_settings['pixel duration'].current * self.sample_rate)
		self._overlap = int(audio_settings['pixel fractional overlap'].current * self._window)
		self.run_rate = audio_settings['read rate'].current

		_, _, spectrogram = signal.spectrogram(
			np.zeros((int(self.sample_rate // self.run_rate),)), self.sample_rate, nperseg=self._window,
			noverlap=self._overlap)

		self._nx = spectrogram.shape[1]

		self._xq = np.linspace(0, 1, num=self._nx)
		self._yq = np.linspace(0, int(self.sample_rate / 2), num=int(self._window / 2 + 1))

		if audio_settings['log scaling'].current:
			self._zq = np.logspace(int(np.log10(audio_settings['minimum frequency'].current)), int(
          np.log10(audio_settings['maximum frequency'].current)), num=int(audio_settings['frequency resolution'].current))
		else:
			self._zq = np.linspace(int(np.log10(audio_settings['minimum frequency'].current)), int(
				audio_settings['maximum frequency'].current),
								   num=int(audio_settings['frequency resolution'].current))

		self._freq_correct = audio_settings['noise correction'].current
		print(f'_nx is {self._nx} and _nfft is {self._nfft}')

	def start(self,filepath=None,display=False):
		if self.index is not None:
			self.stream = self.audio.open(format=self.format,
			                              channels=self.channels,
			                              input_device_index=self.index,
			                              frames_per_buffer=int(self.sample_rate // self.run_rate),
			                              rate=self.sample_rate,
			                              stream_callback=self.capture_chunk,
			                              input=True
			                              )
		self.filepath=filepath
		self.file=filepath
		self.data = display
		self.running = True

		if filepath is None:
			self._has_filepath = False
		else:
			self._has_filepath = True


	def run(self):
		if self._has_runner:
			return
		self._has_runner=True
		self.stream.start_stream()
		data_time = time.time() - self.run_interval
		while True:
			self.sleep(data_time)
			with self._running_lock:
				# try to capture the next data segment
				if not self._running:
					self._has_runner = False
					return

	def capture_chunk(self,in_data, frame_count, time_info, status):
		self.data = self.new_data
		if self._has_runner:
			data_chunk = nptdms.ChannelObject(self.group_name,
			                                  self.channel_name,
			                                  self.data,
			                                  properties={})
			if self.filepath is not None:
				with nptdms.TdmsWriter(self.filepath,'a') as writer:
					writer.write_segment([data_chunk])

		return (self.data, pyaudio.paContinue)


	def predisplay(self, data):
		'''
		Calculate the spectrogram of the data and send to connected browsers.
		There are many ways to approach this, in particular by using wavelets or by using
		overlapping FFTs. For now just trying non-overlapping FFTs ~ the simplest approach.
		'''

		_, _, spectrogram = signal.spectrogram(
			data, self.sample_rate, nperseg=self._window, noverlap=self._overlap)

		print(self._xq.shape, self._yq.shape, spectrogram.shape, self._zq.shape)
		interpSpect = interpolate.RectBivariateSpline(
			self._yq, self._xq, spectrogram)(self._zq, self._xq)  # TODO: try linear instead of spline, univariate instead of bivariate

		if self._freq_correct:
			interpSpect *= self._zq[:, np.newaxis]
			# corrects for 1/f noise by multiplying with f

		thisMin = np.amin(interpSpect, axis=(0, 1))
		interpSpect -= thisMin

		thisMax = np.amax(interpSpect, axis=(0, 1))

		if thisMax > 0:
		  interpSpect /= thisMax  # normalized to [0,1]

		# interpSpect = mpl.cm.viridis(interpSpect) * 255  # colormap
		interpSpect = interpSpect * 255  # TODO: decide how to handle colormapping?
		return interpSpect.astype(np.uint8)

	def end_run(self):
		self.stream.stop_stream()
		self.stream.close()
		#self.audio.terminate()
		self.stream=None
		if self.filepath:
			audio,_=read_audio(self.filepath)
			wavfile.write(self.filepath[:-4]+'wav',self.sample_rate,audio)
			print('save USB mic')

	def close(self):
		self.audio.terminate()





