# idea: 
# should not run at same time as ag


# files/directories management
# .tdms to .mat (squeaks?)
# log of errors?
# run offline dlc model on videos 
# copy to server and hard drive
# generate plots (histogram of view angle, mouse location, maybe squeak analysis down the line)

# raw and processed subfolders

# separated into threads? each runs through a sequence of tasks
# one for video
# run dlc models on 1 video at a time?
# generate plots
# sends to server, HDD
# one for audio
# convert to .mat / .wav etc.
# down the line: run deepsqueak
# down the line: analyze squeaks and plot?
# sends to server, HDD
# one for other stuff
# compile errors/config info into a file in the directory
# calibration
# sends to server, HDD

# when all the threads are done
# delete SSD folder

from utils.path_operation_utils import copy_config, load_config
from utils.calibration_utils import undistort_videos,undistort_markers
from utils.geometry_utils import find_window_center
from utils.dlc_utils import dlc_analysis
import os
from nptdms import TdmsFile
from main import audio_settings
import scipy.io as sio


class ProcessingGroup:

	def __init__(self):
		self.rootpath = None
		self.dlcpath = None


	def __call__(self, rootpath,dlcpath):
		self.dlcpath =dlcpath
		self.rootpath=rootpath
		self.processpath = os.path.join(self.rootpath, 'processed') # make it a property
		self.config_path = os.path.join(self.rootpath,'config')

	def copy_configs(self):
		if self.rootpath:
			copy_config(self.rootpath)

	def post_process(self,calib=True,
					 mat=True,
					 dlc=True,
					 dsqk=True,
					 server=True,
					 HDD=True):
		if calib:
			self.post_calibration()
		if mat:
			self.tdms2mat()
		if dlc:
			self.dlc_analysis()
		if dsqk:
			self.dsqk_analysis()
		if server:
			self.SSD2server()
		if HDD:
			self.SSD2HDD()

	def post_calibration(self):
		# for each camera, there is one "extrinsic" and "intrinsic"

		# undistort the videos
		# time consumimg
		result = undistort_videos(self.rootpath)
		if result is not None:
			print("saved undistorted videos!")

		# undistort the markers
		undistort_markers(self.rootpath)

		# find window and arena center
		find_window_center(self.rootpath)

	def tdms2mat(self):
		dir_list = os.listdir(self.rootpath)
		item_list = [self.rootpath + item for item in dir_list if '.tdms' in item and '.tdms_index' not in item]
		f = item_list[0]

		with TdmsFile.open(f) as file:
			group = file.groups()[0]
		channel = group.channels()[0]
		data = channel[:]
		audio_name = os.path.split(self.rootpath)[1]
		audio = {audio_name:data,'fs':audio_settings['fs']}
		sio.savemat(os.path.join(self.rootpath,'audio.mat'),audio)

	def dlc_analysis(self):
		# dlc anlysis on TOP CAMERA only
		dlc_analysis(self.rootpath, self.dlcpath)

	def dsqk_analysis(self):
		pass

	def SSD2server(self):
		# copy and paste
		pass

	def SSD2HDD(self):
		# copy and paste
		pass
