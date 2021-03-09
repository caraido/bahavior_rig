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

from utils.path_operation_utils import copy_config, global_config_path
from utils.calibration_utils import undistort_videos, undistort_markers, Calib, TOP_CAM
from utils.dlc_utils import dlc_analysis
from utils.geometry_utils import find_board_center_and_windows
import os
import numpy as np

import shutil
import toml
import time
from utils.calibration_3d_utils import get_extrinsics

model_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\Alec_second_try-Devon-2020-12-07\config.yaml'
HDD_path = r'E:\behavior_data_archive'


class ProcessingGroup:

	def __init__(self):
		self.rootpath = None
		self.dlcpath = None
		self.global_config_path = global_config_path
		self.in_calib = Calib('intrinsic')
		self.al_calib = Calib('alignment')
		self.ex_calib = Calib('extrinsic')

	def __call__(self, rootpath, dlcpath=model_path):
		self.dlcpath = dlcpath
		self.rootpath = rootpath
		self.processpath = os.path.join(self.rootpath, 'processed')  # make it a property
		self.config_path = os.path.join(self.rootpath, 'config')  # make it a property
		if not os.path.exists(self.processpath):
			os.mkdir(self.processpath)
		if not os.path.exists(self.config_path):
			os.mkdir(self.config_path)

	def copy_configs(self):
		if self.rootpath:
			copy_config(self.rootpath)

	def check_process(self):
		# check under rootpath first
		# check calib:
		## is_done_intrinsic = self.check_intrinsic()
		## is done_alignment = self.check_alignment()
		## is done_extrinsic =self.check_extrinsic()
		## is_done_undistort =self.check_undistort()
		# check copied
		# check dlc
		# check dsqk
		# check uploaded to server
		# don't need to check if saved in HDD since the temp folder will be deleted
		# return {'intrinsic':is_done_intrinsic,
		# 		'alignment': is_done_alignment,
		# 		'extrinsic': is_done_extrinsic,
		# 		'undistorted: 'is_done_undistort,
		#		'copied': is copied,
		# 		'dlc': is_done_dlc,
		# 		'dsqk': is_done_dsqk,
		# 		'2server': is_done_2server	}
		pass

	def post_process(self,
					 intrinsic=True,
					 alignment=True,
					 extrinsic=True,
					 undistort=True,
					 copy=True,
					 dlc=True,
					 dsqk=True,
					 server=True,
					 HDD=True):

		# self.copy_configs()

		if intrinsic:
			# do the following under behavior_rig/config/
			self.get_intrinsic_config()

		if alignment:
			self.get_alignment_config()
			self.find_windows()

		if extrinsic:
			self.get_extrinsic_config()

		if copy:
			self.copy_configs()

		if undistort:
			undistort_videos(self.rootpath)

		if dsqk:
			self.dsqk_analysis()
		if dlc:
			self.dlc_analysis()
		if server:
			self.SSD2server()
		if HDD:
			self.SSD2HDD()

	def get_intrinsic_config(self):
		# get item list
		items = os.listdir(self.global_config_path)
		for item in items:
			if 'intrinsic' in item and 'temp' in item:
				temp_config_path = os.path.join(self.global_config_path, item)
				self.in_calib.save_processed_config(temp_config_path)
				os.remove(temp_config_path)

	def get_alignment_config(self):
		alignment = 'config_alignment_%s_temp.toml' % TOP_CAM
		path = os.path.join(self.global_config_path, alignment)
		self.al_calib.save_processed_config(path)
		try:
			os.remove(path)
		except:
			pass

	def get_extrinsic_config(self):
		# get item list
		items = os.listdir(self.global_config_path)
		intrinsic_list = []
		video_list = []
		board = self.ex_calib.board

		for item in items:
			if 'intrinsic' in item and 'temp' not in item:
				intrinsic_path = os.path.join(self.global_config_path, item)
				intrinsic_list.append(intrinsic_path)
			if 'MOV' in item:
				video_path = os.path.join(self.global_config_path, item)
				video_list.append(video_path)

		if len(video_list)==4:
			intrinsic_list.sort()
			video_list.sort()
			loaded = [toml.load(path) for path in intrinsic_list]
			has_top = [True if TOP_CAM in item else False for item in video_list]
			cam_align = has_top.index(True)
			vid_indices = list(range(len(has_top)))

			# calibration
			extrinsic, error = get_extrinsics(vid_indices,
											  video_list,
											  loaded,
											  cam_align,
											  board)
			time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
			extrinsic_new = dict(zip(map(str, extrinsic.keys()), extrinsic.values()))
			results = {'extrinsic': extrinsic_new,
					   'error': error,
					   'date': time_stamp}
			extrinsic_path = 'config_extrinsic.toml'
			saving_path = os.path.join(self.global_config_path,extrinsic_path)
			with open(saving_path, 'w') as f:
				toml.dump(results, f)

			# remove everything after calibration
			for item in items:
				if 'extrinsic' in item and 'temp' in item:
					os.remove(os.path.join(self.global_config_path,item))

	def find_windows(self):
		alignment = 'config_alignment_%s.toml'%TOP_CAM
		rig = 'config_behavior_rig.toml'

		with open(os.path.join(self.global_config_path, alignment), 'r') as f:
			config = toml.load(f)
		if 'recorded_center' not in config.keys():
			try:
				new_corners = np.array(config['undistorted_corners'])
			except:
				new_corners = np.array(config['corners'])
			ids = config['ids']
			ids = [int(i[0][0]) for i in ids]
			new_corners = np.array([corner for _, corner in sorted(zip(ids, new_corners))])

			if new_corners.shape != (6, 1, 4, 2):
				raise Exception("can't proceed! missing corners")
			with open(os.path.join(self.global_config_path, rig), 'r') as f:
				rig = toml.load(f)
			results = find_board_center_and_windows(new_corners, rig)
			with open(os.path.join(self.global_config_path, alignment), 'a') as f:
				toml.dump(results, f, encoder=toml.TomlNumpyEncoder())

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
		name = os.path.split(self.rootpath)[-1]
		backup_path = os.path.join(HDD_path, name)
		try:
			shutil.copytree(self.rootpath, backup_path)
		except shutil.Error as e:
			for src, dst, msg in e.args[0]:
				# src is source name
				# dst is destination name
				# msg is error message from exception
				print(dst, src, msg)
		pass


if __name__ == '__main__':
	import tqdm
	GLOBAL_CONFIG_PATH = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig'
	working_dir = r'D:\Desktop'
	items =os.listdir(working_dir)
	pg = ProcessingGroup()

	for item in tqdm.tqdm(items):
		path = os.path.join(working_dir,item)
		pg(path)
		pg.post_process(intrinsic=False,
						alignment=False,
						extrinsic=False,
						undistort=True,
						copy=True,
						dlc=False,
						dsqk=False,
						server=False,
						HDD=False)
