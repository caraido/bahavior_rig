import os
import time

saving_path_prefix = 'C:/Users/SchwartzLab'
default_saving_path= 'Desktop'
default_folder_name = 'Testing'

@property
def get_saving_path_prefix():
	return saving_path_prefix

@property
def get_default_path():
	return default_saving_path

@property
def get_default_name():
	return default_folder_name

def change_default_path(input_path):
	global default_saving_path
	if os.path.exists(saving_path_prefix + '/' + default_saving_path):
		default_saving_path = input_path
		print('changed default saving path into: '+ input_path)
	else:
		raise NotADirectoryError("The specified path doesn't exist!")

def change_default_name(input_name):
	global default_folder_name
	default_folder_name=input_name
	print('changed default saving folder name into: ' + input_name)

def change_path_prefix(input_prefix):
	global saving_path_prefix
	if os.path.exists(os.path.normpath(saving_path_prefix)):
		saving_path_prefix = input_prefix
	else:
		raise NotADirectoryError("The specified path doesn't exist!")

def reformat_filepath(path,name,camera:list):
	date= time.strftime("%Y-%m-%d_",time.localtime())
	if path == '':
		real_path = saving_path_prefix+'/'+default_saving_path
		print("No file path specified. Will use default path")
	else:
		real_path = saving_path_prefix+'/'+path

	if not os.path.exists(real_path):
		os.makedirs(real_path)
		print("file path %s doesn't exist, creating one..." % real_path)

	if name =='':
		full_path = real_path + '/' + date + default_folder_name
		print("No folder name specified. Will use default folder name")
	else:
		full_path = real_path + '/' + date + name

	if not os.path.exists(full_path):
		os.mkdir(full_path)
		print("file path %s doesn't exist, creating one..." % real_path)
	else:
		i=1
		while True:
			if os.path.exists(full_path+'('+str(i)+')'):
				i+=1
			else:
				full_path=full_path+'('+str(i)+')'
				os.mkdir(full_path)
				break

	filepaths = []
	for serial_number in camera:
		camera_filepath = full_path+'/'+'camera_' + serial_number+'.MOV'
		filepaths.append(camera_filepath)

	audio_filepath = full_path+'/'+'audio.tdms'
	filepaths.append(audio_filepath)
	return filepaths
