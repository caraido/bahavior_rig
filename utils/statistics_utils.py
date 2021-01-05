import scipy.io as sio
#from scipy.stats import *
import os
import mat4py
import pandas as pd

def load_stats(root_path):
	gaze_path = os.path.join(root_path,'gaze')
	if not os.path.exists(gaze_path):
		gaze_path=os.path.join(root_path,'processed','gaze')
	items=os.listdir(gaze_path)

	mono = [a for a in items if '0' in a and '.mat' in a]
	bino = [a for a in items if '32' in a and '.mat' in a]

	mono=mono[0]
	bino=bino[0]

	mono_result = mat4py.loadmat(os.path.join(gaze_path,mono))
	bino_result = mat4py.loadmat(os.path.join(gaze_path,bino))

	mono_stats = mono_result['stats']
	bino_stats = bino_result['stats']

	return mono_stats,bino_stats

if __name__ == '__main__':
	path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\multimedia\videos'
	items = os.listdir(path)

	mono=[]
	bino_right=[]
	bino_left=[]
	body=[]
	win_pref = []

	for item in items:
		window_mouse = item[-1]
		if window_mouse in 'ABC':
			rootpath = os.path.join(path,item)
			mono_stats,bino_stats=load_stats(rootpath)
			mono_stats = pd.DataFrame(mono_stats)
			bino_stats=pd.DataFrame(bino_stats)

			mono_right = mono_stats.loc['right',['windowA','windowB','windowC']]
			mono_right.name=window_mouse

			body = mono_stats.loc['body', ['windowA', 'windowB', 'windowC']]
			mono_window_pref = mono_stats.loc[:,'window preference']

			right = bino_stats.loc['right', ['windowA', 'windowB', 'windowC']]
			right.name=window_mouse
			left = bino_stats.loc['left', ['windowA', 'windowB', 'windowC']]
			left.name=window_mouse

			bino_window_pref = bino_stats.loc[:, 'window preference']

			mono.append(mono_right)
			bino_right.append(right)
			bino_left.append(left)

	mono=pd.DataFrame(mono)
	bino_right=pd.DataFrame(bino_right)



