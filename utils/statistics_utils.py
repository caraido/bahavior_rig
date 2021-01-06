import scipy.io as sio
from scipy.stats import ttest_1samp,ttest_ind
import os
import mat4py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

	return mono_stats, bino_stats

def arrange_stats(data:pd.DataFrame):
	A=data.loc[data.index=='A','windowA']
	notA =data.loc[data.index!='A','windowA']

	B = data.loc[data.index == 'B', 'windowB']
	notB = data.loc[data.index != 'B', 'windowB']

	C = data.loc[data.index == 'C', 'windowC']
	notC = data.loc[data.index != 'A', 'windowC']

	with_mouse = pd.concat([A,B,C])
	without_mouse = pd.concat([notA,notB,notC])
	return with_mouse,without_mouse

def plotting(data_with:pd.DataFrame, data_without:pd.DataFrame,flag):
	fig,ax = plt.subplots()
	stat, p = ttest_ind(data_with.tolist(), data_without.tolist())
	words="%s, t stat: %.2f, p value: %.3f" % (flag,stat, p)
	print(words)
	#ax.boxplot([data_with.tolist(),data_without.tolist()])
	#ax.set_xticklabels(['mouse presented','empty chambers'])
	#ax.set_ylabel('frequency')
	ax.set_title(words)
	#ax.hist(data_with.tolist())
	this = list(data_with)
	that= list(data_without)
	full = this +that
	ax.hist(that,bins=5)
	ax.set_xlabel("empty windows")

	stat_with,p_with = ttest_1samp(data_with.tolist(),popmean=1/3)
	stat_without,p_without = ttest_1samp(data_without.tolist(),popmean=1/3)
	#print('%s, preference to the window with mouse presented. t stat: %.2f, p value: %.2f'%(flag,stat_with,p_with))
	#print('%s, preference to the empty window. t stat: %.2f, p value: %.2f' % (flag,stat_without, p_without))

	#plt.show()

if __name__ == '__main__':
	path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\multimedia\videos'
	items = os.listdir(path)

	mono=[]
	bino_right=[]
	bino_left=[]
	body_pref=[]
	win_pref = []

	for item in items:
		window_mouse = item[-1]
		if window_mouse in 'ABC' and 'Bully' not in item:
			rootpath = os.path.join(path,item)
			mono_stats,bino_stats=load_stats(rootpath)
			mono_stats = pd.DataFrame(mono_stats)
			bino_stats=pd.DataFrame(bino_stats)

			mono_right = mono_stats.loc['right',['windowA','windowB','windowC']]
			mono_right.name=window_mouse

			body = mono_stats.loc['body', ['windowA', 'windowB', 'windowC']]
			body.name=window_mouse
			mono_window_pref = mono_stats.loc["right",'window preference']

			right = bino_stats.loc['right', ['windowA', 'windowB', 'windowC']]
			right.name=window_mouse
			left = bino_stats.loc['left', ['windowA', 'windowB', 'windowC']]
			left.name=window_mouse

			window_pref = bino_stats.loc[:, 'window preference']
			window_pref.loc['mono']=mono_window_pref

			mono.append(mono_right)
			bino_right.append(right)
			bino_left.append(left)
			win_pref.append(window_pref)
			body_pref.append(body)

	mono=pd.DataFrame(mono)
	bino_right=pd.DataFrame(bino_right)
	bino_left = pd.DataFrame(bino_left)
	win_pref=pd.DataFrame(win_pref)
	body_pref = pd.DataFrame(body_pref)

	print("Single window preference vs. each other. Tested if there's significant difference between 'with mouse' group and 'empty windows' group")

	mono_with,mono_without = arrange_stats(mono)
	plotting(mono_with,mono_without,flag='binocular view')

	bino_left_with, bino_left_without = arrange_stats(bino_left)
	plotting(bino_left_with,bino_left_without,flag='left eye view')

	bino_right_with, bino_right_without = arrange_stats(bino_right)
	plotting(bino_right_with,bino_right_without,flag='right eye view')

	body_with, body_without = arrange_stats(body_pref)
	plotting(body_with,body_without,flag='body location')

	#plt.show()
	'''
	# window preference
	print("window preferences. Baseline is 1 which indicates no preference.")
	print("Value below 1 indicates aversion, beyond one indicates stronger preference")
	win_body = win_pref['body']
	stat,p=ttest_1samp(win_body,popmean=1)
	print("window preference for body location, t stat: %.2f, p value: %.2f" % (stat, p))
	plt.figure()
	plt.hist(win_body)
	plt.title("window preference for body location, t stat: %.2f, p value: %.2f" % (stat, p))

	win_right = win_pref['right']
	stat,p=ttest_1samp(win_right,popmean=1)
	print("window preference for right view, t stat: %.2f, p value: %.2f" % (stat, p))
	plt.figure()
	plt.hist(win_right,bins=5)
	plt.title("window preference for right view, t stat: %.2f, p value: %.2f" % (stat, p))

	win_left = win_pref['left']
	stat, p = ttest_1samp(win_left, popmean=1)
	print("window preference for left view, t stat: %.2f, p value: %.2f" % (stat, p))
	plt.figure()
	plt.hist(win_left,bins=5)
	plt.title("window preference for left view, t stat: %.2f, p value: %.2f" % (stat, p))

	win_mono = win_pref['mono']
	stat, p = ttest_1samp(win_mono, popmean=1)
	print("window preference for binocular view, t stat: %.2f, p value: %.2f" % (stat, p))
	plt.figure()
	plt.hist(win_mono,bins=5)
	plt.title("window preference for binocular view, t stat: %.2f, p value: %.2f" % (stat, p))

	plt.show()

	'''




