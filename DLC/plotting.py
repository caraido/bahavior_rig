import pandas as pd
from utils.geometry_utils import Config,get_r_pixel,get_board_side_length_pixel
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import numpy as np
from multiprocessing.pool import Pool
from functools import partial



if __name__=='__main__':
	# get config setup
	config_folder_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'
	config = Config(config_folder_path)
	circle_center = config.circle_center
	corners = config.config_top_cam['corners']
	inner_r = config.config_rig['inner_r']
	outer_r = config.config_rig['outer_r_']
	board_size = config.config_rig['board_size']
	window_A = config.window_A
	window_B = config.window_B
	window_C = config.window_C

	# get the pose estimation path
	coord_path = 'C:\\Users\\SchwartzLab\\Downloads\\Acclimation_videos_1\\'
	all_file = os.listdir(coord_path)
	name = [a for a in all_file if '.csv' in a]
	coord = coord_path + name[0]

	# read csv data
	data = pd.read_csv(coord, header=[1, 2])
	data_length = data.shape[0]

	# get the locations of the dots
	snoutx = data['snout']['x']
	snouty = data['snout']['y']

	rightearx = data['rightear']['x']
	righteary = data['rightear']['y']

	leftearx = data['leftear']['x']
	lefteary = data['leftear']['y']

	# tailbasex = data['tailbase']['x']
	# tailbasey = data['tailbase']['y']

	between_earsx = (np.array(leftearx) + np.array(rightearx)) / 2
	between_earsy = (np.array(lefteary) + np.array(righteary)) / 2

	# get the direction of the dot
	head_dirx = np.array(snoutx) - between_earsx
	head_diry = np.array(snouty) - between_earsy
	head_point = np.array([head_dirx, head_diry])
	angle = np.arctan2(head_diry, head_dirx)

	# recalculate the pixel length of circle radius
	corners_pixel = get_board_side_length_pixel(corners)
	inner_r_pixel = get_r_pixel(inner_r,corners_pixel,board_size)
	outer_r_pixel = get_r_pixel(outer_r,corners_pixel,board_size)

	degree=[]
	flags=[]
	input_combine=[([between_earsx[i],between_earsy[i]],[np.array(snoutx)[i], np.array(snouty)[i]]) for i in range(data_length)]

	par = partial(find_interection,circle_center=circle_center,inner_r=inner_r_pixel,outer_r=outer_r_pixel)
	pool=Pool(processes=18)
	result = pool.map(par,input_combine)
	pool.close()
	pool.join()

	result=pd.DataFrame(result)

	inner=result[result[1] =='inner'][0]
	outer = result[result[1] == 'outer'][0]

	inner_right = inner[inner>0]
	inner_left=inner[inner<=0]
	inner=inner.append(inner_right-2*np.pi)
	inner=inner.append(inner_left+2*np.pi)

	outer_right = outer[outer>0]
	outer_left = outer[outer<=0]
	outer=outer.append(outer_right-2*np.pi)
	outer=outer.append(outer_left+2*np.pi)

	window_A_center = np.array(window_A)-np.array(circle_center)
	window_B_center = np.array(window_B)-np.array(circle_center)
	window_C_center = np.array(window_C) - np.array(circle_center)

	window_A_angle = np.sort(np.arctan2(window_A_center[:,1],window_A_center[:,0]))
	window_B_angle = np.sort(np.arctan2(window_B_center[:, 1], window_B_center[:, 0]))
	window_C_angle = np.sort(np.arctan2(window_C_center[:, 1], window_C_center[:, 0]))

	window_A_angle_extra= np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_A_angle])
	window_B_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_B_angle])
	window_C_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_C_angle])

	'''
	plt.figure()
	plt.axvspan(xmin=window_A_angle[0], xmax=window_A_angle[1],facecolor='orange',alpha=0.3)
	plt.axvspan(xmin=window_B_angle[0], xmax=window_B_angle[1],facecolor='red',alpha=0.3)
	plt.axvspan(xmin=window_C_angle[0], xmax=window_C_angle[1],facecolor='magenta',alpha=0.3)
	plt.axvspan(xmin=window_A_angle_extra[0], xmax=window_A_angle_extra[1],facecolor='orange',alpha=0.3)
	plt.axvspan(xmin=window_B_angle_extra[0], xmax=window_B_angle_extra[1],facecolor='red',alpha=0.3)
	plt.axvspan(xmin=window_C_angle_extra[0], xmax=window_C_angle_extra[1],facecolor='magenta',alpha=0.3)
	handles = [Rectangle((0, 0), 1, 1, color=c, ec="k",alpha=0.3) for c in ['orange', 'red', 'magenta']]
	labels = ["window A", "window B", "window C"]
	plt.legend(handles, labels)
	plt.show()
	'''

	# saving the angles
	outer.to_csv(coord_path+"outer_angles.csv")
	inner.to_csv(coord_path+"inner_angles.csv")

	# historam -2pi to 2pi
	
	plt.subplot(2,1,1)

	plt.axvspan(xmin=window_A_angle[0], xmax=window_A_angle[1], facecolor='orange', alpha=0.3)
	plt.axvspan(xmin=window_B_angle[0], xmax=window_B_angle[1], facecolor='red', alpha=0.3)
	plt.axvspan(xmin=window_C_angle[0], xmax=window_C_angle[1], facecolor='magenta', alpha=0.3)
	plt.axvspan(xmin=window_A_angle_extra[0], xmax=window_A_angle_extra[1], facecolor='orange', alpha=0.3)
	plt.axvspan(xmin=window_B_angle_extra[0], xmax=window_B_angle_extra[1], facecolor='red', alpha=0.3)
	plt.axvspan(xmin=window_C_angle_extra[0], xmax=window_C_angle_extra[1], facecolor='magenta', alpha=0.3)
	handles = [Rectangle((0, 0), 1, 1, color=c, ec="k",alpha=0.3) for c in ['orange', 'red', 'magenta']]
	labels = ["window A", "window B", "window C"]
	plt.legend(handles, labels)
	plt.title("Viewpoint density on outer wall")
	plt.hist(outer, bins=60, density=True)
	
	plt.subplot(2,1,2)
	plt.hist(inner, bins=60, density=True)
	plt.axvspan(xmin=window_A_angle[0], xmax=window_A_angle[1], facecolor='orange', alpha=0.3)
	plt.axvspan(xmin=window_B_angle[0], xmax=window_B_angle[1], facecolor='red', alpha=0.3)
	plt.axvspan(xmin=window_C_angle[0], xmax=window_C_angle[1], facecolor='magenta', alpha=0.3)
	plt.axvspan(xmin=window_A_angle_extra[0], xmax=window_A_angle_extra[1], facecolor='orange', alpha=0.3)
	plt.axvspan(xmin=window_B_angle_extra[0], xmax=window_B_angle_extra[1], facecolor='red', alpha=0.3)
	plt.axvspan(xmin=window_C_angle_extra[0], xmax=window_C_angle_extra[1], facecolor='magenta', alpha=0.3)
	handles = [Rectangle((0, 0), 1, 1, color=c, ec="k",alpha=0.3) for c in ['orange', 'red', 'magenta']]
	labels = ["window A", "window B", "window C"]
	plt.legend(handles, labels)
	plt.title("Viewpoint density on inner wall")

	plt.show()



