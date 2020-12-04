# import deeplabcut as dlc
import pandas as pd
from utils.geometry_utils import *
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

# get config setup
config_folder_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'
config = Config(config_folder_path)
circle_center = config.circle_center
corners = config.config_top_cam['corners']
inner_r = config.config_rig['inner_r']
outer_r = config.config_rig['outer_r_']
board_size = config.config_rig['board_size']

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


for i in range(data_length):
	output,flag = find_interection([between_earsx[i], between_earsy[i]],
							  [np.array(snoutx)[i], np.array(snouty)[i]],
							  circle_center,
							  inner_r=inner_r_pixel,
							  outer_r=outer_r_pixel)
	print(flag)
	degree.append(output)
	flags.append(flag)

degree=np.array(degree)
flags=np.array(flags)
print(np.sum(flags=='inner'))
print(np.sum(flags=='outer'))


