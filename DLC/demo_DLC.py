# import deeplabcut as dlc
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection


coord_path = 'C:\\Users\\SchwartzLab\\Downloads\\Acclimation_videos_1\\'
all_file = os.listdir(coord_path)
name= [a for a in all_file if '.csv' in a]
coord = coord_path+name[0]

data= pd.read_csv(coord,header=[1,2])

snoutx = data['snout']['x']
snouty = data['snout']['y']

rightearx = data['rightear']['x']
righteary = data['rightear']['y']

leftearx = data['leftear']['x']
lefteary = data['leftear']['y']

tailbasex = data['tailbase']['x']
tailbasey = data['tailbase']['y']

between_earsx = (np.array(leftearx)+np.array(rightearx))/2
between_earsy = (np.array(lefteary)+np.array(righteary))/2

head_dirx = np.array(snoutx)-between_earsx
head_diry = np.array(snouty)-between_earsy
head_point=np.array([head_dirx,head_diry])
angle = np.arctan2(head_diry,head_dirx)

snout = np.histogram2d(np.array(snoutx), np.array(snouty), bins=40)
tailbase = np.histogram2d(np.array(tailbasex), np.array(tailbasey), bins=40)

xedges=[0,1280]
yedges=[0,1080]
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.figure()
plt.imshow(snout[0].T, extent=extent, origin='upper')
plt.title('snout')


plt.figure()
plt.imshow(tailbase[0].T, extent=extent, origin='upper')
plt.title('tail base')


r = Affine2D().rotate_deg(180)

#fig,ax =plt.subplots(1,1)
ax=plt.gca()
ax.scatter(snoutx,snouty,c=angle,cmap='twilight',s=5)
'''
for x in ax.collections:
    #trans = x.get_transform()
    #x.set_transform(r+trans)
    if isinstance(x, PathCollection):
        transoff = x.get_offset_transform()
        x._transOffset = r+transoff
'''
ax.invert_yaxis()
#fig.colorbar(ax0,ax=ax)
plt.xlim([0,1000])
plt.ylim([0,1000])
#old = ax.axis()
#ax.axis(tuple(-np.array(old)))
plt.show()




