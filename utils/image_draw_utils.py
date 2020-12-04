import PIL
import cv2
import colorcet as cc

class Display:

	def __init__(self,cmap='bmy',radius=6,pcutoff=0.5):
		self.cmap=cmap
		self.colors=None
		self.radius=radius
		self.pcutoff=pcutoff

	# currently useless due to the gray scale display setting
	def set_display(self,bodyparts):
		all_colors=getattr(cc,self.cmap)
		self.colors=all_colors[:: int(len(all_colors)/bodyparts)]

	def draw_dots(self,frame, pose=None):
		for i in range(pose.shape[0]):
			if pose[i, 2] > self.pcutoff:
				try:
					x=pose[i,0]
					y=pose[i,1]
					cv2.circle(frame,(int(x),int(y)),self.radius,255,-1)
				except Exception as e:
					print(e)


def draw_dots(frame, pose=None):
	if pose is not None:
		for i in range(pose.shape[0]):
			if pose[i, 2] > 0.7:
				try:
					x=pose[i,0]
					y=pose[i,1]
					cv2.circle(frame, (int(x),int(y)), 6, 255, -1)
				except Exception as e:
					print(e)