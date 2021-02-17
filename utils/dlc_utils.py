import deeplabcut
import os
from utils.geometry_utils import Gaze_angle


def dlc_analysis(root_path, dlc_config_path):
	processed_path = os.path.join(root_path, 'processed')
	config_path = os.path.join(root_path,'config')
	things = os.listdir(processed_path)
	movie = [a for a in things if '.MOV' in a and '17391304']
	if len(movie)==0:
		things=os.listdir(root_path)
		movie= [a for a in things if '.MOV' in a and '17391304']
		print("didn't find the processed videos. Analyzing on the raw video")
	movie_path = os.path.join(processed_path, movie[0])

	deeplabcut.analyze_videos(dlc_config_path,
							  [movie_path],
							  save_as_csv=True,
							  videotype='mov',
							  shuffle=1,
							  gputouse=0)
	deeplabcut.create_labeled_video(dlc_config_path,
									[movie_path],
									save_frames=False,
									trailpoints=5,
									videotype='mov',
									draw_skeleton='True')

	gaze_model =Gaze_angle(config_path)
	gaze_model.gazePoint=0.5725
	bino=gaze_model(processed_path,save=True)
	gaze_model.gazePoint=0
	mono=gaze_model(processed_path,save=True)

	gaze_model.plot(bino)
	gaze_model.plot(mono)





