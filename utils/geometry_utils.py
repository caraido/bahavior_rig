
import numpy as np
import cv2
import toml
import os
import pandas as pd
from sympy import Circle,Point
from utils.head_angle_analysis import project_from_head_to_walls,is_in_window

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pickle as pk
from scipy import io as sio


def get_board_side_length_pixel(corners):
    corners=np.array(corners)
    shape = corners.shape
    length = []
    for d in range(shape[0]):
        one =corners[d][0]
        length.append(np.sqrt((one[0][0]-one[1][0])**2 + (one[0][1]-one[1][1])**2))
        length.append(np.sqrt((one[1][0] - one[2][0]) ** 2 + (one[1][1] - one[2][1]) ** 2))
        length.append(np.sqrt((one[2][0] - one[3][0]) ** 2 + (one[2][1] - one[3][1]) ** 2))
        length.append(np.sqrt((one[3][0] - one[0][0]) ** 2 + (one[3][1] - one[0][1]) ** 2))

    average_length = np.mean(length)
    return average_length


def get_r_pixel(r, corner_pixel, board):
    resize = corner_pixel/board * r
    return resize


def find_intersect(board1_xy, board2_xy, board1_dist, board2_dist):
    board1_xy = Point(board1_xy[0],board1_xy[1])
    board2_xy = Point(board2_xy[0], board2_xy[1])
    circle1 = Circle(board1_xy,board1_dist)
    circle2 = Circle(board2_xy, board2_dist)
    intersect = circle1.intersect(circle2)
    intersect = np.array(intersect.args,dtype=float)
    return intersect

def find_window(rootpath):
    config_folder_path = os.path.join(rootpath,'config')
    items=os.listdir(config_folder_path)
    extrinsic = [a for a in items if 'extrinsic' in a and '17391304' in a]
    rig = [a for a in items if 'rig' in a]

    if len(extrinsic) != 0 and len(rig)==1:
        with open(os.path.join(config_folder_path,extrinsic[0]),'r') as f:
            config = toml.load(f)
        try:
            new_corners = np.array(config['new_corners'])
        except:
            new_corners = np.array(config['corners'])

        if new_corners.shape != (6,1,4,2):
            raise Exception("can't proceed! 6")
        with open(os.path.join(config_folder_path,rig[0]),'r') as f:
            rig = toml.load(f)

def find_board_center(corners:np.ndarray):
    pass



def _triangle_area(point1,point2,point3):
    a=np.linalg.norm(point1-point2,axis=0)
    b=np.linalg.norm(point3-point2,axis=0)
    c=np.linalg.norm(point1-point3,axis=0)
    s=(a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area

def triangle_area(pose):
    snout_x = pose['snout']['x'].to_numpy()
    snout_y = pose['snout']['y'].to_numpy()

    leftear_x = pose['leftear']['x'].to_numpy()
    leftear_y = pose['leftear']['y'].to_numpy()

    rightear_x = pose['rightear']['x'].to_numpy()
    rightear_y = pose['rightear']['y'].to_numpy()

    tailbase_x = pose['rightear']['x'].to_numpy()
    tailbase_y = pose['rightear']['y'].to_numpy()

    snout=np.array([snout_x,snout_y])
    leftear = np.array([leftear_x, leftear_y])
    rightear = np.array([rightear_x, rightear_y])
    tailbase = np.array([tailbase_x, tailbase_y])

    head_triangle_area = _triangle_area(snout,leftear,rightear)
    body_triangle_area = _triangle_area(tailbase,leftear,rightear)

    return head_triangle_area,body_triangle_area


def body_center(pose):
    leftear_x = pose['leftear']['x'].to_numpy()
    leftear_y = pose['leftear']['y'].to_numpy()

    rightear_x = pose['rightear']['x'].to_numpy()
    rightear_y = pose['rightear']['y'].to_numpy()

    center = np.array([(leftear_x+rightear_x)/2,(leftear_y+rightear_y)/2])
    return center

class Config:
    def __init__(self,path:str,save_center=False):
        self.config_folder_path=path
        self.img=None
        self.config_rig=None
        self.config_top_cam=None
        self._load_config()
        if self.config_rig:
            self.circle_center=np.array(self.config_rig['recorded_center'])
            self.window_A = np.array(self.config_rig['window_A'])
            self.window_B = np.array(self.config_rig['window_B'])
            self.window_C = np.array(self.config_rig['window_C'])

        self.save_center=save_center

    def set_img(self,path:str):
        self.img = cv2.imread(path)
        if self.img is None:
            cap=cv2.VideoCapture(path)
            ret,frame = cap.read()
            self.img=frame
            cap.release()

    def _load_config(self):
        last = os.path.split(self.config_folder_path)[-1]
        if '.' not in last:
            dir = os.listdir(self.config_folder_path)
            for item in dir:
                if 'behavior_rig' in item:
                    path = os.path.join(self.config_folder_path, item)
                    self.config_rig=toml.load(path)
                if 'extrinsic_17391304' in item:
                    path = os.path.join(self.config_folder_path,item)
                    self.config_top_cam = toml.load(path)
        elif '.toml' in last:
            if 'behavior_rig' in last:
                self.config_rig = toml.load(self.config_folder_path)
            if 'extrinsic_17391304' in last:
                self.config_top_cam = toml.load(self.config_folder_path)


        if self.config_top_cam is None or self.config_rig is None:
            print("one or both configuration file not found")

    def _save_center(self):
        if self.save_center:
            if os.path.exists(self.config_folder_path+'\\config_behavior_rig.toml'):
                with open(self.config_folder_path+'\\config_behavior_rig.toml', 'a') as f:
                    new_toml_string = toml.dump({'recorded_center':self.circle_center}, f)
                    print('saved')
                    print(new_toml_string)
            else:
                with open(self.config_folder_path+'\\config_behavior_rig.toml', 'w') as f:
                    new_toml_string = toml.dump({'recorded_center': self.circle_center}, f)
                    print('saved')
                    print(new_toml_string)

        else:
            pass

    def on_EVENT_LBUTTONDOWN(self, event, x, y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            self.circle_center=[x,y]
            self._save_center()
            print(xy)
            cv2.circle(self.img, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(self.img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)
            cv2.imshow("image", self.img)

    def get_loc_pixel(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.on_EVENT_LBUTTONDOWN)
        cv2.imshow("image", self.img)
        while True:
            try:
                cv2.waitKey(100)
            except Exception:
                cv2.destroyWindow("image")
                break


class Gaze_angle:
    def __init__(self,main_config, config_folder_path,gazePoint=0.5725):
        main_config = Config(main_config)
        #self.circle_center = config.circle_center
        self.corners = main_config.config_top_cam['corners']
        self.inner_r = main_config.config_rig['inner_r']
        self.outer_r = main_config.config_rig['outer_r_']
        self.board_size = main_config.config_rig['board_size']

        #self.windowA = config.window_A
        #self.windowB = config.window_B
        #self.windowC = config.window_C
        local_config = toml.load(config_folder_path)
        self.windowA = np.array([local_config['A1'],local_config['A2']])
        self.windowB = np.array([local_config['B1'], local_config['B2']])
        self.windowC = np.array([local_config['C1'], local_config['C2']])
        self.circle_center = np.array(local_config['recorded_center'])

        self.windows = self._get_window_angle()

        # recalculate the pixel length of circle radius
        corners_pixel = get_board_side_length_pixel(self.corners)
        self.inner_r_pixel = get_r_pixel(self.inner_r, corners_pixel, self.board_size)
        self.outer_r_pixel = get_r_pixel(self.outer_r, corners_pixel, self.board_size)

        self.inner=None
        self.outer=None
        self.title_name=None
        self.gazePoint=gazePoint

       # self.engine = matlab.engine.start_matlab()

    def __call__(self,root_path, save=True):
        # get the pose estimation path
        # coord_path = 'C:\\Users\\SchwartzLab\\Downloads\\Acclimation_videos_1\\'
        self.title_name = os.path.split(root_path)[-1]
        all_file = os.listdir(root_path)
        name = [a for a in all_file if 'second' in a and 'csv' in a]
        coord = os.path.join(root_path,name[0])
        pose = pd.read_csv(coord,header=[1,2])


        # gaze
        inner_left,outer_left, inner_right, outer_right = project_from_head_to_walls(pose,
                                                                                      self.inner_r_pixel,
                                                                                      self.outer_r_pixel,
                                                                                      self.circle_center[np.newaxis, :],
                                                                                      gazePoint=self.gazePoint)
        # triangle area
        head_triangle_area,body_triangle_area = triangle_area(pose)

        # body center
        body = body_center(pose)
        recenter_body = body-self.circle_center[:,np.newaxis]
        arc_body = np.arctan2(recenter_body[1],recenter_body[0])[:,np.newaxis]

        # in window
        A_right = np.sum(is_in_window(outer_right,self.windowA,self.circle_center))
        B_right = np.sum(is_in_window(outer_right, self.windowB, self.circle_center))
        C_right = np.sum(is_in_window(outer_right, self.windowC, self.circle_center))

        A_left = np.sum(is_in_window(outer_left, self.windowA, self.circle_center))
        B_left = np.sum(is_in_window(outer_left, self.windowB, self.circle_center))
        C_left = np.sum(is_in_window(outer_left, self.windowC, self.circle_center))

        A_body = np.sum(is_in_window(arc_body, self.windowA, self.circle_center))
        B_body = np.sum(is_in_window(arc_body, self.windowB, self.circle_center))
        C_body = np.sum(is_in_window(arc_body, self.windowC, self.circle_center))

        A_weight_right = A_right /  (A_right + B_right + C_right)
        B_weight_right = B_right / (A_right + B_right + C_right)
        C_weight_right = C_right / (A_right + B_right + C_right)

        A_weight_left = A_left / (A_left + B_left + C_left)
        B_weight_left = B_left / (A_left + B_left + C_left)
        C_weight_left = C_left / (A_left + B_left + C_left)

        A_weight_body = A_body / (A_body + B_body + C_body)
        B_weight_body = B_body / (A_body + B_body + C_body)
        C_weight_body = C_body / (A_body + B_body + C_body)

        winPreference_right = ((A_right + B_right + C_right) / len(pd.isna(outer_right))) * 360 / 99
        winPreference_left = ((A_left + B_left + C_left) / len(pd.isna(outer_left))) * 360 / 99
        winPreference_body = ((A_body + B_body + C_body) / len(pd.isna(arc_body))) * 360 / 99

        table = np.array([A_weight_right,B_weight_right,C_weight_right,winPreference_right,
                          A_weight_left,B_weight_left,C_weight_left,winPreference_left,
                          A_weight_body,B_weight_body,C_weight_body,winPreference_body
                          ]).reshape([3,4])
        column = ['windowA','windowB','windowC','window preference']
        index = ['right','left','body']
        stats=pd.DataFrame(table,columns=column,index=index)
        stats=stats.to_dict()

        result = {'inner_left': inner_left,
                  'inner_right': inner_right,
                  'outer_left': outer_left,
                  'outer_right': outer_right,
                  'head_triangle_area':head_triangle_area,
                  'body_triangle_area':body_triangle_area,
                  'body_position':arc_body,
                  'stats':stats
                  }

        # save file
        if save:
            self.save(root_path, result)
            print('saved')

        return result

    def double(self,side):
        side=side[pd.notna(side)]
        side_right = side[side > 0]
        side_left = side[side <= 0]
        side = np.append(side, side_right - 2 * np.pi)
        side = np.append(side, side_left + 2 * np.pi)
        return side

    def _get_window_angle(self):
        window_A_center = np.array(self.windowA) - np.array(self.circle_center)
        window_B_center = np.array(self.windowB) - np.array(self.circle_center)
        window_C_center = np.array(self.windowC) - np.array(self.circle_center)

        window_A_angle = np.sort(np.arctan2(window_A_center[:, 1], window_A_center[:, 0]))
        window_B_angle = np.sort(np.arctan2(window_B_center[:, 1], window_B_center[:, 0]))
        window_C_angle = np.sort(np.arctan2(window_C_center[:, 1], window_C_center[:, 0]))

        window_A_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_A_angle])
        window_B_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_B_angle])
        window_C_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_C_angle])

        windowA = [window_A_angle,window_A_angle_extra]
        windowB = [window_B_angle, window_B_angle_extra]
        windowC = [window_C_angle, window_C_angle_extra]

        combine = [windowA, windowB, windowC]

        return combine

    def save(self,path, data):
        path =os.path.join(path,'gaze')
        if not os.path.exists(path):
            os.mkdir(path)
        if isinstance(data,pd.DataFrame):
            data.to_csv(os.path.join(path,'all_gaze_angle_%d.csv'%int(self.gazePoint*180/np.pi)))
        else:
            filename = os.path.join(path,'all_gaze_angle_%d.mat'%int(self.gazePoint*180/np.pi))
            matname = 'gaze_angle_%d.mat'%int(self.gazePoint*180/np.pi)
            sio.savemat(filename,{matname:data})

    def plot(self,things:dict,savepath=None,show=False):

        keys = ['inner_left','inner_right','outer_left','outer_right','body_position']
        row=4
        col=2

        plt.figure(figsize=(14, 16))
        for i in range(len(keys)):
            plt.subplot(row, col, i+1)

            plt.axvspan(xmin=self.windows[0][0][0], xmax=self.windows[0][0][1], facecolor='orange', alpha=0.3)
            plt.axvspan(xmin=self.windows[1][0][0], xmax=self.windows[1][0][1], facecolor='red', alpha=0.3)
            plt.axvspan(xmin=self.windows[2][0][0], xmax=self.windows[2][0][1], facecolor='magenta', alpha=0.3)
            plt.axvspan(xmin=self.windows[0][1][0], xmax=self.windows[0][1][1], facecolor='orange', alpha=0.3)
            plt.axvspan(xmin=self.windows[1][1][0], xmax=self.windows[1][1][1], facecolor='red', alpha=0.3)
            plt.axvspan(xmin=self.windows[2][1][0], xmax=self.windows[2][1][1], facecolor='magenta', alpha=0.3)
            handles = [Rectangle((0, 0), 1, 1, color=c, ec="k", alpha=0.3) for c in ['orange', 'red', 'magenta']]
            labels = ["window A", "window B", "window C"]
            plt.legend(handles, labels)
            plt.title("Gaze point density of %s " % list(keys)[i])
            plt.hist(x=self.double(things[list(keys)[i]]), bins=60, density=False)
        plt.suptitle(self.title_name +' gaze angle %d'%float(self.gazePoint*180/np.pi),fontsize=30)

        plt.subplot(4,1,4)
        plt.axis('off')
        stats = things['stats']
        stats = pd.DataFrame(stats)

        cell_text = []
        for row in range(len(stats)):
            cell_text.append(['%1.2f' % x for x in stats.iloc[row]])

        table = plt.table(cellText=cell_text,colLabels=stats.columns,rowLabels=stats.index,loc='center',cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1,3)

        if savepath:
            plt.savefig(os.path.join(savepath,'gaze','gaze_angle_%d.jpg'%float(self.gazePoint*180/np.pi)))

        if show:
            plt.show()

if __name__ == '__main__':
    #import matlab.engine
    '''
    config_folder_path = r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\config'
    img_path = r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\test.png'
  
    path = r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\multimedia\\unprocessed_videos'
    folders = os.listdir(path)[8:]
    for folder in folders:
        ob = Config(path + '\\' + folder, save_center=True)
        video_path = path + '\\' + folder + '\\camera_17391304.MOV'
        ob.set_img(video_path)
        ob.get_loc_pixel()

    '''
    videopaths =  'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\multimedia\\videos'
    config_folder_path = 'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\config'

    videos = os.listdir(videopaths)

    for video in videos:
        if not video.startswith('.'):
            config_folder_path2 = os.path.join(videopaths, video ,'config_behavior_rig.toml')
            gaze_model = Gaze_angle(main_config=config_folder_path,config_folder_path=config_folder_path2,gazePoint=0)
            result = gaze_model(os.path.join(videopaths,video),save=True)
            gaze_model.plot(result,savepath=os.path.join(videopaths,video),show=False)
            #Gaze_model.plot(result,flag='mono', savepath=videopaths + '\\' + video)










