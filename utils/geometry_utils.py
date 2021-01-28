from sympy import *
import numpy as np
import cv2
import toml
import os
import pandas as pd
from functools import partial
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#import matlab.engine

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


class Config:
    def __init__(self,path:str,save_center=False):
        self.config_folder_path=path
        self.img=None
        self.config_rig=None
        self.config_top_cam=None
        self.load_config()
        self.circle_center=np.array(self.config_rig['recorded_center'])
        self.window_A = np.array(self.config_rig['window_A'])
        self.window_B = np.array(self.config_rig['window_B'])
        self.window_C = np.array(self.config_rig['window_C'])

        self.save_center=save_center

    def set_img(self,path:str):
        try:
            self.img = cv2.imread(path)
        except Exception:
            print("Wrong path! No image found")

    def load_config(self):
        dir = os.listdir(self.config_folder_path)
        for item in dir:
            if 'behavior_rig' in item:
                self.config_rig=toml.load(self.config_folder_path+'\\'+item)
            if 'extrinsic_17391304' in item:
                self.config_top_cam = toml.load(self.config_folder_path+'\\'+item)

        if self.config_top_cam is None or self.config_rig is None:
            raise Exception("one or both configuration file not found")

    def _save_center(self):
        if self.save_center:
            with open(self.config_folder_path+'\\config_behavior_rig.toml', 'a') as f:
                new_toml_string = toml.dump({'recorded_center':self.circle_center}, f)
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

    def recenter(self, coord):
        if self.circle_center is None:
            self.circle_center=np.array(self.config_rig['recorded_center'])
        else:
            self.circle_center=np.array(self.circle_center)

        coord=np.array(coord)
        new_coord = coord-self.circle_center
        return new_coord


def find_interection(combined,circle_center,inner_r, outer_r):
    origin=combined[0]
    point=combined[1]
    origin_= Point(origin[0],origin[1])
    point_ = Point(point[0],point[1])
    circle_center_ = Point(circle_center[0], circle_center[1])

    ray=Ray(origin_, point_)
    inner_circle = Circle(circle_center_, sympify(str(inner_r),rational=True))
    outer_circle = Circle(circle_center_, sympify(str(outer_r),rational=True))

    radius = np.sqrt((origin[0]-circle_center[0])**2 + (origin[1]-circle_center[1])**2)
    if radius>outer_r or radius<inner_r:
        angle=None
        flag='NaN'
    else:
        intersect=intersection(ray,inner_circle)

        if not intersect:
            intersect = intersection(ray,outer_circle)
            flag = 'outer'
        else:
            flag = 'inner'

        if len(intersect)>1:
            dics=[]
            for intsc in intersect:
                dics.append(intsc.distance(origin_))
            dics=np.array(dics).astype(np.float64)
            i=np.argmin(dics)
            intersect=intersect[i]
        else:
            intersect=intersect[0]

        intersect_center = np.array(intersect).astype(np.float64)-circle_center
        angle = np.arctan2(intersect_center[1],intersect_center[0])
    return angle, flag


class gaze_angle:
    def __init__(self,config_folder_path,flag='bino'):
        config = Config(config_folder_path)
        self.circle_center = config.circle_center
        self.corners = config.config_top_cam['corners']
        self.inner_r = config.config_rig['inner_r']
        self.outer_r = config.config_rig['outer_r_']
        self.board_size = config.config_rig['board_size']

        self.windowA = config.window_A
        self.windowB = config.window_B
        self.windowC = config.window_C

        self.windows = self._get_window_angle()

        # recalculate the pixel length of circle radius
        corners_pixel = get_board_side_length_pixel(self.corners)
        self.inner_r_pixel = get_r_pixel(self.inner_r, corners_pixel, self.board_size)
        self.outer_r_pixel = get_r_pixel(self.outer_r, corners_pixel, self.board_size)

        if flag=='mono' or flag=='bino':
            self.flag=flag
        else:
            raise ValueError("flag can only be mono or bino")

        self.inner=None
        self.outer=None
        self.processor=18

    def __call__(self,coord_path,save=True):
        # get the pose estimation path
        # coord_path = 'C:\\Users\\SchwartzLab\\Downloads\\Acclimation_videos_1\\'
        if self.flag=='bino':
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

            between_earsx = (np.array(leftearx) + np.array(rightearx)) / 2
            between_earsy = (np.array(lefteary) + np.array(righteary)) / 2

            # get the direction of the dot
            input_combine = [([between_earsx[i], between_earsy[i]], [np.array(snoutx)[i], np.array(snouty)[i]]) for i in
                             range(data_length)]

            result = self.bino_interect(input_combine)
            result = pd.DataFrame(result)
            self.inner = self.double(result, 'inner')
            self.outer = self.double(result, 'outer')

            result.columns = ["inner", 'outer']
            result['inner'] = self.inner
            result["outer"] = self.outer
        else:
            engine=matlab.engine.start_matlab()
            result = engine.poseAnalysis(coord_path,
                                         matlab.double([0.9]),
                                         matlab.int32([int(self.outer_r_pixel)]),
                                         matlab.int32([int(self.inner_r_pixel)]),
                                         self.circle_center.tolist()
                                         )
            result = pd.DataFrame(result)
            self.inner = self.double(result, 'inner')
            self.outer = self.double(result, 'outer')

            result.columns = ["inner", 'outer']
            result['inner'] = self.inner
            result["outer"] = self.outer
        # save file
        if save:
            self.save(coord_path, result)

        return result

    def double(self,result,flag='inner'):
        side = result[result[1]==flag][0]
        side_right = side[side>0]
        side_left = side[side<=0]
        side = side.append(side_right-2*np.pi)
        side = side.append(side_left+2*np.pi)
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

    def bino_interect(self,input_array):
        par = partial(find_interection,
                      circle_center=self.circle_center,
                      inner_r=self.inner_r_pixel,
                      outer_r=self.outer_r_pixel
                      )
        pool=Pool(processes=self.processor)
        result = pool.map(par, input_array)
        pool.close()
        pool.join()
        return result

    def save(self,path, data:pd.DataFrame):
        path =path+'\\gaze\\'
        if not os.path.exists(path):
            os.mkdir(path)
        data.to_csv(path+self.flag+'.csv')

    def plot(self,things:list):
        data = things[0]
        flag =things[1]
        length = len(things)
        if length != len(flag):
            raise Exception("length should be the same!")
        if length==2:
            row=1
            col=2
        elif length==4:
            row=2
            col=2
        else:
            raise Exception("not implemented yet")

        for i in range(length):
            plt.subplot(col, row, i)

            plt.axvspan(xmin=self.windows[0][0][0], xmax=self.windows[0][0][1], facecolor='orange', alpha=0.3)
            plt.axvspan(xmin=self.windows[1][0][0], xmax=self.windows[1][0][1], facecolor='red', alpha=0.3)
            plt.axvspan(xmin=self.windows[2][0][0], xmax=self.windows[2][0][1], facecolor='magenta', alpha=0.3)
            plt.axvspan(xmin=self.windows[0][1][0], xmax=self.windows[0][1][1], facecolor='orange', alpha=0.3)
            plt.axvspan(xmin=self.windows[1][1][0], xmax=self.windows[1][1][1], facecolor='red', alpha=0.3)
            plt.axvspan(xmin=self.windows[2][1][0], xmax=self.windows[2][1][1], facecolor='magenta', alpha=0.3)
            handles = [Rectangle((0, 0), 1, 1, color=c, ec="k", alpha=0.3) for c in ['orange', 'red', 'magenta']]
            labels = ["window A", "window B", "window C"]
            plt.legend(handles, labels)
            plt.title("Viewpoint density on %s wall" % flag)
            plt.hist(k=data[i], bins=60, density=True)

        plt.show()



if __name__ == '__main__':
    config_folder_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'
    img_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\test.png'
    config=Config(config_folder_path)
    config.set_img(img_path)
    config.get_loc_pixel()