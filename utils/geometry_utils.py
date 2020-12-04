from sympy import *
import numpy as np
import cv2
import toml
import os


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
    def __init__(self,path:str):
        self.config_folder_path=path
        self.img=None
        self.config_rig=None
        self.config_top_cam=None
        self.load_config()
        self.circle_center=np.array(self.config_rig['recorded_center'])

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
        with open(self.config_folder_path+'\\config_behavior_rig.toml', 'a') as f:
            new_toml_string = toml.dump({'recorded_center':self.circle_center}, f)
            print('saved')
            print(new_toml_string)

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

    def get_center(self):
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


def find_interection(circle_center,inner_r, outer_r):
    def wrapper(combined):
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
        return angle,flag
    return wrapper



if __name__ == '__main__':
    config_folder_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'
    img_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\test.png'
    config=Config(config_folder_path)
    config.set_img(img_path)
    config.get_center()