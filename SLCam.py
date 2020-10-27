import cv2
import PySpin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ffmpeg
import utils.calibration_utils as cau
import os
import toml
import threading
from io import BytesIO
from PIL import Image


class CharucoBoard:
    def __init__(self, x,y, marker_size=0.8):
        self.x=x
        self.y=y
        self.marker_size = marker_size
        self.default_dictionary = cv2.aruco.DICT_4X4_50  # default
        self.seed = 0
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            self.default_dictionary)

    @property
    def board(self):
        this_board = cv2.aruco.CharucoBoard_create(self.x,
                                                   self.y,
                                                   1,
                                                   self.marker_size,
                                                   self.dictionary)
        return this_board

    @property
    def marker_size(self):
        return self._marker_size

    @marker_size.setter
    def marker_size(self, value):
        if value <= 0 or value >= 1:
            raise ValueError("this value can only be set between 0 ~ 1!")
        else:
            self._marker_size = value

    def save_board(self, img_size=1000):
        if self.default_dictionary == 0:
            file_name = 'charuco_board_shape_' + str(self.x) + 'x' + str(self.y) + '_marker_size_' + str(
                self.marker_size) + '_default.png'
            img = self.board.draw((img_size, img_size))
            result = cv2.imwrite('./multimedia/board/' + file_name, img)
            if result:
                print('save board successfully! Name: ' + file_name)
            else:
                raise Exception('save board failed! Name: '+file_name)

    def print_board(self):
        img = self.board.draw((1000, 1000))
        plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        plt.axis("off")
        plt.show()


class Calib:
    def __init__(self,calib_type):
        self._get_type(calib_type)

        self.allCorners = []
        self.allIds = []
        self.decimator = 0
        self.config = None

        self.board = CharucoBoard(x=6,y=2).board

        self.max_size = cau.get_expected_corners(self.board)
        self.save_path = './config/config_'+self.type+'_'
        self.load_path = './config/'

    def _get_type(self, calib_type):
        if calib_type == 'intrinsic'or calib_type == 'extrinsic':
            self.type = calib_type
        else:
            raise ValueError("type can only be intrinsic or extrinsic!")

    def reset(self):
        del self.allIds, self.allCorners, self.decimator, self.config
        self.allCorners = []
        self.allIds = []
        self.decimator = 0
        self.config=None

    def load_config(self, camera_serial_number):
        if not os.path.exists(self.load_path):
            os.mkdir(self.load_path)
            raise Warning("config directory doesn't exist. creating one...")

        items = os.listdir(self.load_path)
        for item in items:
            if camera_serial_number in item and self.type in item:
                path = self.load_path+'config_' + self.type + '_' + camera_serial_number + '.toml'
                with open(path,'r') as f:
                    self.config = toml.load(f)  # there only should be only one calib file for each camera

    def save_config(self, camera_serial_number, width, height):
        save_path = self.save_path + camera_serial_number + '.toml'
        if os.path.exists(save_path):
            print('\n config file already exists.')
        else:
            if self.type == "intrinsic":
                param = cau.quick_calibrate(self.allCorners,
                                            self.allIds,
                                            self.board,
                                            width,
                                            height)
                param['camera_serial_number'] = camera_serial_number
                with open(save_path, 'w') as f:
                    toml.dump(param, f)
                print('intrinsic calibration configuration saved!')
            else:
                param = {'corners': self.allCorners,
                         'ids': self.allIds, 'CI': 5,
                         'camera_serial_number': camera_serial_number}
                with open(save_path, 'w') as f:
                    toml.dump(param, f)
                print('extrinsic calibration configuration saved!')


class Camera:

    def __init__(self, camlist, index):
        self._spincam = camlist.GetByIndex(index)
        self._spincam.Init()

        # here we will eventually want to enable hardware triggering
        # for now we'll just hardcode the framerate at 30
        self._spincam.AcquisitionFrameRateEnable.SetValue(True)
        self._spincam.AcquisitionFrameRate.SetValue(30)

        self.device_serial_number, self.height, self.width = self.get_camera_property()
        self.in_calib = Calib('intrinsic')
        self.ex_calib = Calib('extrinsic')

        self._running = False
        self._running_lock = threading.Lock()

        self._saving = False
        self.file = None

        self._displaying = False
        self.frame = None
        self.frame_count = 0
        self._frame_lock = threading.Lock()
        self._frame_bytes = BytesIO()

        self._in_calibrating = False
        self._ex_calibrating = False

    def start(self, filepath=None, display=False):
        if filepath:
            self._saving = True

            # we will assume hevc for now
            # will also assume 30fps
            self.file = ffmpeg \
                .input('pipe:', format='rawvideo', pix_fmt='gray', s='1280x1024') \
                .output(filepath, vcodec='libx265') \
                .overwrite_output() \
                .run_async(pipe_stdin=True)
            # self.file = cv2.VideoWriter(
            #     filepath, cv2.VideoWriter_fourcc(*'hvc1'), 30, (1024, 1280), False)

        if display:
            self._displaying = True

        with self._running_lock:
            if not self._running:
                self._running = True
                self._spincam.BeginAcquisition()

    def stop(self):
        with self._running_lock:
            if self._running:
                if self._saving:
                    self._saving = False
                    # self.file.release()
                    self.file.stdin.close()
                    self.file.wait()
                    del self.file
                    self.file = None

                cv2.destroyAllWindows()
                self._running = False
                self._displaying = False

                self._spincam.EndAcquisition()
                self._spincam.DeInit()

    def capture(self):
        im = self._spincam.GetNextImage()
        # parse to make sure that image is complete....
        if im.IsIncomplete():
            status = im.GetImageStatus()
            im.Release()
            raise Exception(f"Image incomplete with image status {status} ...")

        frame = np.reshape(im.GetData(), (self.height, self.width))
        if self._saving:
            self.save(frame)

        # press "i" or "e" key to turn on or off calibration mode
        if cv2.waitKey(1) & 0xFF == ord('c'):
            self.extrinsic_calibration_switch()
        if cv2.waitKey(1) & 0xFF == ord('i'):
            self.intrinsic_calibration_switch()

        # check calibration status
        if self._in_calibrating and self._ex_calibrating:
            raise Warning('Only one type of calibration can be turned on!')

        # intrinsic calibration
        if self._in_calibrating and not self._ex_calibrating:
            self.intrinsic_calibration(frame)

        # extrinsic calibration
        if self._ex_calibrating and not self._in_calibrating:
            self.extrinsic_calibration(frame)

        if self._displaying:
            # acquire lock on frame
            with self._frame_lock:
                self.frame = frame
                self.frame_count += 1
            self.display()

        im.Release()

    def save(self, frame):
        self.file.stdin.write(frame.tobytes())

    def get_camera_property(self):
        nodemap_tldevice = self._spincam.GetTLDeviceNodeMap()
        device_serial_number = PySpin.CStringPtr(
            nodemap_tldevice.GetNode('DeviceSerialNumber')).GetValue()
        nodemap = self._spincam.GetNodeMap()
        height = PySpin.CIntegerPtr(nodemap.GetNode('Height')).GetValue()
        width = PySpin.CIntegerPtr(nodemap.GetNode('Width')).GetValue()
        return device_serial_number, height, width

    def intrinsic_calibration_switch(self):
        if not self._in_calibrating:
            print('turning on intrinsic calibration mode')
            self._in_calibrating = True
            self.in_calib.reset()
        else:
            print('turning off intrinsic calibration mode')
            self._in_calibrating = False
            self.in_calib.save_config(self.device_serial_number,
                                      self.width,
                                      self.height)

    def extrinsic_calibration_switch(self):
        if not self._ex_calibrating:
            print('turning on extrinsic calibration mode')
            self._ex_calibrating = True
            self.ex_calib.reset()
            self.ex_calib.load_config(self.device_serial_number)
        else:
            print('turning off extrinsic calibration mode')
            self._ex_calibrating = False
            self.ex_calib.save_config(self.device_serial_number,
                                      self.width,
                                      self.height)

    def intrinsic_calibration(self, frame):
        # write something on the frame
        text = 'Intrinsic calibration mode On'
        cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 125), 2)

        # key step: detect markers
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 100
        params.adaptiveThreshWinSizeMax = 700
        params.adaptiveThreshWinSizeStep = 50
        params.adaptiveThreshConstant = 5

        # get corners and refine them in openCV
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            frame, self.in_calib.board.dictionary, parameters=params)
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
            cv2.aruco.refineDetectedMarkers(frame, self.in_calib.board, corners, ids,
                                            rejectedImgPoints, parameters=params)

        # interpolate corners and draw corners
        if len(detectedCorners) > 0:
            rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
                detectedCorners, detectedIds, frame, self.in_calib.board)
            if detectedCorners is not None and 2 <= len(
                    detectedCorners) <= self.in_calib.max_size and self.in_calib.decimator % 3 == 0:
                self.in_calib.allCorners.append(detectedCorners)
                self.in_calib.allIds.append(detectedIds)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)
        self.in_calib.decimator += 1

        return frame

    def extrinsic_calibration(self,frame):
        if self.ex_calib.config is None:
            text = 'No configuration file found. Performing initial extrinsic calibration... '
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            # key step: detect markers
            params = cau.get_calib_param()

            # get corners and refine them in openCV
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                frame, self.ex_calib.board.dictionary, parameters=params)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)
            self.ex_calib.allCorners = corners
            self.ex_calib.allIds = ids
        else:
            text = 'Found configuration file for this camera. Calibrating...'
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

            truecorners=self.ex_calib.config['corners']
            trueids = self.ex_calib.config['ids']
            CI = self.ex_calib.config['CI'] # pixels

            # key step: detect markers
            params = cau.get_calib_param()
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                frame, self.ex_calib.board.dictionary, parameters=params)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

            # check if aligned:

            for id, corner in zip(ids, corners):
                color = cau.check_aligned(id, corner, trueids,truecorners,CI)
                cv2.rectangle(frame, truecorners[i][0], truecorners[i][[2]], color, 5)

    def display(self):
        cv2.imshow('frame', self.frame)
        '''
        with self._frame_lock:
            frame_count = self.frame_count  # get the starting number of frames
        while True:
            with self._frame_lock:
                if self.frame_count > frame_count:  # display the frame if it's new ~ might run into issues here?
                    frame_count = self.frame_count
                    self._frame_bytes.seek(0)  # go to the beginning of the buffer
                    Image.fromarray(self.frame).save(self._frame_bytes, 'bmp')
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + self._frame_bytes.getvalue() + b'\r\n'
        '''

    def run(self):
        while True:
            with self._running_lock:
                if self._running:
                    self.capture()
                else:
                    return

    def __del__(self):
       self.stop()


class CameraGroup:
    def __init__(self):
        self._system = PySpin.System.GetInstance()
        self._camlist = self._system.GetCameras()
        self.nCameras = self._camlist.GetSize()
        self.cameras = [Camera(self._camlist, i)
                        for i in range(self.nCameras)]

    def start(self, filepaths=None, isDisplayed=None):
        if not filepaths:
            filepaths = [None] * self.nCameras
        if not isDisplayed:
            isDisplayed = [False] * self.nCameras

        for cam, fp, disp in zip(self.cameras, filepaths, isDisplayed):
            cam.start(filepath=fp, display=disp)

    def stop(self):
        for cam in self.cameras:
            cam.stop()

    def __del__(self):
        for cam in self.cameras:
            cam.stop()
            del cam
        self._camlist.Clear()
        #self._system.ReleaseInstance()


if __name__ == '__main__':
     board = CharucoBoard(6,2)
     #board.save_board(2000)
     cg = CameraGroup()

     for i, cam in enumerate(cg.cameras):
         # cam.start(filepath=f'testing{i:02d}.mov')
         cam.start(display=True)

     for j in range(500):
         for i, cam in enumerate(cg.cameras):
             cam.capture()
     for i, cam in enumerate(cg.cameras):
         cam.stop()

     del cg
#     for i, cam in enumerate(cg.cameras):
#        cam.stop()
