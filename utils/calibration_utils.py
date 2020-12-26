import cv2
import numpy as np
import itertools
from time import time
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import toml
import pandas as pd

CALIB_UPDATE_EACH = 3 # frame interval for calibration update

def get_expected_corners(board):
  board_size = board.getChessboardSize()
  return (board_size[0] - 1) * (board_size[1] - 1)


def check_ids(ids):
  if ids is None or ids == []:
    return False
  else:
    return True


def check_corners(corners):
  check = []
  for corner in corners:
    if np.shape(corner)[1] == 4:
      check.append(True)
    else:
      check.append(False)
  if all(check) and len(check) >= 3:
    return True
  else:
    return False


def check_aligned(idf, corner, trueids, truecorners, CI):
  trueids = list(map(int, trueids))
  index = idf == trueids
  assert any(index), AssertionError(
      "can't find the detected id in configuration file!")

  check = []
  truecorner = np.array(truecorners)[index]
  for point1, point2 in zip(corner[0], truecorner[0]):
    if point2[0]+CI >= point1[0] >= np.maximum(point2[0]-CI, 0) and \
            point2[1]+CI >= point1[1] >= np.maximum(point2[1]-CI, 0):
      check.append(True)
    else:
      check.append(False)
  if all(check):
    return 255, True
  else:
    return 200, False


def get_align_color(ids, corners, trueids, truecorners, CI):
  aligns = []
  colors = []

  for idf, corner in zip(ids, corners):
    color, align = check_aligned(idf, corner, trueids, truecorners, CI)
    aligns.append(align)
    colors.append(color)
  return aligns, colors


def reformat(ids):
  new_ids = []
  for item in ids:
    new_ids.append(item[0])
  return np.array(new_ids)


def trim_corners(allCorners, allIds, maxBoards=85):
  '''
  only take "maxBoard" number of optimal allCorners
  '''
  counts = np.array([len(cs) for cs in allCorners])
  # detected more 6 corners
  sufficient_corners = np.greater_equal(counts, 6)
  sort = -counts + np.random.random(size=counts.shape) / 10
  subs = np.argsort(sort)[:maxBoards]
  allCorners = [allCorners[ix] for ix in subs if sufficient_corners[ix]]
  allIds = [allIds[ix] for ix in subs if sufficient_corners[ix]]
  return allCorners, allIds


def reformat_corners(allCorners, allIds):
  markerCounter = np.array([len(cs) for cs in allCorners])
  allCornersConcat = itertools.chain.from_iterable(allCorners)
  allIdsConcat = itertools.chain.from_iterable(allIds)

  allCornersConcat = np.array(list(allCornersConcat))
  allIdsConcat = np.array(list(allIdsConcat))

  return allCornersConcat, allIdsConcat, markerCounter


def quick_calibrate_charuco(allCorners, allIds, board, width, height):
  print("\ncalibrating...")
  tstart = time()

  cameraMat = np.eye(3)
  distCoeffs = np.zeros(14)
  dim = (width, height)
  calib_flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + \
      cv2.CALIB_FIX_PRINCIPAL_POINT
  calib_flags2 = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
      cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
  # all model included with 14 coeffifcent. about the flag please check:
  # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
  calib_flags3 = cv2.CALIB_RATIONAL_MODEL + \
      cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL

  error, cameraMat, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
      allCorners, allIds, board,
      dim, cameraMat, distCoeffs,
      flags=calib_flags3)

  tend = time()
  tdiff = tend - tstart
  print("\ncalibration took {} minutes and {:.1f} seconds".format(
      int(tdiff / 60), tdiff - int(tdiff / 60) * 60))

  out = dict()
  out['error'] = error
  out['camera_mat'] = cameraMat.tolist()
  out['dist_coeff'] = distCoeffs.tolist()
  out['width'] = width
  out['height'] = height

  return out


def quick_calibrate(someCorners, someIds, board, width, height):
  allCorners = []
  allIds = []

  allCorners.extend(someCorners)
  allIds.extend(someIds)

  allCorners, allIds = trim_corners(allCorners, allIds, maxBoards=100)
  allCornersConcat, allIdsConcat, markerCounter = reformat_corners(
      allCorners, allIds)

  expected_markers = get_expected_corners(board)+1

  print("\nfound {} markers, {} boards, {} complete boards".format(
      len(allCornersConcat), len(markerCounter),
      np.sum(markerCounter == expected_markers)))
  if len(allCornersConcat) < 10 or len(markerCounter) < 10:
    print("There are not enough markers to perform intrinsic calibration!")
    return {}
  else:
    calib_params = quick_calibrate_charuco(
        allCorners, allIds, board, width, height)
    return calib_params


class CharucoBoard:
  def __init__(self, x, y, marker_size=0.8,type=None):
    self.x = x
    self.y = y
    self.marker_size = marker_size
    self.default_dictionary = type  # default
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

  @property
  def default_dictionary(self):
    return self._default_dictionary

  @default_dictionary.setter
  def default_dictionary(self,type):
    if type is None:
      self._default_dictionary = cv2.aruco.DICT_4X4_50
    elif type == 'intrinsic':
      self._default_dictionary = cv2.aruco.DICT_5X5_50
    elif type == 'extrinsic' :
      self._default_dictionary = cv2.aruco.DICT_4X4_50
    else:
      raise ValueError('wrong type')

  def save_board(self, img_size=1000):
    file_name = 'charuco_board_shape_%dx%d_marker_size_%d_default_%d.png'%(self.x,self.y,self.marker_size,self.default_dictionary)
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
  def __init__(self, calib_type):
    self._get_type(calib_type)

    self.root_config_path = None

    self.allCorners = []
    self.allIds = []
    self.config = None

    self.charuco_board = CharucoBoard(x=self.x, y=self.y,type=self.type)
    self.board = self.charuco_board.board

    self.max_size = get_expected_corners(self.board)
    self.load_path = './config/'

  def _get_type(self, calib_type):
    if calib_type == 'extrinsic':
      self.type = calib_type
      self.x=6
      self.y=2
    elif calib_type == 'intrinsic':
      self.type = calib_type
      self.x=4
      self.y=5
    else:
      raise ValueError("type can only be intrinsic or extrinsic!")

  def reset(self):
    del self.allIds, self.allCorners, self.config
    self.allCorners = []
    self.allIds = []
    self.config = None

  @property
  def params(self):
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 700
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5
    return params

  @property
  def root_config_path(self):
    return self._root_config_path

  @root_config_path.setter
  def root_config_path(self,path):
    if path is not None:
      if os.path.exists(path):
        try:
          os.mkdir(os.path.join(path,'config')) # is it needed?
        except:
          pass
        self._root_config_path = os.path.join(path,'config')
      else:
        raise FileExistsError("root file folder doens't exist!")
    else:
      self._root_config_path=None

  # load configuration only for extrinsic calibration
  def load_ex_config(self, camera_serial_number):
    if not os.path.exists(self.load_path):
      os.mkdir(self.load_path)
      raise Warning("config directory doesn't exist. creating one...")

    items = os.listdir(self.load_path)
    for item in items:
      if camera_serial_number in item and self.type in item:
        path = os.path.join(self.load_path, 'config_%s_%d.toml'%(self.type, camera_serial_number))
        with open(path, 'r') as f:
          # there only should be only one calib file for each camera
          self.config = toml.load(f)
          try:
            self.config['ids'] = reformat(self.config['ids'])
            self.config['corners'] = reformat(self.config['corners'])
            markers = pd.DataFrame({'truecorners': list(self.config['corners'])},
                                   index=list(self.config['ids']))
            self.config['markers'] = markers
          except ValueError:
            print("Missing ids/corners/markers in the configuration file. Please check.")

  def save_config(self, camera_serial_number, width, height):
    save_path = os.path.join(self.root_config_path, 'config_%s_%d.toml'%(self.type,camera_serial_number))
    save_copy_path = self.load_path # overwrite

    if os.path.exists(save_path):
      return 'Configuration file already exists.'
    else:
      if self.type == "intrinsic":
        # time consuming
        param = quick_calibrate(self.allCorners,
                                    self.allIds,
                                    self.board,
                                    width,
                                    height)
        param['camera_serial_number'] = camera_serial_number
        param['date'] = time.strftime("%Y-%m-%d-_%H:%M:%S",time.localtime())
        if len(param) > 1:
          with open(save_path, 'w') as f:
            toml.dump(param, f)
          # save a copy to the configuration folder. Overwrite the previous one
          with open(save_copy_path,'w') as f:
            toml.dump(param, f)

          return "intrinsic calibration configuration saved!"
        else:
          return "intrinsic calibration configuration NOT saved due to lack of markers."
      else:
        if self.allIds is not None and not len(self.allIds) < self.max_size+1:
          param = {'corners': np.array(self.allCorners),
                   'ids': np.array(self.allIds), 'CI': 5,
                   'camera_serial_number': camera_serial_number,
                   'date': time.strftime("%Y-%m-%d-_%H:%M:%S",time.localtime())}
          with open(save_path, 'w') as f:
            toml.dump(param, f, encoder=toml.TomlNumpyEncoder())
          with open(save_copy_path,'w') as f:
            toml.dump(param, f, encoder=toml.TomlNumpyEncoder())

          return 'extrinsic calibration configuration saved!'
        else:
          return "failed to record all Ids! Can't save configuration. Please calibrate again."

  def in_calibrate(self,frame,data_count):
    # write something on the frame
    # text = 'Intrinsic calibration mode On'
    # cv2.putText(frame, text, (50, 50),
    #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 125), 2)

    # get corners and refine them in openCV for every 3 frames
    if data_count % CALIB_UPDATE_EACH == 0:
      corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, self.board.dictionary, parameters=self.params)
      detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
          cv2.aruco.refineDetectedMarkers(frame, self.board, corners, ids,
                                          rejectedImgPoints, parameters=self.params)
      # interpolate corners and draw corners
      if len(detectedCorners) > 0:
        rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
            detectedCorners, detectedIds, frame, self.board)
        if detectedCorners is not None and 2 <= len(
                detectedCorners) <= self.max_size:
          self.allCorners.append(detectedCorners)
          self.allIds.append(detectedIds)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

      return {'corners': detectedCorners, 'ids': detectedIds}
    else:
      return {'corners':[],'ids':[]}

  def ex_calibrate2(self,frame, data_count):
    allDetected = False
    if data_count % CALIB_UPDATE_EACH ==0:
      corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        frame, self.board.dictionary, parameters=self.params)
      detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        cv2.aruco.refineDetectedMarkers(frame, self.board, corners, ids,
                                        rejectedImgPoints, parameters=self.params)
      # interpolate corners and draw corners
      if len(detectedCorners) > 0:
        rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
          detectedCorners, detectedIds, frame, self.board)
        #if detectedCorners is not None and 2 <= len(
        #        detectedCorners) <= self.max_size:
        if detectedCorners is not None and len(detectedCorners)==self.max_size:
          self.allCorners=detectedCorners
          self.allIds=detectedIds
          allDetected=True

      return {'corners': detectedCorners, 'ids': detectedIds,'allDetected': allDetected}
    else:
      return {'corners': [], 'ids': [],'allDetected': allDetected}


def ex_calibrate(self,frame,data_count):
    # if there isn't configuration on the screen, save corners and ids
    allAligns = False  # TODO fix the logic here
    if self.config is None:
      # text = 'No configuration file found. Performing initial extrinsic calibration... '
      # cv2.putText(frame, text, (50, 50),
      #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      # calibrate every 3 frames
      if data_count % 3 == 0:  # TODO: move to constant at top of file
        # get parameters
        params = self.params

        # detect corners
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, self.board.dictionary, parameters=params)
        if ids is not None:
          # draw corners on the screen
          # cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

          if len(ids) >= len(self.allIds):
            self.allCorners = corners
            self.allIds = ids
    else:
      # text = 'Found configuration file for this camera. Calibrating...'
      # cv2.putText(frame, text, (50, 50),
      #             cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

      if True:  # process['calibrator'].decimator % 3 == 0:
        truecorners = self.config['corners']  # float numbers
        trueids = self.config['ids']  # int numbers
        CI = self.config['CI']  # int pixels
        markers = self.config['markers']

        # key step: detect markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, self.board.dictionary, parameters=self.params)

        # make sure there are ids and markers and the number of ids is no less than 3
        if check_ids(ids) and check_corners(corners):

          # check if aligned:
          aligns, colors = get_align_color(
            ids, corners, trueids, truecorners, CI)

          markers['aligns'] = pd.Series(
            aligns, index=list(map(str, reformat(ids))))
          markers['colors'] = pd.Series(
            colors, index=list(map(str, reformat(ids))))

          # any way to make it more concise?
          for tid, truecorner in zip(trueids, truecorners):
            real_color = int(markers['colors'][tid]) if pd.notna(
              markers['colors'][tid]) else 200
            point1 = tuple(np.array(truecorner[0], np.int))
            point2 = tuple(np.array(truecorner[1], np.int))
            point3 = tuple(np.array(truecorner[2], np.int))
            point4 = tuple(np.array(truecorner[3], np.int))
            cv2.line(frame, point1, point2, color=real_color, thickness=CI * 2)
            cv2.line(frame, point2, point3, color=real_color, thickness=CI * 2)
            cv2.line(frame, point3, point4, color=real_color, thickness=CI * 2)
            cv2.line(frame, point4, point1, color=real_color, thickness=CI * 2)
          # draw the detected markers on top of the true markers.
          # cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=225)

          allAligns = all(aligns)
          #   text = 'Enough corners aligned! Ready to go'
          #   cv2.putText(frame, text, (500, 1000),
          #               cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
          # else:
          #   text = "Missing ids or corners!"
          #   cv2.putText(frame, text, (500, 1000),
          #               cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    return {'corners': corners, 'ids': ids, 'allAligns': allAligns}
