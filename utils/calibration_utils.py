import cv2
import numpy as np
import itertools
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import toml
import ffmpeg
import re
import utils.calibration_3d_utils as ex_3d
from utils.calibration_3d_utils import get_expected_corners

CALIB_UPDATE_EACH = 1  # frame interval for calibration update
GLOBAL_CONFIG_PATH = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig'
TOP_CAM='17391304'


def transform_ids(ids):
  new_ids = []
  for item in ids:
    new_item = np.array(list(map(lambda x: int(x[0]), item)))
    new_item = new_item[:, np.newaxis]
    new_ids.append(new_item)
  return new_ids


def transform_corners(corners):
  new_corners=[]
  for item in corners:
    item = np.array(item, dtype=np.float32)
    new_corners.append(item)
  return new_corners

def transform_cornersWorld(corners):
  new_corners=[]
  for item in corners:
    item = np.array(item, dtype=np.float32)
    new_corners.append(item[:,0,:,:])
  return new_corners


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


def trim_corners(allCorners, allIds,allCornersWorld, maxBoards=85):
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
  allCornersWorld = [allCornersWorld[ix] for ix in subs if sufficient_corners[ix]]
  return allCorners, allIds,allCornersWorld


def reformat_corners(allCorners, allIds,allCornersWorld):
  markerCounter = np.array([len(cs) for cs in allCorners])
  allCornersConcat = itertools.chain.from_iterable(allCorners)
  allIdsConcat = itertools.chain.from_iterable(allIds)
  allCornersWorldConcat = itertools.chain.from_iterable(allCornersWorld)

  allCornersConcat = np.array(list(allCornersConcat))
  allIdsConcat = np.array(list(allIdsConcat))
  allCornersWorldConcat = np.array(list(allCornersWorldConcat))

  return allCornersConcat, allIdsConcat, allCornersWorldConcat,markerCounter


def quick_calibrate_charuco(allCorners, allIds, board, width, height):
  print("\ncalibrating...")
  tstart = time.time()

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
  calib_flags4 = cv2.CALIB_RATIONAL_MODEL

  error, cameraMat, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
      allCorners, allIds, board,
      dim, cameraMat, distCoeffs,
      flags=calib_flags4)

  tend = time.time()
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


def quick_calibrate_fisheye(someCorners,width,height):

    print("\ncalibrating...")
    tstart = time.time()

    # transform into image points list
    imgp = transform_corners(someCorners)

    # get object points
    obj = np.zeros((1, 4 * 5, 3), np.float32)
    obj[0, :, :2] = np.mgrid[0:4, 0:5].T.reshape(-1, 2)
    N_corners = len(imgp)
    objp = [obj] * N_corners

    # define calibration flag
    calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                   cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    dim = (width, height)

    # calibrate (fast)
    error, cameraMat, distCoeffs, rvecs, tvecs =cv2.fisheye.calibrate(objp,
                                                                      imgp,
                                                                      dim,
                                                                      None,None,flags=calib_flags)
    # count time
    tend = time.time()
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


def quick_calibrate(someCorners, someIds, cornersWorld,board, width, height):
  allCorners = []
  allIds = []

  allCorners.extend(someCorners)
  allIds.extend(someIds)

  allCorners, allIds,allCornersWorld = trim_corners(allCorners, allIds, cornersWorld, maxBoards=100)
  allCornersConcat, allIdsConcat, allcornersWorldConcat,markerCounter = reformat_corners(
      allCorners, allIds,allCornersWorld)

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


class Checkerboard:
  def __init__(self, squaresX, squaresY, squareLength):
    self.squaresX = squaresX
    self.squaresY = squaresY
    self.squareLength = squareLength

    objp = np.zeros((squaresX * squaresY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:squaresY, 0:squaresX].T.reshape(-1, 2)
    objp *= squareLength
    self.chessboardCorners = objp
    self.objPoints = objp

  def getChessboardSize(self):
    size = (self.squaresX, self.squaresY)
    return size

  def getGridSize(self):
    return self.getChessboardSize()

  def getSquareLength(self):
    return self.squareLength


class CharucoBoard:
  def __init__(self, x, y, marker_size=0.8,type=None):
    self.x = x
    self.y = y
    self.marker_size = marker_size
    self.default_dictionary = type
    self.cornersWorldWhole=[]
    self.dictionary = cv2.aruco.getPredefinedDictionary(
        self.default_dictionary)

    self._buildBoardCorrdination_3D()

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
  def default_dictionary(self, type):
    if type == 'intrinsic' or type == 'extrinsic':
      self._default_dictionary = cv2.aruco.DICT_5X5_50
    elif type == 'alignment':
      self._default_dictionary = cv2.aruco.DICT_4X4_50
    else:
      raise ValueError('wrong type')

  def _buildBoardCorrdination_3D(self):
    squareSize = 1 / 100
    for y in range(self.y - 1):
      for x in range(self.x - 1):
        self.cornersWorldWhole.append([[x * squareSize, y * squareSize, 0]])

    self.cornersWorldWhole = np.array(self.cornersWorldWhole, 'float32')

  def save_board(self, img_size=1000):
    file_name = 'charuco_board_shape_%dx%d_marker_size_%d_default_%d.png' % (
    self.x, self.y, self.marker_size, self.default_dictionary)
    img = self.board.draw((img_size, img_size))
    result = cv2.imwrite('./multimedia/board/' + file_name, img)
    if result:
      print('save board successfully! Name: ' + file_name)
    else:
      raise Exception('save board failed! Name: ' + file_name)

  def print_board(self):
    img = self.board.draw((1000, 1000))
    plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
    plt.axis("off")
    plt.show()


class Calib:
  def __init__(self, calib_type):
    self._get_type(calib_type)
    # TODO: how to set the local config path without assigning it?? Or should we
    self.root_config_path = GLOBAL_CONFIG_PATH

    self.allCorners = []
    self.allIds = []

    self.config = None
    self.temp_file=None

    self.charuco_board = CharucoBoard(x=self.x, y=self.y, type=self.type)
    self.board = self.charuco_board.board

    self.max_size = get_expected_corners(self.board)

  def _get_type(self, calib_type):
    if calib_type == 'alignment':
      self.type = calib_type
      self.x = 6
      self.y = 2
    elif calib_type == 'intrinsic':
      self.type = calib_type
      self.x = 4
      self.y = 5
    elif calib_type == 'extrinsic':
      self.type = calib_type
      self.x = 3
      self.y = 5
    else:
      raise ValueError("wrong type!")

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
  def root_config_path(self, path):
    if path is not None:
      if os.path.exists(path):
        try:
          os.makedirs(os.path.join(path, 'config'))  # is it needed?
        except:
          pass
        self._root_config_path = os.path.join(path, 'config')
      else:
        raise FileExistsError("root file folder doens't exist!")
    else:
      self._root_config_path = None

  # check before extrinsic calibration
  def load_in_config(self, camera_serial_number):
      path = os.path.join(self.root_config_path, 'config_%s_%s.toml' % ('intrinsic', camera_serial_number))
      with open(path, 'r') as f:
        self.config = toml.load(f)

  def load_temp_config(self,camera_serial_number):
    load_path = os.path.join( self.root_config_path, 'config_%s_%s_temp.toml' % (self.type, camera_serial_number))
    with open(load_path,'r') as f:
      temp_file= toml.load(f)
    temp_file['ids']=transform_ids(temp_file['ids'])
    temp_file['corners']=transform_corners(temp_file['corners'])
    return temp_file

  def save_temp_config(self,camera_serial_number, width,height):
    save_path = os.path.join( self.root_config_path, 'config_%s_%s_temp.toml' % (self.type, camera_serial_number))
    stuff={'corners':self.allCorners,
          'ids':self.allIds,
          'camera_serial_number': camera_serial_number,
          'width':width,
          'height':height,
          'type':self.type,
          'date':time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}
    with open(save_path,'w') as f:
      toml.dump(stuff,f,encoder=toml.TomlNumpyEncoder())
    return "temp calibration file saved!"


  # used in post processing
  def save_processed_config(self,temp_path):

    if os.path.exists(temp_path):
      stuff = toml.load(temp_path)
      save_path = os.path.join(self.root_config_path,
                               'config_%s_%s.toml' % (self.type, stuff['camera_serial_number']))

      if self.type=="intrinsic":
        if str(stuff['camera_serial_number'])==TOP_CAM:
          # regular intrinsic calibration for top camera
          param=quick_calibrate(stuff['corners'],
                                stuff['ids'],
                                self.board,
                                stuff['width'],
                                stuff['height'])
        else:
          # fisheye calibration for side cameras
          param = quick_calibrate_fisheye(stuff['corners'],
                                          stuff['width'],
                                          stuff['height'])

        # add camera serial number
        param['camera_serial_number']=stuff['camera_serial_number']
        param['date'] = stuff['date']
        if len(param) > 2:
          with open(save_path, 'w') as f:
            toml.dump(param, f)

      elif self.type == 'alignment':
        param = {'corners': np.array(stuff['corners']),
                 'ids': np.array(stuff['ids']),
                 'camera_serial_number': stuff['camera_serial_number'],
                 'date': stuff['date']}
        with open(save_path, 'w') as f:
          toml.dump(param, f, encoder=toml.TomlNumpyEncoder())

      elif self.type == 'extrinsic':
        # TODO
        pass

  def in_calibrate(self, frame, data_count,serial_number):
    # get corners and refine them in openCV for every CALIB_UPDATE_EACH frames
    if data_count % CALIB_UPDATE_EACH == 0:
      if str(serial_number) == TOP_CAM:
        ret, corners=cv2.findChessboardCorners(frame, (self.x,self.y),None)
        if ret:
          SUB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.1)
          corners = cv2.cornerSubPix(frame, corners, (3, 3), (-1, -1), SUB_CRITERIA)
          self.allCorners.append(corners)
        return {'corners':corners,'ret':ret}
      else:
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            frame, self.board.dictionary, parameters=self.params)
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
            cv2.aruco.refineDetectedMarkers(frame, self.board, corners, ids,
                                            rejectedImgPoints, parameters=self.params)

        # interpolate corners
        if len(detectedCorners) > 0:
          recoveredIdxsst, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
              detectedCorners, detectedIds, frame, self.board)
          if detectedCorners is not None and 2 <= len(
                  detectedCorners) <= self.max_size:
            self.allCorners.append(detectedCorners)
            self.allIds.append(detectedIds)

          return {'corners': corners, 'ids': ids}
        else:
          return {'corners': [], 'ids': []}

  def al_calibrate(self, frame, data_count):
    allDetected = False
    if data_count % CALIB_UPDATE_EACH == 0:
      corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
          frame, self.board.dictionary, parameters=self.params)

      if corners is not None and len(corners) == self.max_size+1:
        # update the latest markers
        self.allCorners = corners
        self.allIds = ids
        allDetected = True

      return {'corners': corners, 'ids': ids, 'allDetected': allDetected}
    else:
      return {'corners': [], 'ids': [], 'allDetected': allDetected}

  def ex_calibrate(self,frame, data_count):
    if self.config is not None:
      if data_count % CALIB_UPDATE_EACH == 0:
        # detect corners
        detectedCorners, detectedIds, corners, ids=ex_3d.detect_aruco_2(frame,
                                                                        intrinsics=self.config,
                                                                        params=self.params,
                                                                        board=self.board)
        self.allCorners.append(detectedCorners)
        self.allIds.append(detectedIds)
        return {'corners': corners, 'ids': ids}
      else:
        return {'corners': [], 'ids': []}
    else:
      return {'corners': [], 'ids': None}
'''
def ex_calibrate(self, frame, data_count):
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
'''

# for all cameras

def undistort_videos(rootpath):
  raw_items = os.listdir(rootpath)
  config_path = os.path.join(rootpath, 'config')
  items = os.listdir(config_path)
  processed_path = os.path.join(rootpath, 'processed')
  if not os.path.exists(os.path.join(rootpath,'processed')):
    os.mkdir(os.path.join(rootpath,'processed'))
  processed_path = os.path.join(rootpath,'processed')

  intrinsics = None
  for item in items:
    if 'intrinsic' in item:
      serial_number = re.findall("\d+", item)[0]
      with open(os.path.join(config_path, item), 'r') as f:
        intrinsics = toml.load(f)
      width = intrinsics['width']
      height = intrinsics['height']
      mtx = np.array(intrinsics['camera_mat'])
      dist = np.array(intrinsics['dist_coeff'])
      resolution = (width, height)
      newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
          mtx, dist, resolution, 1, resolution)

      movie = [a for a in raw_items if serial_number in a and '.MOV' in a]
      movie_path = os.path.join(rootpath, movie[0])

      # TODO: frame rate needs to coordinate with raw video
      video_writer = ffmpeg \
          .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{width}x{height}', framerate=30) \
          .output(os.path.join(processed_path, 'undistorted_'+movie[0]), vcodec='libx265') \
          .overwrite_output() \
          .run_async(pipe_stdin=True)
      # TODO: switch to quiet mode, see Camera.open_file

      cap = cv2.VideoCapture(movie_path)
      ret = True
      while ret:
        ret, frame = cap.read()
        if ret:
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          dst = cv2.undistort(gray, mtx, dist, None, newcameramtx)
          video_writer.stdin.write(dst.tobytes())
      video_writer.stdin.close()
      video_writer.wait()
      del video_writer

  return intrinsics

# only for top camera

def undistort_markers(rootpath):
  config_path = os.path.join(rootpath, 'config')
  items = os.listdir(config_path)
  if not os.path.exists(os.path.join(rootpath, 'processed')):
    os.mkdir(os.path.join(rootpath, 'processed'))

  intrinsics = [a for a in items if 'intrinsic' in a]

  for intrinsic in intrinsics:
    serial_number = re.findall("\d+", intrinsic)[0]
    # find top camera
    if serial_number == TOP_CAM:
      with open(os.path.join(config_path, intrinsic), 'r') as f:
        in_config = toml.load(f)

      mtx = np.array(in_config['camera_mat'])
      dist = np.array(in_config['dist_coeff'])

      with open(os.path.join(config_path, 'config_extrinsic_%s.toml' % serial_number), 'r') as f:
        ex_config = toml.load(f)
      corners = np.array(ex_config['corners'])
      shape = corners.shape
      new_corners = []
      for i in range(shape[0]):
        new_corner = cv2.undistortPoints(corners[i], mtx, dist, P=mtx)
        new_corners.append(new_corner)
      undistort_corners = np.swapaxes(np.array(new_corners), 1, 2).tolist()
      new_item = {'undistorted_corners': undistort_corners}

      with open(os.path.join(config_path, 'config_extrinsic_%s.toml' % serial_number), 'a') as f:
        toml.dump(new_item, f)
