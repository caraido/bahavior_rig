import cv2
import numpy as np
import itertools
from time import time


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
  def __init__(self, x, y, marker_size=0.8):
    self.x = x
    self.y = y
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
  def __init__(self, calib_type):
    self._get_type(calib_type)

    self.allCorners = []
    self.allIds = []
    self.decimator = 0
    self.config = None

    self.board = CharucoBoard(x=6, y=2).board

    self.max_size = cau.get_expected_corners(self.board)
    self.save_path = './config/config_'+self.type+'_'
    self.load_path = './config/'

  def _get_type(self, calib_type):
    if calib_type == 'intrinsic' or calib_type == 'extrinsic':
      self.type = calib_type
    else:
      raise ValueError("type can only be intrinsic or extrinsic!")

  def reset(self):
    del self.allIds, self.allCorners, self.decimator, self.config
    self.allCorners = []
    self.allIds = []
    self.decimator = 0
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

  def load_config(self, camera_serial_number):
    if not os.path.exists(self.load_path):
      os.mkdir(self.load_path)
      raise Warning("config directory doesn't exist. creating one...")

    items = os.listdir(self.load_path)
    for item in items:
      if camera_serial_number in item and self.type in item:
        path = self.load_path+'config_' + self.type + \
            '_' + camera_serial_number + '.toml'
        with open(path, 'r') as f:
          # there only should be only one calib file for each camera
          self.config = toml.load(f)
          try:
            self.config['ids'] = cau.reformat(self.config['ids'])
            self.config['corners'] = cau.reformat(self.config['corners'])
            markers = pd.DataFrame({'truecorners': list(self.config['corners'])},
                                   index=list(self.config['ids']))
            self.config['markers'] = markers
          except ValueError:
            print("there's nothing in the configuration file called ids! Please check.")

  def save_config(self, camera_serial_number, width, height):
    save_path = self.save_path + camera_serial_number + '.toml'
    if os.path.exists(save_path):
      print('Configuration file already exists.')
    else:
      if self.type == "intrinsic":
        param = cau.quick_calibrate(self.allCorners,
                                    self.allIds,
                                    self.board,
                                    width,
                                    height)
        param['camera_serial_number'] = camera_serial_number
        if len(param) > 1:
          with open(save_path, 'w') as f:
            toml.dump(param, f)
          print('intrinsic calibration configuration saved!')
        else:
          print("intrinsic calibration configuration NOT saved due to lack of markers.")
      else:
        if self.allIds is not None and not len(self.allIds) < self.max_size+1:
          param = {'corners': np.array(self.allCorners),
                   'ids': np.array(self.allIds), 'CI': 5,
                   'camera_serial_number': camera_serial_number}
          with open(save_path, 'w') as f:
            toml.dump(param, f, encoder=toml.TomlNumpyEncoder())
            print('extrinsic calibration configuration saved!')
        else:
          # TODO: should be a pop up window/show up on the screen
          raise Exception(
              "failed to record all Ids! can't save configuration. Please calibrate again.")
