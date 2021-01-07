import cv2
import numpy as np
import itertools
from time import time
import toml,os,re,ffmpeg

def get_expected_corners(board):
    board_size = board.getChessboardSize()
    return (board_size[0] - 1) * (board_size[1] - 1)


def check_ids(ids):
    if ids is None or ids==[]:
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
    if all(check) and len(check)>=3:
        return True
    else:
        return False


def check_aligned(idf, corner, trueids, truecorners, CI):
    trueids=list(map(int,trueids))
    index = idf == trueids
    assert any(index), AssertionError("can't find the detected id in configuration file!")

    check =[]
    truecorner = np.array(truecorners)[index]
    for point1, point2 in zip(corner[0], truecorner[0]):
        if point2[0]+CI>=point1[0]>=np.maximum(point2[0]-CI,0) and \
                point2[1]+CI>=point1[1]>=np.maximum(point2[1]-CI,0):
            check.append(True)
        else:
            check.append(False)
    if all(check):
        return 255, True
    else:
        return 200, False

def get_align_color(ids,corners,trueids,truecorners,CI):
    aligns=[]
    colors=[]

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
    calib_flags2 = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    # all model included with 14 coeffifcent. about the flag please check:
    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    calib_flags3 = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL

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
    allCornersConcat, allIdsConcat, markerCounter = reformat_corners(allCorners, allIds)

    expected_markers = get_expected_corners(board)

    print("\nfound {} markers, {} boards, {} complete boards".format(
        len(allCornersConcat), len(markerCounter),
        np.sum(markerCounter == expected_markers)))
    if len(allCornersConcat)<10 or len(markerCounter)<10:
        print("There are not enough markers to perform intrinsic calibration!")
        return {}
    else:
        calib_params = quick_calibrate_charuco(allCorners, allIds, board, width, height)
        return calib_params

# for all cameras
def undistort_videos(rootpath):
  raw_items = os.listdir(rootpath)
  config_path = os.path.join(rootpath,'config')
  items = os.listdir(config_path)
  if not os.path.exists(os.path.join(rootpath,'processed')):
    os.mkdir(os.path.join(rootpath,'processed'))
  processed_path = os.path.join(rootpath,'processed')

  intrinsics = None
  for item in items:
    if 'intrinsic' in item:
      serial_number = re.findall("\d+",item)[0]
      with open(os.path.join(config_path,item),'r') as f:
        intrinsics = toml.load(f)
      width = intrinsics['width']
      height = intrinsics['height']
      mtx = np.array(intrinsics['camera_mat'])
      dist = np.array(intrinsics['dist_coeff'])
      resolution = (width, height)
      newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, resolution, 1, resolution)

      movie = [a for a in raw_items if serial_number in a and '.MOV' in a]
      movie_path = os.path.join(rootpath,movie[0])

      video_writer = ffmpeg \
        .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{width}x{height}', framerate=30) \
        .output(os.path.join(processed_path,'undistorted_'+movie[0]), vcodec='libx265') \
        .overwrite_output() \
        .run_async(pipe_stdin=True)

      cap = cv2.VideoCapture(movie_path)
      ret =True
      while ret:
        ret,frame=cap.read()
        if frame is not None:
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
  if not os.path.exists(os.path.join(rootpath,'processed')):
    os.mkdir(os.path.join(rootpath,'processed'))

  intrinsics = [a for a in items if 'intrinsic' in a]

  for intrinsic in intrinsics:
    serial_number = re.findall("\d+", intrinsic)[0]
    # find top camera
    if serial_number == '17391304':
      with open(os.path.join(config_path,intrinsic),'r') as f:
        in_config = toml.load(f)

      mtx = np.array(in_config['camera_mat'])
      dist = np.array(in_config['dist_coeff'])

      with open(os.path.join(config_path,'config_extrinsic_%s.toml'%serial_number),'r') as f:
        ex_config = toml.load(f)
      corners=np.array(ex_config['corners'])
      shape = corners.shape
      new_corners=[]
      for i in range(shape[0]):
        new_corner = cv2.undistortPoints(corners[i][0],mtx,dist,P=mtx)
        new_corner=new_corner[:,0,:][np.newaxis, :]
        new_corners.append(new_corner)

      undistort_corners=np.array(new_corners).tolist()
      new_item = {'undistorted_corners': undistort_corners}

      with open(os.path.join(config_path, 'config_extrinsic_%s.toml'%serial_number), 'a') as f:
        toml.dump(new_item,f)

      print("saved undistorted video!")

if __name__=='__main__':
    rootpath=r'C:\Users\SchwartzLab\Desktop\2021-01-06_Testing(2)'
    #undistort_markers(rootpath)
    undistort_videos(rootpath)
