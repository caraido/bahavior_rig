import cv2
import numpy as np
import itertools
from time import time


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

    expected_markers = get_expected_corners(board)+1

    print("\nfound {} markers, {} boards, {} complete boards".format(
        len(allCornersConcat), len(markerCounter),
        np.sum(markerCounter == expected_markers)))
    if len(allCornersConcat)<10 or len(markerCounter)<10:
        print("There are not enough markers to perform intrinsic calibration!")
        return {}
    else:
        calib_params = quick_calibrate_charuco(allCorners, allIds, board, width, height)
        return calib_params

