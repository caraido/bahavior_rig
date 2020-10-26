import cv2
import numpy as np
import itertools
from time import time


def get_expected_corners(board):
    board_size = board.getChessboardSize()
    return (board_size[0] - 1) * (board_size[1] - 1)


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

    calib_params = quick_calibrate_charuco(allCorners, allIds, board, width, height)
    return calib_params
