import os
import cv2
import toml
import queue
import numpy as np

from time import time
from tqdm import trange
from collections import defaultdict, Counter

from scipy import optimize
from scipy.cluster.vq import whiten
from scipy.sparse import lil_matrix
from scipy.cluster.hierarchy import linkage, fcluster

from utils.calibration_utils import get_expected_corners

from utils.triangulate import triangulate_simple, triangulate_points, \
    reprojection_error_und


def fill_points(corners, ids, board):
    num_corners = get_expected_corners(board)

    out = np.zeros((num_corners, 2))
    out.fill(np.nan)

    if ids is None or corners is None:
        return out

    corners = corners.reshape(-1, 2)
    ids = ids.flatten()

    for ix, corner in zip(ids, corners):
        out[ix] = corner

    return out


def reconstruct_checkerboard(row, camera_mats, camera_mats_dist):
    vid_indices = sorted(row.keys())
    mats = [camera_mats[name] for name in vid_indices]

    num_points = row[vid_indices[0]].shape[0]
    p3ds = []
    errors = []
    for i in range(num_points):
        pts = [row[name][i] for name in vid_indices]
        pts = np.array(pts).reshape(-1, 2)
        p3d = triangulate_simple(pts, mats)
        error = reprojection_error_und(p3d, pts, mats, camera_mats_dist)
        p3ds.append(p3d)
        errors.append(error)
    p3ds = np.array(p3ds)
    errors = np.array(errors)
    
    return p3ds, errors

def detect_aruco_2(gray,intrinsics, board,params):
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, board.dictionary, parameters=params)
    cv2.aruco.drawDetectedMarkers(gray, corners, ids, borderColor=225)

    if intrinsics is None:
        INTRINSICS_K = INTRINSICS_D = None
    else:
        INTRINSICS_K = np.array(intrinsics['camera_mat'])
        INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    if ids is None:
        return [], []
    elif len(ids) < 2:
        return corners, ids

    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        cv2.aruco.refineDetectedMarkers(gray, board, corners, ids,
                                        rejectedImgPoints,
                                        INTRINSICS_K, INTRINSICS_D,
                                        parameters=params)

    if len(detectedCorners) > 0:
        ret, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
            detectedCorners, detectedIds, gray, board)

        if detectedIds is None:
            detectedCorners = detectedIds = []

    return detectedCorners, detectedIds

def detect_aruco(gray, intrinsics, board):
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)

    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 600
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, board.dictionary, parameters=params)
    cv2.aruco.drawDetectedMarkers(gray, corners, ids, borderColor=225)

    if intrinsics is None:
        INTRINSICS_K = INTRINSICS_D = None
    else:
        INTRINSICS_K = np.array(intrinsics['camera_mat'])
        INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    if ids is None:
        return [], []
    elif len(ids) < 2:
        return corners, ids

    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        cv2.aruco.refineDetectedMarkers(gray, board, corners, ids,
                                    rejectedImgPoints,
                                    INTRINSICS_K, INTRINSICS_D,
                                    parameters=params)

    if len(detectedCorners) > 0:
        ret, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
            detectedCorners, detectedIds, gray, board)

        if detectedIds is None:
            detectedCorners = detectedIds = []

    return detectedCorners, detectedIds

def estimate_pose_aruco_2(detectedCorners, detectedIds, intrinsics,board):
    if len(detectedIds) < 3:
        return False, None

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            detectedCorners, detectedIds, board, INTRINSICS_K, INTRINSICS_D,
            rvec=np.array([]), tvec=np.array([]), useExtrinsicGuess=False)

    if not ret or rvec is None or tvec is None:
        return False, None

    return True, (detectedCorners, detectedIds, rvec, tvec)

def estimate_pose_aruco(gray, intrinsics, board):

    detectedCorners, detectedIds = detect_aruco(gray, intrinsics, board)
    if len(detectedIds) < 3:
        return False, None

    INTRINSICS_K = np.array(intrinsics['camera_mat'])
    INTRINSICS_D = np.array(intrinsics['dist_coeff'])

    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            detectedCorners, detectedIds, board, INTRINSICS_K, INTRINSICS_D,
            rvec=np.array([]), tvec=np.array([]), useExtrinsicGuess=False)

    if not ret or rvec is None or tvec is None:
        return False, None

    return True, (detectedCorners, detectedIds, rvec, tvec)


def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    
    return out


def get_rtvec(M):
    rvec = cv2.Rodrigues(M[:3, :3])[0].flatten()
    tvec = M[:3, 3].flatten()
    
    return rvec, tvec


def get_most_common(vals):
    Z = linkage(whiten(vals), 'ward')
    n_clust = max(len(vals)/10, 3)
    clusts = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusts[clusts >= 0])
    most = cc.most_common(n=1)
    top = most[0][0]
    good = clusts == top
    
    return good


def select_matrices(Ms):
    Ms = np.array(Ms)
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in Ms]
    tvecs = np.array([M[:3, 3] for M in Ms])
    best = get_most_common(np.hstack([rvecs, tvecs]))
    Ms_best = Ms[best]
    
    return Ms_best


def mean_transform(M_list):
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in M_list]
    tvecs = [M[:3, 3] for M in M_list]

    rvec = np.mean(rvecs, axis=0)
    tvec = np.mean(tvecs, axis=0)

    return make_M(rvec, tvec)


def mean_transform_robust(M_list, approx=None, error=0.3):
    if approx is None:
        M_list_robust = M_list
    else:
        M_list_robust = []
        for M in M_list:
            rot_error = (M - approx)[:3,:3]
            m = np.max(np.abs(rot_error))
            if m < error:
                M_list_robust.append(M)
                
    return mean_transform(M_list_robust)

def get_matrices_2(cam_indices,
                   ids_list,
                   corners_list,
                   intrinsics_list,
                   board, skip=40):
    minlen = np.inf

    for cam_idx, ids in zip(cam_indices, ids_list):
        length = len(ids)
        minlen = min(length, minlen)

    go = skip
    all_Ms = []
    all_points = []

    for index in trange(minlen, ncols=70):
        M_dict = dict()
        point_dict = dict()

        for cam_idx in cam_indices:
            success, result = estimate_pose_aruco_2(corners_list[cam_idx][index],
                                                    ids_list[cam_idx][index],
                                                    intrinsics_list[cam_idx],
                                                    board)
            if not success:
                continue
            corners, ids, rvec, tvec = result
            intrinsics = intrinsics_list[cam_idx]

            M_dict[cam_idx] = make_M(rvec, tvec)

            points = fill_points(corners, ids, board)
            points_flat = points.reshape(-1, 1, 2)
            points_new = cv2.undistortPoints(points_flat,
                                             np.array(intrinsics['camera_mat']),
                                             np.array(intrinsics['dist_coeff']))

            point_dict[cam_idx] = points_new.reshape(points.shape)


        if len(M_dict) >= 2:
            go = skip
            all_Ms.append(M_dict)
            all_points.append(point_dict)

        go = max(0, go-1)

    return all_Ms, all_points

def get_matrices(vid_indices, videos, intrinsics_dict, board, skip=40):
    minlen = np.inf
    caps = dict()
    for vid_idx, vid in zip(vid_indices, videos):
        cap = cv2.VideoCapture(vid)
        caps[vid_idx] = cap
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        minlen = min(length, minlen)

    go = skip
    all_Ms = []
    all_points = []

    for framenum in trange(minlen, ncols=70):
        M_dict = dict()
        point_dict = dict()

        for vid_idx in vid_indices:
            cap = caps[vid_idx]
            ret, frame = cap.read()

            if framenum % skip != 0 and go <= 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intrinsics = intrinsics_dict[vid_idx]
            success, result = estimate_pose_aruco(gray, intrinsics, board)
            if not success:
                continue

            corners, ids, rvec, tvec = result
            M_dict[vid_idx] = make_M(rvec, tvec)

            points = fill_points(corners, ids, board)
            points_flat = points.reshape(-1, 1, 2)
            points_new = cv2.undistortPoints(points_flat,
                                             np.array(intrinsics['camera_mat']),
                                             np.array(intrinsics['dist_coeff']))

            point_dict[vid_idx] = points_new.reshape(points.shape)


        if len(M_dict) >= 2:
            go = skip
            all_Ms.append(M_dict)
            all_points.append(point_dict)

        go = max(0, go-1)

    for vid_idx, cap in caps.items():
        cap.release()

    return all_Ms, all_points


def get_transform(matrix_list, left, right):
    L = []
    for d in matrix_list:
        if left in d and right in d:
            M = np.matmul(d[left], np.linalg.inv(d[right]))
            L.append(M)
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    # M_mean = mean_transform_robust(L, M_mean, error=0.5)
    # M_mean = mean_transform_robust(L, M_mean, error=0.2)
    M_mean = mean_transform_robust(L, M_mean, error=0.1)
    
    return M_mean


def get_all_matrix_pairs(matrix_list, vid_indices):
    out = dict()

    for left in vid_indices:
        for right in vid_indices:
            if left == right:
                continue

            M = get_transform(matrix_list, left, right)
            out[(left, right)] = M

    return out


def get_calibration_graph(imgpoints, vid_indices):
    n_cams = len(vid_indices)
    connections = defaultdict(int)

    for p in imgpoints:
        keys = sorted(p.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a = keys[i]
                b = keys[j]
                connections[(a,b)] += 1
                connections[(b,a)] += 1

    components = dict(zip(vid_indices, range(n_cams)))
    edges = set(connections.items())

    print(sorted(edges))

    graph = defaultdict(list)

    for edgenum in range(n_cams-1):
        if len(edges) == 0:
            return None

        (a, b), weight = max(edges, key=lambda x: x[1])
        graph[a].append(b)
        graph[b].append(a)

        match = components[a]
        replace = components[b]
        for k, v in components.items():
            if match == v:
                components[k] = replace

        for e in edges.copy():
            (a,b), w = e
            if components[a] == components[b]:
                edges.remove(e)
                
    return graph

def find_calibration_pairs(graph, source=0):
    pairs = []
    explored = set()

    q = queue.deque()
    q.append(source)

    while len(q) > 0:
        item = q.pop()
        explored.add(item)

        for new in graph[item]:
            if new not in explored:
                q.append(new)
                pairs.append((item, new))
                
    return pairs


def compute_camera_matrices(matrix_list, pairs, source=0):
    extrinsics = dict()
    extrinsics[source] = np.identity(4)
    for (a,b) in pairs:
        ext = get_transform(matrix_list, b, a)
        extrinsics[b] = np.matmul(ext, extrinsics[a])
    return extrinsics


def estimate_calibration_errors(point_list, intrinsics_dict, extrinsics):
    errors = []
    for points in point_list:
        vid_indices = points.keys()
        cam_mats = np.array([extrinsics[vid_idx] for vid_idx in vid_indices])
        cam_mats_dist = np.array([intrinsics_dict[vid_idx]['camera_mat'] for vid_idx in vid_indices])
        pts = np.array([points[vid_idx] for vid_idx in vid_indices])
        for i in range(pts.shape[1]):
            if np.sum(~np.isnan(pts[:, i, 0])) < 2:
                continue
    
            good = ~np.isnan(pts[:, i, 0])
            p3d = triangulate_simple(pts[good, i], cam_mats[good])
            error = reprojection_error_und(p3d, pts[good, i], cam_mats[good], cam_mats_dist[good])
            errors.append(error)

    return np.array(errors)


def mats_to_params(mats):
    params = np.zeros(len(mats)*6)
    for i, M in enumerate(mats):
        rvec, tvec = get_rtvec(M)
        s = i*6
        params[s:s+3] = rvec
        params[s+3:s+6] = tvec
    return params


def params_to_mats(params):
    # cam_mats = [np.identity(4)]
    cam_mats = []
    n_cams = len(params) // 6
    for i in range(n_cams):
        s = i*6
        MX = make_M(params[s:s+3], params[s+3:s+6])
        cam_mats.append(MX)
    cam_mats = np.array(cam_mats)
    return cam_mats


def setup_bundle_problem(point_list, extrinsics, cam_align):
    out = []
    vid_indices = sorted(extrinsics.keys())

    points = point_list[0]
    v = list(points.values())[0]
    template = v*np.nan

    for points in point_list:
        pts = []
        for vid_idx in vid_indices:
            if vid_idx in points:
                p = points[vid_idx]
            else:
                p = template
            pts.append(p)
        pts = np.array(pts)
        for i in range(pts.shape[1]):
            if np.sum(~np.isnan(pts[:, i, 0])) < 2:
                continue
            out.append(pts[:, i])

    out = np.array(out)
    return out


def evaluate_errors(the_points, cam_mats, p3ds, good=None):
    points_pred = np.zeros(the_points.shape)
    for i in range(cam_mats.shape[0]):
        pp = p3ds.dot(cam_mats[i].T)
        points_pred[:, i, :] = pp[:, :2] / pp[:, 2, None]
    errors = np.clip(points_pred - the_points, -1e6, 1e6)
    if good is None:
        return errors
    else:
        return errors[good]


def make_error_fun(the_points, n_samples=None, sum=False):
    the_points_sampled = the_points
    if n_samples is not None and n_samples < the_points.shape[0]:
        samples = np.random.choice(the_points.shape[0], n_samples, replace=False)
        the_points_sampled = the_points[samples]

    n_cameras = the_points_sampled.shape[1]
    good = ~np.isnan(the_points_sampled)

    def error_fun(params):
        sub = n_cameras * 6
        cam_mats = params_to_mats(params[:sub])
        p3ds = params[sub:].reshape(-1, 3)
        p3ds = np.hstack([p3ds, np.ones((len(p3ds), 1))])

        errors = evaluate_errors(the_points_sampled, cam_mats, p3ds, good)
        out = errors
        if sum:
            return np.sum(out)
        else:
            return out
        
    return error_fun, the_points_sampled


def build_jac_sparsity(the_points):
    point_indices = np.zeros(the_points.shape, dtype='int32')
    cam_indices = np.zeros(the_points.shape, dtype='int32')

    for i in range(the_points.shape[0]):
        point_indices[i] = i

    for j in range(the_points.shape[1]):
        cam_indices[:, j] = j

    good = ~np.isnan(the_points)

    n_points = the_points.shape[0]
    n_cams = the_points.shape[1]
    n_params = n_cams*6 + n_points*3
    n_errors = np.sum(good)

    A_sparse = lil_matrix((n_errors, n_params), dtype='int16')
    
    cam_indices_good = cam_indices[good]
    point_indices_good = point_indices[good]

    ix = np.arange(n_errors)

    for i in range(6):
        A_sparse[ix, cam_indices_good*6 + i] = 1

    for i in range(3):
        A_sparse[ix, n_cams*6 + point_indices_good*3 + i] = 1

    return A_sparse


def bundle_adjust(all_points, vid_indices, cam_mats, loss='linear'):
    """performs bundle adjustment to improve estimates of camera matrices

    Parameters
    ----------
    all_points: 2d points, array of shape (n_points, n_cams, 2)
       undistorted 2d points
    vid_indices: array like
    cam_mats: array of shape (n_cams, 4)
    """

    n_cameras = len(cam_mats)

    error_fun, points_sampled = make_error_fun(all_points, n_samples=int(300e3))
    p3ds_sampled, _ = triangulate_points(points_sampled, cam_mats)

    params_cams = mats_to_params(cam_mats)
    params_points = p3ds_sampled[:, :3].reshape(-1)
    params_full = np.hstack([params_cams, params_points])

    jac_sparse = build_jac_sparsity(points_sampled)

    f_scale = np.std(points_sampled[~np.isnan(points_sampled)])*1e-2

    opt = optimize.least_squares(error_fun, params_full,
                                 jac_sparsity=jac_sparse, f_scale=f_scale,
                                 x_scale='jac', loss=loss, ftol=1e-6,
                                 method='trf', tr_solver='lsmr', verbose=2,
                                 max_nfev=1000)
    best_params = opt.x
    mats_new = params_to_mats(best_params[:n_cameras*6])

    extrinsics_new = dict(zip(list(map(str,vid_indices)), mats_new))

    return extrinsics_new


def get_extrinsics(vid_indices, videos, intrinsics_dict, cam_align, board, skip=40):
    matrix_list, point_list = get_matrices(vid_indices, videos, intrinsics_dict, board, skip=skip)
    
    # pairs = get_all_matrix_pairs(matrix_list, sorted(vid_indices))
    graph = get_calibration_graph(matrix_list, vid_indices)
    pairs = find_calibration_pairs(graph, source=cam_align)
    extrinsics = compute_camera_matrices(matrix_list, pairs, source=cam_align)

    errors = estimate_calibration_errors(point_list, intrinsics_dict, extrinsics)
    print('\nBefore bundle adjustment, mean reprojection error is {:.5f}'.format(np.mean(errors)))

    all_points = setup_bundle_problem(point_list, extrinsics, cam_align)

    cam_mats = np.array([extrinsics[vid_idx] for vid_idx in vid_indices])

    t1 = time()
    extrinsics_new = bundle_adjust(all_points, vid_indices, cam_mats)
    t2 = time()
    print('\nbundle adjustment took {:.1f} seconds'.format(t2 - t1))

    errors = estimate_calibration_errors(point_list, intrinsics_dict, extrinsics_new)
    print('\nAfter bundle adjustment, mean reprojection error is {:.5f}'.format(np.mean(errors)))

    return extrinsics_new, np.mean(errors)


def get_extrinsics_2(cam_indices,
                   ids_list,
                   corners_list,
                   intrinsics_list,
                   cam_align,
                   board, skip=40):

    matrix_list, point_list = get_matrices_2(cam_indices,
                                               ids_list,
                                               corners_list,
                                               intrinsics_list,
                                               board, skip=skip)

    # pairs = get_all_matrix_pairs(matrix_list, sorted(vid_indices))
    graph = get_calibration_graph(matrix_list, cam_indices)
    pairs = find_calibration_pairs(graph, source=cam_align)
    extrinsics = compute_camera_matrices(matrix_list, pairs, source=cam_align)

    errors = estimate_calibration_errors(point_list, intrinsics_list, extrinsics)
    print('\nBefore bundle adjustment, mean reprojection error is {:.5f}'.format(np.mean(errors)))

    all_points = setup_bundle_problem(point_list, extrinsics, cam_align)

    cam_mats = np.array([extrinsics[cam_idx] for cam_idx in cam_indices])

    t1 = time()
    extrinsics_new = bundle_adjust(all_points, cam_indices, cam_mats)
    t2 = time()
    print('\nbundle adjustment took {:.1f} seconds'.format(t2 - t1))

    errors = estimate_calibration_errors(point_list, intrinsics_list, extrinsics_new)
    print('\nAfter bundle adjustment, mean reprojection error is {:.5f}'.format(np.mean(errors)))

    return extrinsics_new, np.mean(errors)


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


def calibrate(rootpath):
    items=os.listdir(rootpath)
    videos=[]
    intrinsics_path=[]
    cam_align = 1
    vid_indices=[0,1]
    board_obj= CharucoBoard(4,5,type='intrinsic')
    board=board_obj.board
    for item in items:
        if '.MOV' in item:
            videos.append(os.path.join(rootpath,item))
        if 'intrinsic' in item:
            intrinsics_path.append(os.path.join(rootpath,item))

    videos=sorted(videos)
    intrinsics_path=sorted(intrinsics_path)
    loaded=[toml.load(path) for path in intrinsics_path]

    extrinsics, error = get_extrinsics(vid_indices, videos, loaded, cam_align, board)

    return extrinsics,error

if __name__ =='__main__':
    rootpath='/Users/tianhaolei/Downloads/pseudo_calib'
    extrinsics,error = calibrate(rootpath)
    print(extrinsics)
    print(error)