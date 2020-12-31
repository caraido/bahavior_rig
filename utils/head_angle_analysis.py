import numpy as np
from pandas import DataFrame


def norm_vector(vector):
  ''' Return the normalized version of the vector s.t. its magnitude is 1.0
  '''
  return np.divide(vector, np.sqrt(np.sum(vector**2, axis=1)[:, np.newaxis]))

def cutoff(pose,cutoff=0.9):
    pass

def angle_2pi(va):
  ''' Calculate the signed angle from +x axis to vector a. The result resides on [-pi, pi]
  '''
  # calculate theta on (0,pi)
  nva = norm_vector(va)
  nvb = np.array([1, 0])[np.newaxis, :]
  # nvb = norm_vector(vb)

  theta = np.arccos(np.clip(np.sum(nva * nvb, axis=1), -1.0, +1.0))

  # calculate sin(theta)
  a_cw_90 = np.stack((nva[:, 1], -nva[:, 0]), axis=1)
  sin_ab_neg = np.clip(np.sum(a_cw_90 * nvb, axis=1), -1.0, +1.0) < 0

  # if sin(theta)<0 then theta is negative on (0,2pi)
  theta[sin_ab_neg] = - theta[sin_ab_neg]

  return theta


def arc_2pi(xy, center, radius):
  # for a circle with center (h,k), radius r:
  # x = h + r*cos(theta) for all theta
  # y = k + r*sin(theta) for all theta
  # print('xy: ', xy)

  theta = np.arccos(np.clip(
      (xy[:, 0][:, np.newaxis] - center[:, 0][:, np.newaxis])/radius, -1.0, +1.0))

  sintheta_neg = ((xy[:, 1][:, np.newaxis] - center[:, 1]
                   [:, np.newaxis])/radius) < 0
  theta[sintheta_neg] = -theta[sintheta_neg]

  return theta


def intersect_ray_circle(head_center, arena_center, eye_head, head_arena, r0, r1):
  # the ray- circle intersection problem defines a quadratic equation where the roots are the intersection points
  # we will require a positive root (i.e. the vector points in the direction of the wall)
  # we will test the inner wall first
  # if there are multiple positive real roots we will take the shorter one
  # if there is no real root with inner wall we will take the positive root with the outer wall (guaranteed 1)

  # a = [(left_eye - head_center)^2] sum xy
  # b = [2 * (left_eye - head_center)*(head_center - arena_center)] sum xy
  # c = [left_eye - arena_center)^2] sum xy  - radius^2

  # a === 1
  b = np.sum(2 * eye_head * head_arena, axis=1)
  c = np.sum((head_arena)**2, axis=1)

  # solve the quadratic equation
  hat = b*b - 4*(c-r0*r0)
  hn = hat < 0
  hat[hn] = np.nan  # we didn't intersect the inner circle at all
  hat = np.sqrt(hat)

  t0 = (-b+hat)/2
  t0n = t0 < 0
  t0[t0n] = np.nan  # these are pointing in the wrong direction

  t1 = (-b-hat)/2
  t1n = t1 < 0
  t1[t1n] = np.nan  # also pointing in wrong direction

  # case: the ray intersects the inner circle at a single point (only if inside)
  t0nt1 = np.logical_and(t0n, ~t1n)

  # case: the ray doesn't intersect the inner circle
  t0nt1n = np.logical_or(np.logical_and(t0n, t1n), hn)

  t0[t0nt1] = t1[t0nt1]

  t1i = t1 < t0
  t0[t1i] = t1[t1i]  # take the smallest distance

  # the 2d coordinates of the intersection, or nan
  pos0 = eye_head*t0[:, np.newaxis] + head_center

  # get the outer circle intersection using the same approach
  hat = b[t0nt1n]*b[t0nt1n] - 4*(c[t0nt1n]-r1*r1)
  # this should always be positive since the mouse is inside the outer circle at all times
  # but in case the mouse is mistakenly labelled outside we will account for it
  hat[hat < 0] = np.nan
  hat = np.sqrt(hat)

  t1 = (-b[t0nt1n]+hat)/2
  t1n = t1 < 0

  t2 = (-b[t0nt1n][t1n]-hat[t1n])/2

  # we will have at least one positive
  t1[t1n] = t2

  # again this should only occur if the mouse is labelled outside of the arena
  # in this case the escaped mouse would have to be looking directly away from the arena
  t1[t1 < 0] = np.nan

  pos1 = np.empty(pos0.shape)
  pos1[:] = np.nan
  pos1[t0nt1n] = eye_head[t0nt1n, :]*t1[:, np.newaxis] + head_center[t0nt1n, :]

  return arc_2pi(pos0, arena_center, r0), arc_2pi(pos1, arena_center, r1)


def project_from_head_to_walls(pose, radiusInner, radiusOuter, center, gazePoint=0.5725):
  # left_ear = pose['leftear'][['x', 'y']].to_numpy()  # should be t-by-2
  # right_ear = pose['rightear'][['x', 'y']].to_numpy()
  # snout = pose['snout'][['x', 'y']].to_numpy()
  left_ear = np.stack((
      pose['leftear']['x'],
      pose['leftear']['y'],
  )).transpose()  # should be t-by-2
  right_ear = np.stack((
      pose['rightear']['x'],
      pose['rightear']['y'],
  )).transpose()  # should be t-by-2
  snout = np.stack((
      pose['snout']['x'],
      pose['snout']['y'],
  )).transpose()  # should be t-by-2

  head_center = (left_ear + right_ear)/2
  head_center_axis = head_center - center

  head_long_axis = snout - head_center
  head_angle = angle_2pi(head_long_axis)

  # we assume that the top of the mouse's head is... well... pointing upward
  # this means we can rotate wrt the +z axis and the math is simple
  left_eye_angle = head_angle + gazePoint
  right_eye_angle = head_angle - gazePoint

  # the eye angles define rays originating from the head_center that will intersect at least one of the circles
  left_eye_axis = np.stack((
      np.cos(left_eye_angle),
      np.sin(left_eye_angle)
  )).transpose()
  right_eye_axis = np.stack((
      np.cos(right_eye_angle),
      np.sin(right_eye_angle)
  )).transpose()

  thetaLInner, thetaLOuter = intersect_ray_circle(head_center, center,
                                                  left_eye_axis, head_center_axis, radiusInner, radiusOuter)
  thetaRInner, thetaROuter = intersect_ray_circle(head_center, center,
                                                  right_eye_axis, head_center_axis, radiusInner, radiusOuter)

  return thetaLInner, thetaLOuter, thetaRInner, thetaROuter


def is_in_window(theta, theta_window,arena_center):
    if isinstance(theta,np.ndarray):
        theta_window1=theta_window[0]-arena_center
        theta_window2=theta_window[1]-arena_center

        arc_theta_window1 = np.arctan2(theta_window1[1],theta_window1[0])
        arc_theta_window2 = np.arctan2(theta_window2[1], theta_window2[0])

        min_value=np.minimum(arc_theta_window1,arc_theta_window2)
        max_value=np.maximum(arc_theta_window1,arc_theta_window2)

        compare_min = np.transpose(theta>min_value)[0]
        compare_max = np.transpose(theta<max_value)[0]
        in_window= compare_max&compare_min
        return in_window




if __name__ == '__main__':
  # do some tests
  leftear_x = np.array([1.4, -1.6, -.1])
  leftear_y = np.array([1.1, 1.1, -1.5])

  rightear_x = np.array([1.6, -1.4, +.1])
  rightear_y = np.array([1.1, 1.1, -1.5])

  snout_x = np.array([1.5, -1.5, 0])
  snout_y = np.array([1.5, 1.5, -1.1])

  pose = DataFrame(
      {'leftear': {
          'x': leftear_x,
          'y': leftear_y
      }, 'rightear': {
          'x': rightear_x,
          'y': rightear_y,
      }, 'snout': {
          'x': snout_x,
          'y': snout_y,
      }})

  tLI, tLO, tRI, tRO = project_from_head_to_walls(
      pose, 1, 2, np.array([0, 0])[np.newaxis, :], 32.8/180*np.pi)

  print(tLI, tLO, tRI, tRO)
