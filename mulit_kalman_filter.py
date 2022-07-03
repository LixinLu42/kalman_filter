# refenrence: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
from filterpy.common import Q_discrete_white_noise
from kf_book.mkf_internal import plot_all
import math
from scipy.linalg import inv
import numpy as np
from numpy.random import randn

def compute_person_data(z_var, process_var, count = 1, dt = 1):
    "返回的是我们的track, measurement"
    x, vel = 0., 1.
    z_std  = math.sqrt(z_var)
    p_std  = math.sqrt(process_var)

    xs, zs = [], []

    for _ in range(count):
        v = vel + (randn() * p_std)
        x += v*dt
        z = x + randn() * z_std

        xs.append(x) # predicted position/prior
        zs.append(z) # 观测

    return np.array(xs), np.array(zs)

dt    = 1.0 # 最小时间刻度
R_var = 50  # 测量误差的方差 # 生产设备，进行实验统计而来的
Q_var = 1

x     = np.array([[10.0, 4.5]]).T
P     = np.diag(  [500,   49])

F     = np.array([[1, dt],
                  [0,  1]]) # state transition matrix

H     = np.array([[1., 0]]) # measurement function

R     = np.array([[R_var]])

Q     = Q_discrete_white_noise(dim = 2, dt = dt, var = Q_var)

count = 50
tracks, zs = compute_person_data(R_var, Q_var, count)

xs, cov = [],[]

for z in zs:
    # predict
    x = F @ x
    P = F @ P @ F.T + Q

    # update
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    y = z - H @ x
    x += K @ y
    P = P - K @ H @ P

    xs.append(x)
    cov.append(P)

xs, cov = np.array(xs), np.array(cov)

from matplotlib.pyplot import figure
figure(figsize= (8,6), dpi = 80)

plot_all(xs[:,0], tracks, zs, cov, plot_P=False)





