from socket import EAI_NODATA
import numpy as np
from numpy.random import randn
from math import sqrt
import matplotlib.pyplot as plt

from collections import namedtuple
from statistics import variance

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s : f"N(mean={s[0]:.3f}, var={s[1]:.3f})"


def print_result(predict, update, z, epoch):

    # 细节暂不需要深究

    # predicted_pos, updated_posclear, measured_pos

    predict_template = '{:3.0f} {: 7.3f} {: 8.3f}'

    update_template  = '\t{: .3f}\t{: 7.3f} {: 7.3f}'

    print(predict_template.format(epoch, predict[0], predict[1]),end='\t')

    print(update_template.format(z, update[0], update[1]))

def plot_result(epochs ,prior_list, x_list, z_list):

    epoch_list = np.arange(epochs)

    plt.plot(epoch_list, prior_list, linestyle=':', color='r',label = "prior/predicted_pos", lw=2)

    plt.plot(epoch_list, x_list, linestyle='-', color='g', label = "posterior/updated_pos",lw=2)

    plt.plot(epoch_list, z_list, linestyle=':', color='b', label = "likelihood/measurement", lw=2)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)

def gaussian_multiply(g1, g2):
    mean     = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)

def update(likelihood, prior):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior


# init variables
motion_var = 1.0 # person var
sensor_var = 2.0 # GPS var
x          = gaussian(0, 2.0**2)
velocity   = 1.0
dt         = 1 #
motion_model = gaussian(velocity, motion_var)

# generation data
zs = []
current_x = x.mean
for _ in range(10): # 我们让小明走10s,每走1s就看看GPS，并且把这些数据存起来
    # 2.1 先生成我们的运动数据
    v = velocity + randn() * motion_var
    current_x += v*dt    # 将上一秒的位移加到这一秒
    
    # 2.2 生成观测数据
    measurement = current_x + randn() * sensor_var  # gps观测也有一定误差
    zs.append(measurement)

print(zs)

prior_list, x_list, z_list = [], [], []

print('epoch\tPREDICT\t\t\tUPDATE')

print('     \tx      var\t\t  z\t    x      var')

for epoch, z in enumerate(zs):
    prior = predict(x, motion_model)  # 运动预测 # 两个高斯之和
    likelihood = gaussian(z, sensor_var)

    x = update(likelihood, prior)     # 结合观测 # 两个高斯的交集

    print_result(prior, x, z, epoch)
    prior_list.append(prior.mean)
    x_list.append(x.mean)
    z_list.append(z)

print()
print(f"final estimate:       {x.mean:30.3f}")
print(f"actual final estimate:{current_x:30.3f}")

# plot_result(10, prior_list, x_list, z_list)







