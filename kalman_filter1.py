# measure move under gassian
# nametuple
from collections import namedtuple
from statistics import variance

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s : f"N(mean={s[0]:.3f}, var={s[1]:.3f})"

g1 = gaussian(3.4, 10.1)
g2 = gaussian(mean = 4.5, var = 0.2 ** 2)
print(g1)
print(g2)

g1[0], g2[1]
g1.mean, g2.var

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)

pos = gaussian(10.0, 0.2**2)
movement = gaussian(15.0, 0.7**2)
print(predict(pos, movement))

def gaussian_multiply(g1, g2):
    mean     = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)

def update(likelihood, prior):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

predicted_pos = gaussian(10.0, 0.2**2)
measured_pos  = gaussian(11.0, 0.1**2)

eatimated_pos = update(predicted_pos, measured_pos)

print(predicted_pos)
print(measured_pos)
print(eatimated_pos)