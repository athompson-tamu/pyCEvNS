import numpy as np

from numpy import arccos, cos, sin, sqrt, log, log10, exp, pi

from scipy.integrate import dblquad, quad

import matplotlib.pyplot as plt


theta_det = 0.01
theta_gamma = 0.000001

ma = 1.0
energy = 1000.0

def heaviside(theta, phi):
    return theta_det > arccos(cos(theta)*cos(theta_gamma) \
                            + cos(phi)*sin(theta)*sin(theta_gamma))



def diffxs(theta):
    denom = ma**2.0 + 2.0*energy*(sqrt(energy**2.0 - ma**2.0)*cos(theta) - energy)
    return sin(theta)**3.0 / (denom)**2.0



thetas = np.linspace(0,pi,1000)




#plt.plot(thetas, diffxs(thetas))

#plt.show()
#plt.close()



def integrand(theta, phi):
    return diffxs(theta) * heaviside(theta, phi)

conv = np.vectorize(integrand)




ans = dblquad(conv, 0, pi, 0,pi)[0]

print(ans)




theta_r = np.random.uniform(0,pi/8,10000)
phi_r = np.random.uniform(0,2*pi,10000)

hV = np.vectorize(heaviside)


wgts_r = conv(theta_r, phi_r)


print(wgts_r.shape)

plt.hist2d(phi_r, theta_r, weights=wgts_r, bins=[100,100])
plt.show()

plt.close()













