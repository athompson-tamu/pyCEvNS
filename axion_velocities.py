
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


import sys

from pyCEvNS.axion import IsotropicAxionFromCompton, IsotropicAxionFromPrimakoff
det_dis = 2.5
det_mass = 4
det_am = 65.13e3
det_z = 32
days = 1
det_area = 0.2 ** 2
det_thresh = 0

miner_flux = np.genfromtxt('data/reactor_photon.txt')  # flux at reactor surface
miner_flux[:, 1] *= 1e8 # get flux at the core

def getVelocities(ma, g):
    axion_sim_primakoff = IsotropicAxionFromPrimakoff(miner_flux, ma, g,
                                            240e3, 90, 15e-24, det_dis, 0.2)

    axion_prim_e = axion_sim_primakoff.axion_energy
    axion_prim_v = axion_sim_primakoff.axion_velocity
    axion_prim_w = axion_sim_primakoff.axion_weight
    return axion_prim_v, axion_prim_w

velocity_bins = np.linspace(0.0,1, 20)
velocity_centers = (velocity_bins[1:] + velocity_bins[:-1])/2

v1, w1 = getVelocities(1, 1e-8)
print(np.sum(w1))
v2, w2 = getVelocities(0.1, 1e-8)
v3, w3 = getVelocities(5, 1e-6)

# Plot the velocity distribution
plt.hist(v1, weights=w1, bins=velocity_bins, density=True, color="red",
         histtype='step', label=r"$m_a = 1$ MeV, $g_{a\gamma\gamma} = 10^{-5}$ GeV$^{-1}$")
plt.hist(v2, weights=w2, bins=velocity_bins, density=True, color="blue",
         histtype='step', label=r"$m_a = 0.1$ MeV, $g_{a\gamma\gamma} = 10^{-5}$ GeV$^{-1}$")
plt.hist(v3, weights=w3, bins=velocity_bins, density=True, color="green",
         histtype='step', label=r"$m_a = 5$ MeV, $g_{a\gamma\gamma} = 5\cdot 10^{-6}$ GeV$^{-1}$")
plt.xlabel(r"$v_a / c$", fontsize=15)
plt.ylabel("a.u.", fontsize=15)
#plt.yscale('log')
plt.xlim((0.0,1))
#plt.ylim((1e-4,1e3))
#plt.xscale('log')
plt.legend(fontsize="12", loc="upper left")
plt.show()
plt.clf()