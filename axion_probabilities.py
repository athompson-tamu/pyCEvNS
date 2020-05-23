
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

miner_flux = np.genfromtxt('data/miner/reactor_photon.txt')  # flux at reactor surface
miner_flux[:, 1] *= 1e8 # get flux at the core

# Flux from pythia8 (Doojin)
pot_per_year = 1.1e21
pot_sample = 10000
scale = pot_per_year / pot_sample / 365 / 24 / 3600
gamma_data = np.genfromtxt("data/dune/hepmc_gamma_flux_from_pi0.txt")
gamma_e = 1000*gamma_data[:,0] # convert to mev

# bin coarse flux
flux_edges = np.linspace(min(gamma_e), max(gamma_e), 25)
flux_bins = (flux_edges[:-1] + flux_edges[1:]) / 2
flux_hist = np.histogram(gamma_e, weights=scale*np.ones_like(gamma_e), bins=flux_bins)[0]
flux = np.array(list(zip(flux_bins,flux_hist)))

def getDecayProbability(ma, g):
    axion_sim_primakoff = IsotropicAxionFromPrimakoff(flux, ma, g,
                                            240e3, 90, 15e-24, det_dis, 0.2)
    axion_prim_e = axion_sim_primakoff.axion_energy
    axion_prim_v = axion_sim_primakoff.axion_velocity
    axion_prim_w = axion_sim_primakoff.axion_weight
    axion_prim_surv = axion_sim_primakoff.axion_surv_prob
    axion_prim_decay = axion_sim_primakoff.axion_decay_prob
    return axion_prim_v, axion_prim_w, axion_prim_decay





velocity_bins = np.linspace(0.0,1, 10)
velocity_centers = (velocity_bins[1:] + velocity_bins[:-1])/2

v1, w1, p1 = getDecayProbability(1, 1e-8)
v2, w2, p2 = getDecayProbability(0.1, 1e-8)
v3, w3, p3 = getDecayProbability(5, 5e-9)

# Plot the velocity distribution
plt.scatter(v1, p1, color="red",
         label=r"$m_a = 1$ MeV, $g_{a\gamma\gamma} = 10^{-5}$ GeV$^{-1}$")
plt.scatter(v2, p2, color="blue",
         label=r"$m_a = 0.1$ MeV, $g_{a\gamma\gamma} = 10^{-5}$ GeV$^{-1}$")
plt.scatter(v3, p3, color="green",
         label=r"$m_a = 5$ MeV, $g_{a\gamma\gamma} = 5\cdot 10^{-6}$ GeV$^{-1}$")
plt.xlabel(r"$v_a / c$", fontsize=15)
plt.ylabel("a.u.", fontsize=15)
plt.yscale('log')
#plt.xlim((0.0,1))
#plt.ylim((1e-4,1e3))
#plt.xscale('log')
plt.legend(fontsize="12", loc="upper left")
plt.show()
plt.clf()