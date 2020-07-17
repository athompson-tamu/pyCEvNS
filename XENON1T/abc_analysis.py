# Primakoff explanation of excess
import numpy as np
from numpy import sqrt, pi

import scipy.ndimage as nd

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
import matplotlib.ticker as tickr
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import sys
sys.path.append("../")
from pyCEvNS.axion import primakoff_scattering_xs_complete,primakoff_scattering_xs
from pyCEvNS.constants import *


# Define constants
Z_Xe = 54
r0Xe = 2.18e-10 / meter_by_mev
area_Xe = pi * (96/2)**2  # diameter 96 cm - fiducial?
exposure = 650 * 365 # kg days
n_atoms_Xe = 1000 * mev_per_kg / 122.3e3
prefactor = exposure * area_Xe * n_atoms_Xe * meter_by_mev ** 2

# Read in data
FILE_XENON1T = np.genfromtxt("data/XENON1T.txt", delimiter=",")
energy_edges = np.arange(1, 31, 1)
energy_bins = (energy_edges[1:] + energy_edges[:-1])/2

observations = FILE_XENON1T[:,1]
errors = np.sqrt(observations)

# Read in the ABC flux
FILE_ABC = np.genfromtxt("data/gaeflux.txt")

# Read in the Background model
B0_XENON1T = np.genfromtxt("data/XENON1T_B0.txt", delimiter=",")



# Experimental Efficiency
def eff(er):
    a = 0.87310709
    k = 3.27543615
    x0 = 1.50913422
    return a / (1 + np.exp(-k * (er - x0)))


# Background model
def B0(keV):
    return np.interp(keV, B0_XENON1T[:,0], B0_XENON1T[:,1])

background = B0(energy_bins)

# Solar axion flux
def FluxPrimakoff(ea, g): # g: GeV^-1; ea: keV
     # in keV^-1 cm^-2 s^-1
    return 6e30 * g**2 * np.power(ea, 2.481) * np.exp(-ea / 1.205)

# Solar ABC flux
def ABCFlux(keV, gae): # in 1/(keV cm day)
    coupling = 1e19 * (gae/(0.511*1e-10))**2 / 24 / 3600
    smeared_events = coupling * nd.gaussian_filter1d(FILE_ABC[:,1], sigma=10, mode="nearest")
    return np.interp(keV, FILE_ABC[:,0], smeared_events)


def EventsGeneratorABC(cube):
    gg = np.power(10.0, cube[0])  # in GeV^-1
    ge = np.power(10.0, cube[1])
    ma = np.power(10.0,cube[2]-3) # convert keV to MeV with -3
    print(gg, ge)
    events = np.histogram()
    print(events)
    return background + events


kev = np.linspace(0,20, 2000)
plt.plot(kev, eff(kev)*FluxPrimakoff(kev, 1e-10), label="Primakoff")
plt.plot(energy_bins, eff(energy_bins)*(ABCFlux(energy_bins, 1e-12)), label="ABC")
plt.legend()
plt.show()
plt.close()






