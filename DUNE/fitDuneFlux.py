import numpy as np
from numpy import sqrt, log, log10, pi, exp

import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.special import exp1
from scipy.optimize import curve_fit

from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Declare constants.
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600*24
pot_per_year = 1.1e21


# detector constants
det_mass = 50000
det_am = 37.211e3  # mass of target atom in MeV
det_z = 18  # atomic number
days = 1000  # days of exposure
det_area = 3*6  # cross-sectional det area
det_thresh = 0  # energy threshold
sig_limit = 2.0  # poisson significance (2 sigma)


# Read in flux
pot_sample_g4 = 100000
pot_sample_py8 = 10000
scale_g4 = pot_per_year / pot_sample_g4
scale_py8 = pot_per_year / pot_sample_py8 
g4_data = np.genfromtxt("data/geant4_flux_DUNE_noPi0.txt", delimiter=",")
pythia_data = np.genfromtxt("data/hepmc_gamma_flux_from_pi0.txt")
gamma_e = g4_data[:,0]
gamma_theta = g4_data[:,1]
gamma_wgt = scale_g4*g4_data[:,2]
gamma_e_py8 = pythia_data[:,0]
gamma_theta_py8 = pythia_data[:,-1]
gamma_wgt_py8 = scale_py8*np.ones_like(gamma_e_py8)

energy_edges = np.linspace(0,100,100)
theta_edges = np.linspace(0,pi,100)


# Fit distribution
def GammaDist(energy, a, b, c):
    return a + b*energy + c*energy**2


# Get trimmed data
forward_data_g4 = g4_data[g4_data[:,1] < 0.05]
forward_data_py8 = pythia_data[pythia_data[:,-1] < 0.05]

# Perform the fit
y_data = log10(forward_data_g4[:,2] + np.ones(forward_data_g4.shape[0]))
x_data = log10(forward_data_g4[:,0] + np.ones(forward_data_g4.shape[0]))

popt, pcov = curve_fit(GammaDist, x_data, y_data)


plt.plot(10**x_data, 1e2*scale_g4*(10**GammaDist(x_data, *popt)), 'r-', label='fit')
plt.hist(forward_data_g4[:,0], weights=scale_g4*forward_data_g4[:,2], histtype='step', bins=energy_edges, label="GEANT4")
plt.hist(forward_data_py8[:,0], weights=scale_py8*np.ones_like(forward_data_py8[:,0]), histtype='step', bins=energy_edges, label="Pythia8")
plt.yscale('log')
plt.ylabel("Counts / Year")
plt.xlabel(r"$E$ [GeV]")
plt.title(r"$\theta < 0.05$ radians")
plt.legend()
plt.show()
plt.clf()