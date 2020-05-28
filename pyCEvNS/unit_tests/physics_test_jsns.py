import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# CONSTANTS
pe_per_mev = 10000
exposure = 3 * 208 * 50000
pim_rate_coherent = 0.0457
pim_rate_jsns = 0.4962
pim_rate_ccm = 0.0259
pim_rate = pim_rate_jsns
pot_mu = 0.07
pot_sigma = 0.04

prompt_pdf = np.genfromtxt('data/jsns/pion_kaon_neutrino_timing.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/jsns/mu_neutrino_timing.txt', delimiter=',')

def prompt_time(t):
    return np.interp(1000*t, prompt_pdf[:,0], prompt_pdf[:,1])

def delayed_time(t):
  return np.interp(1000 * t, delayed_pdf[:, 0], delayed_pdf[:, 1])

integral_delayed = quad(delayed_time, 0, 2)[0]
integral_prompt = quad(prompt_time, 0, 2)[0]

def prompt_prob(ta, tb):
  return quad(prompt_time, ta, tb)[0] / integral_prompt

def delayed_prob(ta, tb):
  return quad(delayed_time, ta, tb)[0] / integral_delayed


brem_photons = np.genfromtxt("data/jsns/brem.txt")  # binned photon spectrum from
Pi0Info = np.genfromtxt("data/jsns/Pi0_Info.txt")
pion_energy = Pi0Info[:,4] - massofpi0
pion_azimuth = np.arccos(Pi0Info[:,3] / np.sqrt(Pi0Info[:,1]**2 + Pi0Info[:,2]**2 + Pi0Info[:,3]**2))  # arccos of the unit z vector gives the azimuth angle
pion_cos = np.cos(np.pi/180 * Pi0Info[:,0])
pion_flux = np.array([pion_azimuth, pion_cos, pion_energy])
pion_flux = pion_flux.transpose()

# TODO: get pe efficiency for JSNS2. The following is for COHERENT.
def efficiency(pe):
  return 1
  a = 0.6655
  k = 0.4942
  x0 = 10.8507
  f = a / (1 + np.exp(-k * (pe - x0)))
  if pe < 5:
    return 0
  if pe < 6:
    return 0.5 * f
  return f


def get_energy_bins(e_a, e_b):
  return np.arange(e_a, e_b, step=20000 / pe_per_mev)


# Set up energy and timing bins
hi_energy_cut = 300  # mev
lo_energy_cut = 0.0  # mev
hi_timing_cut = 2.0
lo_timing_cut = 0.0
energy_edges = np.linspace(lo_energy_cut, hi_energy_cut,51) #get_energy_bins(lo_energy_cut, hi_energy_cut)
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.linspace(lo_timing_cut, hi_timing_cut, 51)
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_bg = 405 * exposure / exposure / (12 * 12) * np.ones((energy_bins.shape[0], len(timing_bins)))
flat_index = 0
for i in range(0, energy_bins.shape[0]):
  for j in range(0, timing_bins.shape[0]):
    n_meas[flat_index, 0] = energy_bins[i]
    n_meas[flat_index, 1] = timing_bins[j]
    flat_index += 1

flat_energies = n_meas[:,0]
flat_times = n_meas[:,1]

# Get neutrino spectrum
flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('jsns_prompt')
det = Detector("jsns_scintillator")
nsi = NSIparameters(0)
gen = NeutrinoElectronElasticVector(nsi)


er_list = np.linspace(0, 1000, 1000)
sigma_list = [gen.rates(er, 'mu', prompt_flux, det) for er in er_list]

plt.plot(er_list, sigma_list)
plt.yscale('log')
plt.show()







