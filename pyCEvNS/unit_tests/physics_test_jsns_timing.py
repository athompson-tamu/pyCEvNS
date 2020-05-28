from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc


# CONSTANTS
pe_per_mev = 10000
exposure = 3 * 208 * 50000
pim_rate_coherent = 0.0457
pim_rate_jsns = 0.4962
pim_rate_ccm = 0.0259
pim_rate = pim_rate_jsns
pot_mu = 0.07
pot_sigma = 0.04

# PDFs
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

# TODO: get pe efficiency for JSNS2. The following is for COHERENT.



def get_energy_bins(e_a, e_b):
  return np.arange(e_a, e_b, step=20000 / pe_per_mev)


# Set up energy and timing bins
hi_energy_cut = 300  # mev
lo_energy_cut = 0.0  # mev
hi_timing_cut = 2.0
lo_timing_cut = 0.0
energy_edges = np.linspace(lo_energy_cut, hi_energy_cut,50) #get_energy_bins(lo_energy_cut, hi_energy_cut)
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.linspace(lo_timing_cut, hi_timing_cut, 48)
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


# Plot timing PDF, and events separately and multiplied.

# Get event weights
prompt_weights = np.empty_like(n_prompt)
delayed_weights = np.empty_like(n_prompt)

flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('jsns_prompt')
prompt_continuous = flux_factory.get('jsns_prompt_continuous')
delayed_flux = flux_factory.get('jsns_delayed')
det = Detector("jsns_scintillator")
nsi = NSIparameters(0)
gen = NeutrinoElectronElasticVector(nsi)
flat_index = 0
for i in range(0, energy_bins.shape[0]):
  print("On energy bin ", energy_bins[i], " out of ", energy_bins[-1])
  for j in range(0, timing_bins.shape[0]):
    e_a = energy_edges[i]
    e_b = energy_edges[i + 1]
    prompt_weights[flat_index] = gen.events(e_a, e_b, 'mu', prompt_flux, det, exposure)
    prompt_weights[flat_index] += gen.events(e_a, e_b, 'mu', prompt_continuous, det, exposure)
    delayed_weights[flat_index] = gen.events(e_a, e_b, 'e', delayed_flux, det, exposure) + gen.events(e_a, e_b, 'mubar', delayed_flux, det, exposure)
    n_prompt[flat_index] = gen.events(e_a, e_b, 'mu', prompt_flux, det, exposure) * \
                    prompt_prob(timing_edges[j], timing_edges[j+1])
    n_prompt[flat_index] += gen.events(e_a, e_b, 'mu', prompt_continuous, det, exposure) * \
                    prompt_prob(timing_edges[j], timing_edges[j+1])
    n_delayed[flat_index] = (gen.events(e_a, e_b, 'e', delayed_flux, det, exposure)
                    + gen.events(e_a, e_b, 'mubar', delayed_flux, det, exposure)) * \
                      delayed_prob(timing_edges[j], timing_edges[j+1])
    flat_index += 1

n_nu = n_prompt + n_delayed



# grab PDFs
binned_delayed = np.empty(timing_bins.shape[0])
binned_prompt = np.empty(timing_bins.shape[0])
for i in range(0, binned_prompt.shape[0]):
    binned_delayed[i] = delayed_prob(timing_edges[i], timing_edges[i+1])
    binned_prompt[i] = prompt_prob(timing_edges[i], timing_edges[i+1])


plt.hist([flat_times,flat_times], weights=[n_prompt, n_delayed], bins=timing_edges,
         stacked=True, histtype='step', density=True, color=['teal','tan'], label=["Prompt", "Delayed"])
plt.hist([timing_bins,timing_bins], weights=[binned_prompt, binned_delayed], bins=timing_edges,
         stacked=True, histtype='step', density=True, color=['blue','red'], ls='dashed', label=["Prompt PDF", "Delayed PDF"])

plt.legend()
plt.xlabel(r'$t$ [$\mu$s]')
plt.ylabel(r'a.u.')
plt.show()
plt.clf()


plt.hist([flat_energies,flat_energies], weights=[prompt_weights, delayed_weights], bins=energy_bins,
         stacked=False, histtype='step', density=False, color=['blue','red'], label=[r"Prompt ($\nu_\mu$)", r"Delayed ($\nu_e, \bar{\nu}_\mu$)"])
plt.legend()
plt.yscale('log')
plt.xlabel(r'$E_r$ [MeV]')
plt.ylabel(r'Flux $\times \sigma(\nu e^- \to \nu e^-)$')
plt.show()
plt.clf()
