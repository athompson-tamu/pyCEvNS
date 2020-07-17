import sys
sys.path.append("/Users/adrianthompson/physics/dark_photon/pyCEvNS_physics_analysis/pyCEvNS/")

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


prompt_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')
ac_bon = np.genfromtxt('data/coherent/data_anticoincidence_beamOn.txt', delimiter=',')
c_bon = np.genfromtxt('data/coherent/data_coincidence_beamOn.txt', delimiter=',')
ac_boff = np.genfromtxt('data/coherent/data_anticoincidence_beamOff.txt', delimiter=',')
c_boff = np.genfromtxt('data/coherent/data_coincidence_beamOff.txt', delimiter=',')
nin_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_promptNeutrons.txt', delimiter=',')



# CONSTANTS
pe_per_mev = 0.0878 * 13.348 * 1000
exposure = 4466
pim_rate_coherent = 0.0457
pim_rate_jsns = 0.4962
pim_rate_ccm = 0.0259
pim_rate = pim_rate_coherent


# Prompt, delayed PDFs and efficiency functions for COHERENT CsI.
def prompt_time(t):
    if t < 0.25 or t > 11.75:
        return 0
    else:
        return prompt_pdf[int((t-0.25)/0.5), 1]

def delayed_time(t):
    if t < 0.25 or t > 11.75:
        return 0
    else:
        return delayed_pdf[int((t-0.25)/0.5), 1]

def nin_time(t):
    if t < 0.25 or t > 11.75:
        return 0
    else:
        return nin_pdf[int((t-0.25)/0.5), 1]

def efficiency(pe):
  a = 0.6655
  k = 0.4942
  x0 = 10.8507
  f = a / (1 + np.exp(-k * (pe - x0)))
  if pe < 5:
    return 0
  if pe < 6:
    return 0.5 * f
  return f





hi_energy_cut = 52/pe_per_mev
lo_energy_cut = 0/pe_per_mev
hi_timing_cut = 6.25
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 0.25/pe_per_mev) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2
n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))

flat_index = 0
for i in range(0, energy_bins.shape[0]):
  for j in range(0, timing_bins.shape[0]):
    n_meas[flat_index, 0] = energy_bins[i]
    n_meas[flat_index, 1] = timing_bins[j]
    flat_index += 1

flat_energies = n_meas[:,0]
flat_times = n_meas[:,1]

flat_energies = n_meas[:,0]
flat_times = n_meas[:,1]
kevnr = flat_energies*1000
kevnr_bins = energy_bins*1000
kevnr_edges = energy_edges*1000


# Setup prompt and delayed
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_prompt_cont = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_eff = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_ideal = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_ideal_55 = np.zeros(energy_bins.shape[0] * len(timing_bins))

# Get the theoretical prediction for the neutrino events.
flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
prompt_flux_continuous = flux_factory.get('jsns_prompt_continuous')
nsi = NSIparameters(0)
nu_gen = NeutrinoNucleusElasticVector(nsi)
flat_index = 0
for i in range(0, energy_bins.shape[0]):
    for j in range(0, timing_bins.shape[0]):
        e_a = energy_edges[i]
        e_b = energy_edges[i + 1]
        pe = energy_bins[i] * pe_per_mev
        t = timing_bins[j]
        n_prompt[flat_index] = (nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('csi'), exposure) \
                                + nu_gen.events(e_a, e_b, 'mu',prompt_flux_continuous, Detector('csi'), exposure)) \
                               * prompt_time(t) * efficiency(pe)
        n_prompt_cont[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux_continuous, Detector('csi'), exposure) \
                                    * prompt_time(t) * efficiency(pe)
        n_ideal[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('csi'), exposure) \
                               * prompt_time(t)
        n_eff[flat_index] = efficiency(pe)
        flat_index += 1

# make plots
plt.hist(kevnr,weights=n_ideal,bins=kevnr_edges, color='green',
         histtype='step', density=True, label=r"perfect efficiency $\frac{d\sigma}{dE_r} dE_r$, delta")
plt.hist(kevnr,weights=n_prompt_cont,bins=kevnr_edges, color='m',
         histtype='step', density=True, label=r"perfect efficiency $\frac{d\sigma}{dE_r} dE_r$, continuous")
plt.hist(kevnr,weights=n_eff,bins=kevnr_edges, color='red',
         histtype='step', density=True, label=r"efficiency $\epsilon(E_r)$")
plt.hist(kevnr,weights=n_prompt,bins=kevnr_edges, color='blue',
         histtype='step', density=True, label=r"prompt total $\epsilon(E_r) \times \frac{d\sigma}{dE_r} dE_r$")

plt.title(r"Prompt $\nu_\mu$ neutrinos (29 MeV)")
plt.xlabel(r"$E_r$ [keV]")
plt.ylabel(r"a.u.")
plt.xlim((0,35))
plt.legend()

plt.show()
plt.close()