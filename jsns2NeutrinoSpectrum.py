import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

prompt_pdf = np.genfromtxt('data/jsns/pion_kaon_neutrino_timing.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/jsns/mu_neutrino_timing.txt', delimiter=',')


# CONSTANTS
# TODO: update pe_per_mev for JSNS2
pe_per_mev = 0.0878 * 13.348 * 1000
exposure = 3 * 365 * 1000  #4466


# TODO: update prompt and delayed pdfs.
def prompt_time(t):
    return np.interp(1000*t, prompt_pdf[:,0], prompt_pdf[:,1]) / np.sum(prompt_pdf[:,1])


def delayed_time(t):
  return np.interp(1000 * t, delayed_pdf[:, 0], delayed_pdf[:, 1]) / np.sum(delayed_pdf[:, 1])


prompt_flux = Flux('prompt')
delayed_flux = Flux('delayed')

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
  return np.arange(e_a, e_b, step=2000 / pe_per_mev)








# Set up energy and timing bins
hi_energy_cut = 100  # mev
lo_energy_cut = 0.0  # mev
energy_edges = get_energy_bins(lo_energy_cut, hi_energy_cut)
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.linspace(0, 2, 20)
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_bg = 405 * exposure / exposure / (12 * 12) * np.ones((energy_bins.shape[0], len(timing_bins)))
flat_index = 0
for i in range(0, energy_bins.shape[0]):
  for j in range(0, timing_bins.shape[0]):
    n_meas[flat_index, 0] = energy_bins[i] * pe_per_mev
    n_meas[flat_index, 1] = timing_bins[j]
    flat_index += 1

energies = n_meas[:, 0] / pe_per_mev






# Plot JSNS^2 Dark Matter Signal
def GetDMEvents(m_chi, m_dp, m_med, g):
    photon_flux = np.genfromtxt("data/jsns_3gev_photon_totals_1e5_POT.txt")  # binned photon spectrum from
    dm_gen = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi, life_time=1, expo=exposure, detector_type='ar')
    dm_flux = DMFluxIsoPhoton(photon_flux, dark_photon_mass=m_dp, coupling=g, dark_matter_mass=m_chi,
                              detector_distance=24, pot_mu=0.145, pot_sigma=0.1, pot_sample=1e5,
                              sampling_size=2000, verbose=False)
    dm_gen.fx = dm_flux
    return dm_gen.events(m_med, g, n_meas, channel="electron")


dm_events = GetDMEvents(m_chi=0.1, m_dp=75, m_med=25, g=1e-4)
dm_events2 = GetDMEvents(m_chi=5, m_dp=75, m_med=25, g=1e-4)
dm_events3 = GetDMEvents(m_chi=10, m_dp=75, m_med=25, g=1e-4)
dm_events4 = GetDMEvents(m_chi=20, m_dp=75, m_med=25, g=1e-4)
dm_events5 = GetDMEvents(m_chi=50, m_dp=150, m_med=25, g=1e-4)
dm_events6 = GetDMEvents(m_chi=80, m_dp=200, m_med=25, g=1e-4)





# Get neutrino spectrum
flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
delayed_flux = flux_factory.get('coherent_delayed')
det = Detector("ar")
nsi = NSIparameters(0)
gen = NeutrinoElectronElasticVector(nsi)
flat_index = 0
for i in range(0, energy_bins.shape[0]):
  print("On energy bin ", energy_bins[i], " out of ", energy_bins[-1])
  for j in range(0, timing_bins.shape[0]):
    e_a = energy_edges[i]
    e_b = energy_edges[i + 1]
    n_prompt[flat_index] = efficiency(energy_bins[i] * pe_per_mev) * \
                  gen.events(e_a, e_b, 'mu', prompt_flux, det, exposure) * prompt_time(timing_bins[j])
    n_delayed[flat_index] = efficiency(energy_bins[i] * pe_per_mev) * \
                   (gen.events(e_a, e_b, 'e', delayed_flux, det, exposure)
                    + gen.events(e_a, e_b, 'mubar', delayed_flux, det, exposure)) * delayed_time(timing_bins[j])
    flat_index += 1

n_nu = n_prompt + n_delayed

plt.clf()
plt.plot(timing_bins, prompt_time(timing_bins))
plt.plot(timing_bins, delayed_time(timing_bins))
plt.savefig("plots/jsns2/timing.png")

plt.clf()
plt.hist2d(n_meas[:,0] / pe_per_mev, n_meas[:,1], weights=n_nu, bins=[energy_edges, timing_edges])
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"$t$ [$\mu$s]")
plt.savefig("plots/jsns2/neutrino_energy_spectrum.png")

plt.clf()
plt.hist2d(n_meas[:,0] / pe_per_mev, n_meas[:,1], weights=n_delayed, bins=[energy_edges, timing_edges])
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"$t$ [$\mu$s]")
plt.savefig("plots/jsns2/delayed_energy_spectrum.png")

plt.clf()
plt.hist2d(n_meas[:,0] / pe_per_mev, n_meas[:,1], weights=n_prompt, bins=[energy_edges, timing_edges])
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"$t$ [$\mu$s]")
plt.savefig("plots/jsns2/prompt_energy_spectrum.png")

plt.clf()





# Plot Dark Matter against Neutrino Spectrum
plt.hist(n_meas[:,0]/pe_per_mev, weights=n_nu, bins=energy_edges,
         histtype='stepfilled', color='dimgray', label="Prompt + Delayed Neutrinos")
plt.hist(energies,weights=dm_events,bins=energy_edges,
         histtype='step',label=r"DM ($m_{\chi} = 0.1$ MeV)")
plt.hist(energies,weights=dm_events2,bins=energy_edges,
         histtype='step',label=r"DM ($m_{\chi} = 5$ MeV)")
plt.hist(energies,weights=dm_events3,bins=energy_edges,
         histtype='step',label=r"DM ($m_{\chi} = 10$ MeV)")
plt.hist(energies,weights=dm_events4,bins=energy_edges,
         histtype='step',label=r"DM ($m_{\chi} = 20$ MeV)")
plt.hist(energies,weights=dm_events5,bins=energy_edges,
         histtype='step',label=r"DM ($m_{\chi} = 50$ MeV)")
plt.hist(energies,weights=dm_events6,bins=energy_edges,
         histtype='step',label=r"DM ($m_{\chi} = 80$ MeV)")
plt.xlabel(r"$E_r$ [MeV]")
#plt.yscale("log")
plt.ylabel(r"a.u.")
plt.xlim((0,100))
plt.legend()
plt.savefig("plots/jsns2/neutrino_dm_spectra.png")







