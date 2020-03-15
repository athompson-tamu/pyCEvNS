import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

prompt_pdf = np.genfromtxt('data/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')


# CONSTANTS
pe_per_mev = 0.0878 * 13.348 * 1000
exposure = 4466
pim_rate_coherent = 0.0457
pim_rate_jsns = 0.4962
pim_rate_ccm = 0.0259
pim_rate = pim_rate_coherent


# TODO: update prompt and delayed pdfs.
def prompt_time(t):
    return np.interp(1000*t, prompt_pdf[:,0], prompt_pdf[:,1]) / np.sum(prompt_pdf[:,1])


def delayed_time(t):
  return np.interp(1000 * t, delayed_pdf[:, 0], delayed_pdf[:, 1]) / np.sum(delayed_pdf[:, 1])


prompt_flux = Flux('prompt')
delayed_flux = Flux('delayed')

# TODO: get pe efficiency for JSNS2. The following is for COHERENT.
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


def get_energy_bins(e_a, e_b):
  return np.arange(e_a, e_b, step=1000 / pe_per_mev)





# Set up energy and timing bins
hi_energy_cut = 20  # mev
lo_energy_cut = 0.0  # mev
energy_edges = get_energy_bins(lo_energy_cut, hi_energy_cut)
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
print("Making ", energy_bins.shape[0], " energy bins on [",energy_bins[0], ",", energy_bins[-1], "]")
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
    brem_flux = np.genfromtxt("data/coherent/brem.txt")  # binned photon spectrum from
    dm_gen = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi, life_time=1, expo=exposure, detector_type='csi')
    brem_flux = DMFluxIsoPhoton(brem_flux, dark_photon_mass=m_dp, coupling=g, dark_matter_mass=m_chi,
                              pot_sample=1e5, sampling_size=1000, verbose=False)
    pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_dp, coupling_quark=g, dark_matter_mass=m_chi, pion_rate=pim_rate)
    dm_gen.fx = brem_flux
    brem_events = dm_gen.events(m_med, g, n_meas, channel="nucleus")

    dm_gen.fx = pim_flux
    pim_events = dm_gen.events(m_med, g, n_meas, channel="nucleus")

    return brem_events + pim_events


dm_events1 = GetDMEvents(m_chi=1, m_dp=75, m_med=25, g=5e-4)
dm_events2 = GetDMEvents(m_chi=20, m_dp=120, m_med=25, g=5e-4)





# Get neutrino spectrum
flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
delayed_flux = flux_factory.get('coherent_delayed')
det = Detector("csi")
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
print("neutrinos:", n_nu)

plt.clf()
plt.plot(timing_bins, prompt_time(timing_bins))
plt.plot(timing_bins, delayed_time(timing_bins))
plt.savefig("plots/coherent/timing.png")

plt.clf()
plt.hist2d(n_meas[:,0] / pe_per_mev, n_meas[:,1], weights=n_nu, bins=[energy_edges, timing_edges])
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"$t$ [$\mu$s]")
plt.savefig("plots/coherent/neutrino_energy_spectrum.png")

plt.clf()
plt.hist2d(n_meas[:,0] / pe_per_mev, n_meas[:,1], weights=n_delayed, bins=[energy_edges, timing_edges])
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"$t$ [$\mu$s]")
plt.savefig("plots/coherent/delayed_energy_spectrum.png")

plt.clf()
plt.hist2d(n_meas[:,0] / pe_per_mev, n_meas[:,1], weights=n_prompt, bins=[energy_edges, timing_edges])
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"$t$ [$\mu$s]")
plt.savefig("plots/coherent/prompt_energy_spectrum.png")

plt.clf()



print(dm_events1)

# Plot Dark Matter against Neutrino Spectrum
density = True
plt.hist([n_meas[:,0]/pe_per_mev,n_meas[:,0]/pe_per_mev], weights=[n_prompt, n_delayed], bins=energy_edges,
         stacked=True, histtype='stepfilled', density=density, color=['black','dimgray'], label=["Prompt", "Delayed"])
plt.hist(energies,weights=dm_events1,bins=energy_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 1$ MeV)")
plt.hist(energies,weights=dm_events2,bins=energy_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 20$ MeV)")
plt.xlabel(r"$E_r$ [MeV]")
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"a.u.")
plt.xlim((0.001,20))
plt.legend()
plt.savefig("plots/coherent/neutrino_dm_spectra.png")
print(np.sum(n_nu), np.sum(dm_events1), np.sum(dm_events2))
plt.show()







