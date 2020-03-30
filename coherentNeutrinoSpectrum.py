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
hi_energy_cut = 50/pe_per_mev  #0.026 mev
lo_energy_cut = 0.0  # 0.014mev
hi_timing_cut = 1.5
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 2/pe_per_mev) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_bg = 405/(12*12)*np.ones(energy_bins.shape[0] * len(timing_bins))

flat_index = 0
for i in range(0, energy_bins.shape[0]):
  for j in range(0, timing_bins.shape[0]):
    n_meas[flat_index, 0] = energy_bins[i]
    n_meas[flat_index, 1] = timing_bins[j]
    flat_index += 1

flat_energies = n_meas[:,0]
flat_times = n_meas[:,1]


flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
delayed_flux = flux_factory.get('coherent_delayed')
nsi = NSIparameters(0)
nu_gen = NeutrinoNucleusElasticVector(nsi)
flat_index = 0
for i in range(0, energy_bins.shape[0]):
    for j in range(0, timing_bins.shape[0]):
        e_a = energy_edges[i]
        e_b = energy_edges[i + 1]
        pe = energy_bins[i] * pe_per_mev
        t = timing_bins[j]
        n_prompt[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('csi'), exposure) \
                               * prompt_time(t) * efficiency(pe)
        n_delayed[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('csi'), exposure)
                                 + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, Detector('csi'), exposure)) \
                                * delayed_time(t) * efficiency(pe)
        flat_index += 1

n_nu = n_prompt+n_delayed


# Get DM events.
brem_photons = np.genfromtxt("data/coherent/brem.txt")  # binned photon spectrum from
Pi0Info = np.genfromtxt("data/coherent/Pi0_Info.txt")
pion_energy = Pi0Info[:,4] - massofpi0
pion_azimuth = np.arccos(Pi0Info[:,3] / np.sqrt(Pi0Info[:,1]**2 + Pi0Info[:,2]**2 + Pi0Info[:,3]**2))
pion_cos = np.cos(np.pi/180 * Pi0Info[:,0])
pion_flux = np.array([pion_azimuth, pion_cos, pion_energy])
pion_flux = pion_flux.transpose()
def GetDMEvents(g, m_dp, m_chi, m_med):
    dm_gen = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi,
                         life_time=0.001, expo=exposure, detector_type='csi')
    brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=m_dp, coupling=1,
                                dark_matter_mass=m_chi, pot_sample=1e5,
                                sampling_size=1000, life_time=0.0001,
                                verbose=False)
    pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_dp,
                                           coupling_quark=1,
                                           dark_matter_mass=m_chi,
                                           pion_rate=pim_rate,
                                           life_time=0.0001)
    pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux,
                                  dark_photon_mass=m_dp, coupling_quark=1,
                                  dark_matter_mass=m_chi,
                                  life_time=0.0001)
    
    dm_gen.fx = brem_flux
    brem_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                                channel="nucleus")

    dm_gen.fx = pim_flux
    pim_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")

    dm_gen.fx = pi0_flux
    pi0_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")

    return brem_events[0] + pim_events[0] + pi0_events[0]



dm_events1 = GetDMEvents(m_chi=1, m_dp=75, m_med=25, g=0.75e-7)
dm_events2 = GetDMEvents(m_chi=20, m_dp=120, m_med=25, g=0.75e-7)




# Plot Dark Matter against Neutrino Spectrum
density = False
plt.hist([flat_energies*pe_per_mev,flat_energies*pe_per_mev], weights=[n_prompt, n_delayed], bins=energy_edges*pe_per_mev,
         stacked=True, histtype='stepfilled', density=density, color=['black','dimgray'], label=["Prompt", "Delayed"])
plt.hist(flat_energies*pe_per_mev,weights=dm_events1,bins=energy_edges*pe_per_mev,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 1$ MeV)")
plt.hist(flat_energies*pe_per_mev,weights=dm_events2,bins=energy_edges*pe_per_mev,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 20$ MeV)")
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"a.u.")
plt.legend()
plt.savefig("plots/coherent/neutrino_dm_spectra.png")
plt.show()
plt.clf()


plt.hist([flat_times,flat_times], weights=[n_prompt, n_delayed], bins=timing_edges,
         stacked=True, histtype='stepfilled', density=density, color=['black','dimgray'], label=["Prompt", "Delayed"])
plt.hist(flat_times,weights=dm_events1,bins=timing_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 1$ MeV)")
plt.hist(flat_times,weights=dm_events2,bins=timing_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 20$ MeV)")
plt.xlabel(r"$t$ [$\mu$s]")
plt.ylabel(r"a.u.")
plt.legend()
plt.show()






