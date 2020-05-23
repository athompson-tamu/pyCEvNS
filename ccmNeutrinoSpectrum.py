import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


# CONSTANTS
pe_per_mev = 0.0878 * 13.348 * 1000
exposure = 3 * 365 * 7000
pim_rate_coherent = 0.0457
pim_rate_jsns = 0.4962
pim_rate_ccm = 0.0259
pim_rate = pim_rate_ccm
dist = 20
pot_mu = 0.145
pot_sigma = pot_mu/2


prompt_pdf = np.genfromtxt('data/ccm/arrivalTimePDF_prompt.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/ccm/arrivalTimePDF_delayed.txt', delimiter=',')

def prompt_time(t):
    return np.interp(t, prompt_pdf[:,0], prompt_pdf[:,1])

def delayed_time(t):
  return np.interp(t, delayed_pdf[:, 0], delayed_pdf[:, 1])

integral_delayed = quad(delayed_time, 0, 2)[0]
integral_prompt = quad(prompt_time, 0, 2)[0]

def prompt_prob(ta, tb):
  return quad(prompt_time, ta, tb)[0] / integral_prompt

def delayed_prob(ta, tb):
  return quad(delayed_time, ta, tb)[0] / integral_delayed


def efficiency(pe):
    return 1
    a = 0.85
    k = 0.4942
    x0 = 10.8507
    f = a / (1 + np.exp(-k * (pe - x0)))
    return f



# Set up energy and timing bins
hi_energy_cut = 0.1  # mev
lo_energy_cut = 0.0  # mev
hi_timing_cut = 0.4
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 0.005) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.03) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))

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
        n_prompt[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('ar'), exposure) \
                               * prompt_prob(timing_edges[j], timing_edges[j+1]) * efficiency(pe)
        n_delayed[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('ar'), exposure)
                                 + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, Detector('ar'), exposure)) \
                                * delayed_prob(timing_edges[j], timing_edges[j+1]) * efficiency(pe)
        flat_index += 1

n_nu = n_prompt+n_delayed


# Get DM events.
brem_photons = np.genfromtxt("data/ccm/brem.txt")  # binned photon spectrum from
Pi0Info = np.genfromtxt("data/ccm/Pi0_Info.txt")
pion_energy = Pi0Info[:,4] - massofpi0
pion_azimuth = np.arccos(Pi0Info[:,3] / np.sqrt(Pi0Info[:,1]**2 + Pi0Info[:,2]**2 + Pi0Info[:,3]**2))
pion_cos = np.cos(np.pi/180 * Pi0Info[:,0])
pion_flux = np.array([pion_azimuth, pion_cos, pion_energy])
pion_flux = pion_flux.transpose()
def GetDMEvents(g, m_dp, m_chi, m_med):
    dm_gen = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi,
                         life_time=0.001, expo=exposure, detector_type='ar')
    brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=m_dp, coupling=1,
                                dark_matter_mass=m_chi, pot_sample=1e5, pot_mu=pot_mu,
                                pot_sigma=pot_sigma, sampling_size=1000, life_time=0.0001,
                                detector_distance=dist, brem_suppress=True, verbose=False)
    pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_dp, coupling_quark=1,
                                           dark_matter_mass=m_chi, pion_rate=pim_rate,
                                           life_time=0.0001, pot_mu=pot_mu, pot_sigma=pot_sigma,
                                           detector_distance=dist)
    pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=m_dp, coupling_quark=1,
                                  dark_matter_mass=m_chi, pot_mu=pot_mu, pot_sigma=pot_sigma,
                                  life_time=0.0001, detector_distance=dist)
    
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



dm_events1 = GetDMEvents(m_chi=1, m_dp=75, m_med=25, g=0.6e-7)
dm_events2 = GetDMEvents(m_chi=20, m_dp=120, m_med=25, g=0.6e-7)




# Plot Dark Matter against Neutrino Spectrum
density = False
plt.hist([flat_energies*1000,flat_energies*1000], weights=[n_prompt, n_delayed], bins=energy_edges*1000,
         stacked=True, histtype='stepfilled', density=density, color=['black','dimgray'], label=["Prompt", "Delayed"])
plt.hist(flat_energies*1000,weights=dm_events1,bins=energy_edges*1000,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 1$ MeV)")
plt.hist(flat_energies*1000,weights=dm_events2,bins=energy_edges*1000,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 20$ MeV)")
plt.xlabel(r"$E_r$ [keVnr]")
plt.ylabel(r"Events")
plt.xlim((1000*lo_energy_cut, 1000*hi_energy_cut))
plt.legend()
plt.savefig("plots/coherent/neutrino_dm_spectra_lar.png")
plt.show()
plt.clf()


plt.hist([flat_times,flat_times], weights=[n_prompt, n_delayed], bins=timing_edges,
         stacked=True, histtype='stepfilled', density=density, color=['black','dimgray'], label=["Prompt", "Delayed"])
plt.hist(flat_times,weights=dm_events1,bins=timing_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 1$ MeV)")
plt.hist(flat_times,weights=dm_events2,bins=timing_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 20$ MeV)")
plt.xlabel(r"$t$ [$\mu$s]")
plt.ylabel(r"Events")
plt.legend()
plt.show()






