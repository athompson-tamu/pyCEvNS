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
    n_prompt[flat_index] = efficiency(energy_bins[i] * pe_per_mev) * \
                  gen.events(e_a, e_b, 'mu', prompt_flux, det, exposure) * \
                    prompt_prob(timing_edges[j], timing_edges[j+1])
    n_delayed[flat_index] = efficiency(energy_bins[i] * pe_per_mev) * \
                   (gen.events(e_a, e_b, 'e', delayed_flux, det, exposure)
                    + gen.events(e_a, e_b, 'mubar', delayed_flux, det, exposure)) * \
                      delayed_prob(timing_edges[j], timing_edges[j+1])
    flat_index += 1

n_nu = n_prompt + n_delayed


# Plot JSNS^2 Dark Matter Signal
def GetDMEvents(m_chi, m_dp, m_med, g, lifetime=0.001):
    #eps = np.sqrt(1/137) * g
    dm_gen = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi, life_time=0.001, expo=exposure, detector_type='jsns_scintillator')
    brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=m_dp, coupling=g, dark_matter_mass=m_chi,
                              detector_distance=24, pot_mu=pot_mu, pot_sigma=pot_sigma, pot_sample=1e5,
                              sampling_size=1000, brem_suppress=True, verbose=False, life_time=lifetime)
    pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_dp, coupling_quark=g, dark_matter_mass=m_chi, pion_rate=pim_rate,
                                           pot_mu=pot_mu, pot_sigma=pot_sigma, detector_distance=24, life_time=lifetime)
    pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=m_dp, coupling_quark=g, dark_matter_mass=m_chi,
                                  pot_mu=pot_mu, pot_sigma=pot_sigma, detector_distance=24, life_time=lifetime)
    
    # First pulse
    dm_gen.fx = brem_flux
    brem_events = dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")[0]

    dm_gen.fx = pim_flux
    pim_events = dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")[0]

    dm_gen.fx = pi0_flux
    pi0_events = dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")[0]
    
    # Second pulse
    brem_flux.pot_mu = pot_mu + 0.5
    pim_flux.pot_mu = pot_mu + 0.5
    pi0_flux.pot_mu = pot_mu + 0.5
    brem_flux.simulate()
    pim_flux.simulate()
    pi0_flux.simulate()
    
    dm_gen.fx = brem_flux
    brem_events += dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")[0]

    dm_gen.fx = pim_flux
    pim_events += dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")[0]

    dm_gen.fx = pi0_flux
    pi0_events += dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")[0]

    return brem_events + pim_events + pi0_events

dm_events1 = GetDMEvents(m_chi=2, m_dp=75, m_med=75, g=1e-4)
dm_events2 = GetDMEvents(m_chi=25, m_dp=75, m_med=75, g=1e-4)





# Plot Dark Matter against Neutrino Spectrum
density = False
plt.hist([flat_energies,flat_energies], weights=[n_prompt, n_delayed], bins=energy_edges,
         stacked=True, histtype='stepfilled', density=density, color=['teal','tan'], label=["Prompt", "Delayed"])
plt.hist(flat_energies,weights=dm_events1,bins=energy_edges,
         histtype='step', density=density, label=r"DM ($M_{A^\prime} = 75$ MeV)")
plt.hist(flat_energies,weights=dm_events2,bins=energy_edges,
         histtype='step', density=density, label=r"DM ($M_{A^\prime} = 300$ MeV)")
plt.xlabel(r"$E_r$ [MeVer]")
plt.yscale("log")
plt.ylabel(r"Events")
plt.title(r"$\epsilon=10^{-4}$, $M_X = 25$ MeV, $m_{\chi} = 5$ MeV, timing cut $0.1 < t < 0.25$ $\mu$s", loc='right')
plt.xlim((lo_energy_cut,hi_energy_cut))
plt.legend()
plt.savefig("plots/jsns2/neutrino_dm_spectra_25e-2mus.png")
plt.show()
plt.clf()


dm_events1 = GetDMEvents(m_chi=25, m_dp=75, m_med=25, g=1e-4, lifetime=0.001)
dm_events2 = GetDMEvents(m_chi=25, m_dp=75, m_med=25, g=2e-4, lifetime=1)
dm_events3 = GetDMEvents(m_chi=5, m_dp=138, m_med=25, g=2e-4, lifetime=1)
print(np.sum(dm_events1), np.sum(dm_events2), np.sum(dm_events3))

density = True
plt.hist([flat_times,flat_times], weights=[n_prompt, n_delayed], bins=timing_edges,
         stacked=True, histtype='stepfilled', density=density, color=['teal','tan'], label=["Prompt", "Delayed"])
plt.hist(flat_times,weights=dm_events1,bins=timing_edges, color='blue',
         histtype='step', density=density, label=r"DM ($M_X = 75$ MeV, $\tau < 0.001$ $\mu$s)")
plt.hist(flat_times,weights=dm_events2,bins=timing_edges, color='crimson',
         histtype='step', density=density, label=r"DM ($M_X = 75$ MeV, $\tau = 1$ $\mu$s)")
plt.hist(flat_times,weights=dm_events3,bins=timing_edges, color='darkorange',
         histtype='step', density=density, label=r"DM ($M_X = 138$ MeV, $\tau = 1$ $\mu$s)")
plt.xlabel(r"$t$ [$\mu$s]", fontsize=15)
plt.ylabel(r"a.u.", fontsize=15)
plt.title(r"JSNS$^2$", loc="right", fontsize=15)
plt.xlim((0,2))
plt.ylim((0,5))
plt.legend(fontsize=12)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()






