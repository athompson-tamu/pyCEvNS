import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.05) # 0.05 mus time resolution
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
def GetDMEvents(g, m_dp, m_chi, m_med, lifetime=0.001):
    dm_gen = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi,
                         life_time=0.001, expo=exposure, detector_type='ar')
    brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=m_dp, coupling=1,
                                dark_matter_mass=m_chi, pot_sample=1e5, pot_mu=pot_mu,
                                pot_sigma=pot_sigma, sampling_size=1000, life_time=lifetime,
                                detector_distance=dist, brem_suppress=True, verbose=False)
    pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_dp, coupling_quark=1,
                                           dark_matter_mass=m_chi, pion_rate=pim_rate,
                                           life_time=lifetime, pot_mu=pot_mu, pot_sigma=pot_sigma,
                                           detector_distance=dist)
    pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=m_dp, coupling_quark=1,
                                  dark_matter_mass=m_chi, pot_mu=pot_mu, pot_sigma=pot_sigma,
                                  life_time=lifetime, detector_distance=dist)
    
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


dm_events1 = GetDMEvents(m_chi=5, m_dp=75, m_med=15, g=0.3e-8)
dm_events2 = GetDMEvents(m_chi=25, m_dp=75, m_med=75, g=0.1e-7)




# Plot Dark Matter against Neutrino Spectrum
density = False
kevnr = flat_energies*1000
kevnr_bins = energy_bins*1000
kevnr_edges = energy_edges*1000
plt.hist([kevnr,kevnr], weights=[n_prompt, n_delayed], bins=kevnr_edges,
         stacked=True, histtype='stepfilled', density=density, color=['teal','tan'], label=[r"Prompt $\nu$", r"Delayed $\nu$"])
plt.hist(kevnr,weights=dm_events1,bins=kevnr_edges, color='blue',
         histtype='step', density=density, label=r"DM ($m_\chi = 5$ MeV, $m_V = 15$ MeV)")
plt.hist(kevnr,weights=dm_events2,bins=kevnr_edges, color='crimson',
         histtype='step', density=density, label=r"DM ($m_\chi = 25$ MeV, $m_V = 75$ MeV)")
plt.vlines(kevnr_edges[10], 0, 100000, ls='dashed')
plt.arrow(kevnr_edges[10], 50000, 3, 0, head_width=2000, head_length=3, color='k')
plt.title(r"CCM LAr, $t < 0.4$ $\mu$s", loc="right", fontsize=15)
plt.xlabel(r"$E_r$ [keV]", fontsize=20)
plt.ylabel(r"Events", fontsize=20)
plt.ylim((0,100000))
plt.xlim((kevnr_edges[2], kevnr_edges[-1]))
plt.legend(fontsize=15, framealpha=1.0, loc="upper right")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()
plt.clf()


dm_events1 = GetDMEvents(m_chi=25, m_dp=75, m_med=25, g=1e-4, lifetime=0.001)
dm_events2 = GetDMEvents(m_chi=25, m_dp=75, m_med=25, g=2e-4, lifetime=1)
dm_events3 = GetDMEvents(m_chi=5, m_dp=138, m_med=25, g=2e-4, lifetime=1)

density = True
plt.hist([flat_times,flat_times], weights=[n_prompt, n_delayed], bins=timing_edges,
         stacked=True, histtype='stepfilled', density=density, color=['teal','tan'], label=[r"Prompt $\nu$", r"Delayed $\nu$"])
plt.hist(flat_times,weights=dm_events1,bins=timing_edges, color='blue',
         histtype='step', density=density, label=r"DM ($M_X = 75$ MeV, $\tau < 0.001$ $\mu$s)")
plt.hist(flat_times,weights=dm_events2,bins=timing_edges, color='crimson',
         histtype='step', density=density, label=r"DM ($M_X = 75$ MeV, $\tau = 1$ $\mu$s)")
plt.hist(flat_times,weights=dm_events3,bins=timing_edges, color='darkorange',
         histtype='step', density=density, label=r"DM ($M_X = 138$ MeV, $\tau = 1$ $\mu$s)")
plt.xlabel(r"$t$ [$\mu$s]", fontsize=20)
plt.ylabel(r"a.u.", fontsize=20)
plt.title(r"CCM", loc="right", fontsize=15)
plt.xlim((0,2.0))
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()






