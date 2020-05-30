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


prompt_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')
ac_bon = np.genfromtxt('data/coherent/data_anticoincidence_beamOn.txt', delimiter=',')
c_bon = np.genfromtxt('data/coherent/data_coincidence_beamOn.txt', delimiter=',')
ac_boff = np.genfromtxt('data/coherent/data_anticoincidence_beamOff.txt', delimiter=',')
c_boff = np.genfromtxt('data/coherent/data_coincidence_beamOff.txt', delimiter=',')



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





hi_energy_cut = 52/pe_per_mev
lo_energy_cut = 0/pe_per_mev
hi_timing_cut = 1.75
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 2/pe_per_mev) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

print(timing_bins)

indx = []
# energy cut is 14keV ~ 16pe
for i in range(c_bon.shape[0]):
    if c_bon[i, 0] < lo_energy_cut*pe_per_mev + 1 \
       or c_bon[i, 0] >= hi_energy_cut*pe_per_mev - 1 \
       or c_bon[i, 1] >= hi_timing_cut-0.25:
        indx.append(i)
c_bon_meas = np.delete(c_bon, indx, axis=0)
ac_bon_meas = np.delete(ac_bon, indx, axis=0)



# Get the observed data
n_meas = c_bon_meas.copy()

# Can subtract backgrounds if needed
#for i in range(c_bon_meas.shape[0]):
#    n_meas[i, 2] -= ac_bon_meas[i, 2]

# Convert PE to MeV in the data array
n_meas[:,0] *= 1/pe_per_mev

n_obs = c_bon_meas[:, 2]
flat_energies = n_meas[:,0]
flat_times = n_meas[:,1]

# Histogram observed data
obs_hist, obs_bin_edges = np.histogram(n_meas[:,0], weights=n_meas[:,2], bins=energy_edges)
n_obs_err = np.sqrt(obs_hist + (0.28*obs_hist)**2)  # poisson error + systematics in quadrature

print(obs_hist.shape[0], n_obs_err.shape[0])


# Get background
n_bg = ac_bon_meas[:, 2]


# Setup prompt and delayed
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
# Get the theoretical prediction for the neutrino events.
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
print("nu = ", np.sum(n_nu))
print("Bkg = ", np.sum(ac_bon_meas[:,2]))
print("obs = ", np.sum(n_obs))
print("excess = ", np.sum(n_obs) - np.sum(n_nu) - np.sum(ac_bon_meas[:,2]))
print("sig = ", (np.sum(n_obs) - np.sum(n_nu) - np.sum(ac_bon_meas[:,2]))/(np.sqrt(np.sum(n_nu) + np.sum(ac_bon_meas[:,2]))))

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



m_dp_bf = np.power(10, 0.213973009690852356e1)
m_chi_bf = m_dp_bf/3
eps_bf = np.power(10,-0.708361403497243902e1)
dm_best_fit = GetDMEvents(m_chi=m_chi_bf, m_dp=m_dp_bf, m_med=m_dp_bf, g=eps_bf)


# Plot best-fit
density = False
kevnr = flat_energies*1000
kevnr_bins = energy_bins*1000
kevnr_edges = energy_edges*1000
plt.hist([kevnr,kevnr,kevnr,kevnr], weights=[n_prompt, n_delayed, n_bg, dm_best_fit],
         bins=kevnr_edges, stacked=True, histtype='stepfilled', density=density,
         color=['teal','tan', 'indianred', 'lightsteelblue'],
         label=[r"Prompt $\nu$", r"Delayed $\nu$", "AC Beam-On Background", r"$\chi N \to \chi N$ Best-fit"])
plt.errorbar(kevnr_bins,obs_hist,yerr=n_obs_err,label=r"Beam-On data", color='k', ls='none', marker='o')
plt.vlines(kevnr_edges[8], 0, 100, ls='dashed')
plt.vlines(kevnr_edges[16], 0, 100, ls='dashed')
plt.arrow(kevnr_edges[8], 30, 2, 0, head_width=1, color='k')
plt.arrow(kevnr_edges[16], 30, -2, 0, head_width=1, color='k')
plt.xlabel(r"$E_r$ [MeV]", fontsize=15)
plt.ylabel(r"Events", fontsize=15)
plt.title(r"COHERENT CsI, $t < 1.5$ $\mu$s", loc="right", fontsize=15)
plt.ylim((0,58))
plt.xlim((0,42.5))
plt.legend(fontsize=13, framealpha=1, loc='upper right')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()
plt.clf()

dm_events1 = GetDMEvents(m_chi=1, m_dp=75, m_med=25, g=0.75e-7)
dm_events2 = GetDMEvents(m_chi=20, m_dp=120, m_med=25, g=0.75e-7)

# Plot Dark Matter against Neutrino Spectrum
density = False
plt.hist([flat_energies,flat_energies], weights=[n_prompt, n_delayed], bins=energy_edges*pe_per_mev,
         stacked=True, histtype='stepfilled', density=density, color=['teal','tan'], label=["Prompt", "Delayed"])
plt.hist(flat_energies*pe_per_mev,weights=dm_events1,bins=energy_edges*pe_per_mev,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 1$ MeV)")
plt.hist(flat_energies*pe_per_mev,weights=dm_events2,bins=energy_edges*pe_per_mev,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 20$ MeV)")
plt.xlabel(r"$E_r$ [MeV]")
plt.ylabel(r"a.u.")
plt.legend()
plt.show()
plt.clf()


plt.hist([flat_times,flat_times], weights=[n_prompt, n_delayed], bins=timing_edges,
         stacked=True, histtype='stepfilled', density=density, color=['teal','tan'], label=["Prompt", "Delayed"])
plt.hist(flat_times,weights=dm_events1,bins=timing_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 1$ MeV)")
plt.hist(flat_times,weights=dm_events2,bins=timing_edges,
         histtype='step', density=density, label=r"DM ($m_{\chi} = 20$ MeV)")
plt.xlabel(r"$t$ [$\mu$s]")
plt.ylabel(r"a.u.")
plt.legend()
plt.show()






