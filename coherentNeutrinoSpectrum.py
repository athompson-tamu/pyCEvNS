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

prompt_pdf = np.genfromtxt('pyCEvNS/data/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('pyCEvNS/data/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')
ac_bon = np.genfromtxt('pyCEvNS/data/data_anticoincidence_beamOn.txt', delimiter=',')
c_bon = np.genfromtxt('pyCEvNS/data/data_coincidence_beamOn.txt', delimiter=',')
ac_boff = np.genfromtxt('pyCEvNS/data/data_anticoincidence_beamOff.txt', delimiter=',')
c_boff = np.genfromtxt('pyCEvNS/data/data_coincidence_beamOff.txt', delimiter=',')
nin_pdf = np.genfromtxt('pyCEvNS/data/arrivalTimePDF_promptNeutrons.txt', delimiter=',')



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

# Convert PE to MeV in the data array
n_meas[:,0] *= 1/pe_per_mev

n_obs = c_bon_meas[:, 2]
flat_energies = n_meas[:,0]
flat_times = n_meas[:,1]
kevnr = flat_energies*1000
kevnr_bins = energy_bins*1000
kevnr_edges = energy_edges*1000


# Histogram observed data
obs_hist, obs_bin_edges = np.histogram(n_meas[:,0], weights=n_meas[:,2], bins=energy_edges)
n_obs_err = np.sqrt(obs_hist + (0.28*obs_hist)**2)  # poisson error + systematics in quadrature

# Get background
n_bg = ac_bon_meas[:, 2]


# Setup prompt and delayed
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_nsi = np.zeros(energy_bins.shape[0] * len(timing_bins))
# Get the theoretical prediction for the neutrino events.
flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
delayed_flux = flux_factory.get('coherent_delayed')
nsi = NSIparameters(0)
nsi.epu = {'ee': 0.0, 'mm': -0.05, 'tt': 0.0,
           'em': 0.0, 'et': 0.0, 'mt': 0.0}
nsi.epd = {'ee': 0.0, 'mm': 0.0, 'tt': 0.0,
           'em': 0.0, 'et': 0.0, 'mt': 0.0}
nu_gen_nsi = NeutrinoNucleusElasticVector(nsi)
nu_gen = NeutrinoNucleusElasticVector(NSIparameters(0))
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
        n_nsi[flat_index] = nu_gen_nsi.events(e_a, e_b, 'mu',prompt_flux, Detector('csi'), exposure) \
                                   * prompt_time(t)* efficiency(pe) \
                                     + (nu_gen_nsi.events(e_a, e_b, 'e', delayed_flux, Detector('csi'), exposure) \
                                       + nu_gen_nsi.events(e_a, e_b, 'mubar', delayed_flux, Detector('csi'), exposure)) \
                                          * delayed_time(t)* efficiency(pe)
        flat_index += 1

n_nu = n_prompt+n_delayed

from matplotlib.cm import get_cmap
cmap = get_cmap('tab20b')
color_prompt= cmap(0.70)
color_delayed = cmap(0.15)
color_brn = cmap(0.55)


density = False
plt.hist([kevnr,kevnr,kevnr], weights=[n_prompt, n_delayed, n_bg],
         bins=kevnr_edges, stacked=True, histtype='stepfilled', density=density,
         color=[color_prompt,color_delayed, color_brn],
         label=[r"Prompt $\nu_\mu$", r"Delayed $\nu_e, \bar{\nu}_\mu$", "AC Beam-On Background"])
plt.hist(kevnr, weights=n_nsi, bins=kevnr_edges, histtype='step', density=density,
         color='m', label=r'NSI, $\epsilon^{u,V}_{\mu\mu} = -0.05$')
plt.errorbar(kevnr_bins,obs_hist,yerr=n_obs_err,label=r"Beam-On data", dash_capstyle="butt",
             capsize=4, fmt='o', ls="none", color='k')
plt.xlabel(r"$E_r$ [keV]", fontsize=20)
plt.ylabel(r"Events", fontsize=20)
plt.title(r"COHERENT CsI, $t < 6.0$ $\mu$s", loc="right", fontsize=15)
plt.ylim((0,125))
plt.xlim((0,42.5))
plt.legend(fontsize=13, framealpha=1, loc='upper right')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()
plt.clf()

