import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


from matplotlib.cm import get_cmap

from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


prompt_pdf = np.genfromtxt('pyCEvNS/data/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('pyCEvNS/data/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')


# CONSTANTS
exposure = 24*1.5*274*0.51  # factored in distance difference from CsI and exposure scale


# Prompt, delayed PDFs and efficiency functions for COHERENT CsI.
def prompt_time(t):
    if t < 0 or t > 11.75:
        return 0
    else:
        return prompt_pdf[int((t-0.25)/0.5), 1]


def delayed_time(t):
    if t < 0 or t > 11.75:
        return 0
    else:
        return delayed_pdf[int((t-0.25)/0.5), 1]


# Efficiency as a function of E_r [MeV]
def efficiency(er):
    a = 0.94617149
    k = 0.2231348
    x0 = 14.98477134
    f = a / (1 + np.exp(-k * (1000*er - x0)))
    return f




# Grab CENNS-10 data
cenns10_obs = np.genfromtxt("cenns10/cenns10_analysisA_data.txt")
cenns10_brn = np.genfromtxt("cenns10/cenns10_analysisA_BRN.txt")
cenns10_error = np.genfromtxt("cenns10/cenns10_errors.txt")


# Set up energy and timing bins
hi_energy_cut = 0.346  # 0.346 MeV = 346 keV
lo_energy_cut = 0.01  # 0.01 MeV = 10 keV
hi_timing_cut = 5.0
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut + 0.028, 0.028) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_bg = np.ones(energy_bins.shape[0] * len(timing_bins))
n_nsi = np.zeros(energy_bins.shape[0] * len(timing_bins))

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
nsi.epu = {'ee': 0.0, 'mm': -0.05, 'tt': 0.0,
           'em': 0.0, 'et': 0.0, 'mt': 0.0}
nsi.epd = {'ee': 0.0, 'mm': 0.0, 'tt': 0.0,
           'em': 0.0, 'et': 0.0, 'mt': 0.0}
nu_gen_nsi = NeutrinoNucleusElasticVector(nsi)
nu_gen = NeutrinoNucleusElasticVector(NSIparameters(0))
det_ar = Detector('ar', efficiency=efficiency)
flat_index = 0
for i in range(0, energy_bins.shape[0]):
    for j in range(0, timing_bins.shape[0]):
        e_a = energy_edges[i]
        e_b = energy_edges[i + 1]
        er = energy_bins[i]
        t = timing_bins[j]
        n_prompt[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, det_ar, exposure) \
                               * prompt_time(t) # ad hoc factor for greater dist
        n_delayed[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, det_ar, exposure)
                                 + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, det_ar, exposure)) \
                                * delayed_time(t) # ad hoc factor for greater dist
        n_nsi[flat_index] = nu_gen_nsi.events(e_a, e_b, 'mu',prompt_flux, det_ar, exposure) \
                                   * prompt_time(t) \
                                     + (nu_gen_nsi.events(e_a, e_b, 'e', delayed_flux, det_ar, exposure) \
                                       + nu_gen_nsi.events(e_a, e_b, 'mubar', delayed_flux, det_ar, exposure)) \
                                          * delayed_time(t)
        n_bg[flat_index] = cenns10_brn[i,1]/timing_bins.shape[0]
        flat_index += 1

n_nu = n_prompt+n_delayed



# Plot Dark Matter against Neutrino Spectrum
cmap = get_cmap('tab20b')
color_prompt= cmap(0.70)
color_delayed = cmap(0.15)
color_brn = cmap(0.55)



density = False
kevnr = flat_energies*1000
kevnr_bins = energy_bins*1000
kevnr_edges = energy_edges*1000
plt.errorbar(cenns10_obs[:,0], cenns10_obs[:,1], yerr=cenns10_error[:,1], color="k",
             dash_capstyle="butt", capsize=4, fmt='o', ls="none", label="Analysis A Data")
plt.hist([kevnr,kevnr,kevnr], weights=[n_prompt, n_delayed, n_bg],
         bins=kevnr_edges, stacked=True, histtype='stepfilled', density=density,
         color=[color_prompt, color_delayed, color_brn], label=[r"Prompt $\nu_\mu$", r"Delayed $\nu_e, \bar{\nu}_\mu$", "BRN"])
plt.hist(kevnr, weights=n_nsi, bins=kevnr_edges, histtype='step', density=density,
         color='m', label=r'NSI, $\epsilon^{u,V}_{\mu\mu} = -0.05$')

plt.title(r"CENNS-10 LAr, $t<5.0$ $\mu$s", loc="right", fontsize=15)
plt.xlabel(r"$E_r$ [keV]", fontsize=20)
plt.ylabel(r"Events", fontsize=20)
plt.xlim(( kevnr_edges[0], kevnr_edges[-1]))
plt.ylim((0,250))
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()






