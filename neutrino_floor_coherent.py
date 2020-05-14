import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *
from pyCEvNS.helper import *

import math
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from mpmath import mpmathify

from scipy.interpolate import UnivariateSpline

from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

prompt_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')
nin_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_promptNeutrons.txt', delimiter=',')
ac_bon = np.genfromtxt('data/coherent/data_anticoincidence_beamOn.txt', delimiter=',')
c_bon = np.genfromtxt('data/coherent/data_coincidence_beamOn.txt', delimiter=',')
ac_boff = np.genfromtxt('data/coherent/data_anticoincidence_beamOff.txt', delimiter=',')
c_boff = np.genfromtxt('data/coherent/data_coincidence_beamOff.txt', delimiter=',')
nin = np.genfromtxt('data/coherent/promptPDF.txt', delimiter=',')

# CONSTANTS
pe_per_mev = 0.0878 * 13.348 * 1000
exposure = 4466
pim_rate = 0.0457


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


# Set up energy and timing bins
hi_energy_cut = 49/pe_per_mev  #0.026 mev
lo_energy_cut = 14/pe_per_mev  # 0.014mev
hi_timing_cut = 1.75
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 2/pe_per_mev) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

indx = []
# energy cut is 14keV ~ 16pe
for i in range(c_bon.shape[0]):
    if c_bon[i, 0] < 15 or c_bon[i, 0] >= 48 or c_bon[i, 1] >= 1.5:
        indx.append(i)
c_bon_meas = np.delete(c_bon, indx, axis=0)
ac_bon_meas = np.delete(ac_bon, indx, axis=0)

# Set the observed data = coincidence - anticoincidence (beam on)
n_meas = c_bon_meas.copy()

# Convert PE to MeV in the data array
n_meas[:,0] *= 1/pe_per_mev

# Set up background, prompt and delayed arrays.
n_bg = ac_bon_meas[:, 2]
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_nin = np.zeros(energy_bins.shape[0] * len(timing_bins))


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
        n_nin[flat_index] = (efficiency(pe-0.5) * nin[int(pe-1), 1]+ efficiency(pe+0.5) * nin[int(pe), 1]) * nin_time(t) * 365000/4466
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
dm_gen = DmEventsGen(dark_photon_mass=75, dark_matter_mass=25,
                         life_time=0.001, expo=exposure, detector_type='csi')
brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=75, coupling=1,
                            dark_matter_mass=25, pot_sample=1e5,
                            sampling_size=1000, life_time=0.0001,
                            brem_suppress=True, verbose=False)
pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=75, coupling_quark=1,
                                       dark_matter_mass=25, pion_rate=pim_rate,
                                       life_time=0.0001)
pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=75, coupling_quark=1,
                              dark_matter_mass=25, life_time=0.0001)
brem_flux.simulate()
pim_flux.simulate()
pi0_flux.simulate()

def GetDMEvents(g, m_dp, m_chi, m_med):  
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


m_dp = 5
m_chi = m_dp / 3
n_dm  = GetDMEvents(1, m_dp, m_chi, m_dp)


def SimpleChi2(n_signal, n_obs, exp):
    sig = n_signal * exp
    bkg = n_obs * exp
    likelihood = np.zeros(sig.shape[0])
    likelihood = (sig)**2 / np.sqrt(bkg**2 + 1)
    return np.sum(likelihood)

def Chi2(n_signal, n_obs, sigma, exp):
    n_signal = n_signal * exp
    n_obs = n_obs * exp
    likelihood = np.zeros(n_obs.shape[0])
    alpha = np.sum(n_signal*(n_obs)/n_obs)/(1/sigma**2+np.sum(n_signal**2/n_obs))
    likelihood += (n_obs-(1+alpha)*n_signal)**2/(n_obs)
    return np.sum(likelihood)+(alpha/sigma)**2

def poisson(k, l):
    """
    poisson distribution
    :param k: observed number
    :param l: mean number
    :return: probability
    """
    return mp.exp(k*mp.log(l) - mp.loggamma(k+1) - l)

def gaussian(x, mu, sigma):
    """
    gaussian distribution
    :param x: number
    :param mu: mean
    :param sigma: standard deviation
    :return: probability density
    """
    return mp.exp(-(x-mu)**2/(2*sigma**2)) / mp.sqrt(2*np.pi*sigma**2)

def lgl(n_sig, n_null, n_obs, sigma, exp):
    n_sig = n_sig * exp  # dm rate
    n_obs = n_obs * exp  # observed rate
    n_null = n_null * exp  # neutrino rate
    sigma = mpmathify(sigma)
    like_sig = np.zeros(n_obs.shape[0])
    like_null = np.zeros(n_obs.shape[0])
    for i in range(n_obs.shape[0]):
        obs_ = mpmathify(n_obs[i])
        null_ = mpmathify(n_null[i])
        sig_ = mpmathify(n_sig[i])
        bg_ = mpmathify(n_bg[i])
        like_sig[i] += mp.log(mp.quad(lambda a: poisson(obs_, (1 + a) * (sig_ + null_) + bg_) * gaussian(a, 0, sigma), [-3 * sigma, 3 * sigma]))
        like_null[i] += mp.log(mp.quad(lambda a: poisson(obs_, (1 + a) * null_ + bg_) * gaussian(a, 0, sigma), [-3 * sigma, 3 * sigma]))
    
    prod_sig = np.sum(like_sig)
    prod_null = np.sum(like_null)
    q = -2*((prod_null) - (prod_sig)) # if prod_null/prod_sig > 0 else return
    if math.isnan(q):
        q = np.inf
    return mp.sqrt(np.abs(q))


def BinarySearch(sigma, exposures, save_file, use_save=False):
    eplist = np.ones_like(exposures)
    tmp = np.logspace(-40, 0, 20)

    if use_save == True:
        print("using saved data")
        saved_limits = np.genfromtxt(save_file, delimiter=",")
        exposures = saved_limits[:,0]
        eplist = saved_limits[:,1]
    else:
        # Binary search.
        print("Running dark photon limits...")
        outlist = open(save_file, "w")
        for i in range(exposures.shape[0]):
            print("Running Exp = ", exposures[i])
            hi = np.log10(tmp[-1])
            lo = np.log10(tmp[0])
            while hi - lo > 0.005:
                mid = (hi + lo) / 2
                print("------- trying g = ", 10**mid)
                n_sig = (10**mid)**2 * n_dm
                lg = lgl(n_sig=n_sig, n_null=n_nu+n_bg, n_obs=n_nu+n_bg, sigma=sigma, exp=exposures[i])
                print("lg = ", lg)
                if lg < 6.18:
                    lo = mid
                else:
                    hi = mid
            eplist[i] = 10**mid
            outlist.write(str(exposures[i]))
            outlist.write(",")
            outlist.write(str(10**mid))
            outlist.write("\n")

        outlist.close()
    return exposures, eplist




def main():
    exposure_list = np.logspace(-1,5,10)
    limits_sigma_5 = BinarySearch(0.05, exposure_list, "limits/neutrino_floor/exposure_limits_coherent_5%_mdp5.txt", use_save=True)
    limits_sigma_10 = BinarySearch(0.1, exposure_list, "limits/neutrino_floor/exposure_limits_coherent_10%_mdp5.txt", use_save=True)
    limits_sigma_28 = BinarySearch(0.28, exposure_list, "limits/neutrino_floor/exposure_limits_coherent_28%_mdp5.txt", use_save=True)
    
    
    # Plot the derived limits
    plt.plot(limits_sigma_5[0], limits_sigma_5[1], label="5\%",
             linewidth=2, color="blue")
    plt.plot(limits_sigma_10[0], limits_sigma_10[1], label="10\%",
             linewidth=2, ls='dashed', color="blue")
    plt.plot(limits_sigma_28[0], limits_sigma_28[1], label="28\%",
             linewidth=2, ls='dotted', color="blue")
    plt.legend(loc="upper left", ncol=2, framealpha=1.0)

    plt.xscale("Log")
    plt.yscale("Log")
    #plt.xlim((10, 5e2))
    plt.ylim((1e-9,1e-6))
    plt.title(r"COHERENT CsI, $m_X = m_V = 3m_\chi = 5$ MeV, $\mathcal{E} = \beta \mathcal{E}_{CsI}$", loc="right")
    plt.ylabel(r"$(\epsilon^\chi)^2$", fontsize=13)
    plt.xlabel(r"$\beta$", fontsize=13)
    plt.tight_layout()
    plt.show()
    
    plt.clf()
    
    # Plot evolution of the floor with m_X
    m_x = [5, 15, 45, 75, 105, 135, 165]
    floor = []





if __name__ == "__main__":
    main()
