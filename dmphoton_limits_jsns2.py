import sys
import time

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


prompt_pdf = np.genfromtxt('data/ccm/arrivalTimePDF_prompt.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/ccm/arrivalTimePDF_delayed.txt', delimiter=',')
nin_pdf = np.genfromtxt('data/arrivalTimePDF_promptNeutrons.txt', delimiter=',')

# get existing limits
relic = np.genfromtxt('pyCEvNS/data/dark_photon_limits/relic.txt', delimiter=",")
ldmx = np.genfromtxt('pyCEvNS/data/dark_photon_limits/ldmx.txt')
lsnd = np.genfromtxt('pyCEvNS/data/dark_photon_limits/lsnd.csv', delimiter=",")
miniboone = np.genfromtxt('pyCEvNS/data/dark_photon_limits/miniboone.csv', delimiter=",")
na64 = np.genfromtxt('pyCEvNS/data/dark_photon_limits/na64.csv', delimiter=",")

# Convert from GeV mass to MeV
lsnd[:,0] *= 1000
miniboone[:,0] *= 1000
miniboone[:,2] *= 1000
na64[:,0] *= 1000

#ldmx[:,1] *= 2 * (3**4)
na64[:,1] *= 2 * (3**4)
lsnd[:,1] *= 4.5 * (3**4)  # TODO: check this
miniboone[:,1] *= 2 * (3**4)
miniboone[:,3] *= 2 * (3**4)


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

prompt_flux = Flux('prompt')
delayed_flux = Flux('delayed')
brem_photons = np.genfromtxt("data/jsns/brem.txt")  # binned photon spectrum from
Pi0Info = np.genfromtxt("data/jsns/Pi0_Info.txt")
EtaInfo = np.genfromtxt("data/jsns/eta_info.txt")
pion_energy = Pi0Info[:,4] - massofpi0
eta_energy = EtaInfo[:,4] - massofeta
pion_azimuth = np.arccos(Pi0Info[:,3] / np.sqrt(Pi0Info[:,1]**2 + Pi0Info[:,2]**2 + Pi0Info[:,3]**2))  # arccos of the unit z vector gives the azimuth angle
eta_azimuth = np.arccos(EtaInfo[:,3] / np.sqrt(EtaInfo[:,1]**2 + EtaInfo[:,2]**2 + EtaInfo[:,3]**2))  # arccos of the unit z vector gives the azimuth angle
pion_cos = np.cos(np.pi/180 * Pi0Info[:,0])
eta_cos = np.cos(np.pi/180 * EtaInfo[:,0])
pion_dist = np.array([pion_azimuth, pion_cos, pion_energy])
pion_dist = pion_dist.transpose()
eta_dist = np.array([eta_azimuth, eta_cos, eta_energy])
eta_dist = eta_dist.transpose()


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


# Set up energy and timing bins
hi_energy_cut = 1000  # mev
lo_energy_cut = 30  # mev
hi_timing_cut = 0.25
lo_timing_cut = 0.1
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 15)
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.linspace(lo_timing_cut, hi_timing_cut, 10)
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









# Get neutrino spectrum
def GetNeutrinoEvents():
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

    return n_prompt + n_delayed


brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=75, coupling=1, dark_matter_mass=25, life_time=0.0001,
                                detector_distance=24, pot_mu=pot_mu, pot_sigma=pot_sigma, pot_sample=1e5,
                                sampling_size=1000, brem_suppress=True, verbose=False)
pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=75, coupling_quark=1, dark_matter_mass=25, pion_rate=pim_rate,
                                           pot_mu=pot_mu, pot_sigma=pot_sigma, detector_distance=24, life_time=0.0001,
                                           sampling_size=1000)
pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_dist, dark_photon_mass=75, coupling_quark=1, dark_matter_mass=25,
                              pot_mu=pot_mu, pot_sigma=pot_sigma, detector_distance=24, life_time=0.0001)
eta_flux = DMFluxFromPi0Decay(pi0_distribution=eta_dist, dark_photon_mass=75, coupling_quark=1,
                              dark_matter_mass=25, pot_mu=pot_mu, pot_sigma=pot_sigma, detector_distance=24,
                              life_time=0.0001, meson_mass=massofeta)
dm_gen = DmEventsGen(dark_photon_mass=75, dark_matter_mass=25, life_time=0.001, expo=exposure, detector_type='jsns_scintillator')

def GetDMEvents(g, m_med, m_dp, m_chi):
    dm_gen.dp_mass = m_med
    dm_gen.dm_mass = m_chi

    brem_flux.dp_m = m_dp
    brem_flux.dm_m = m_chi
    eta_flux.dp_m = m_dp
    eta_flux.dm_m = m_chi
    pim_flux.dp_m = m_dp
    pim_flux.dm_m = m_chi
    pi0_flux.dp_m = m_dp
    pi0_flux.dm_m = m_chi

    brem_flux.simulate()
    pim_flux.simulate()
    pi0_flux.simulate()
    eta_flux.simulate()

    dm_gen.fx = brem_flux
    brem_events = dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")

    dm_gen.fx = pim_flux
    pim_events = dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")

    dm_gen.fx = pi0_flux
    pi0_events = dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")

    dm_gen.fx = eta_flux
    eta_events = dm_gen.events(m_med, g, energy_edges, timing_edges, channel="electron")

    return brem_events[0] + pim_events[0] + pi0_events[0] + eta_events[0]





def Chi2WithBkg(n_signal, n_bg, n_obs, sigma):
    likelihood = np.zeros(n_obs.shape[0])
    print(np.sum(n_obs))
    print(np.sum(n_signal))
    alpha = np.sum(n_signal*(n_obs-n_bg)/n_obs)/(1/sigma**2+np.sum(n_signal**2/n_obs))
    likelihood += (n_obs-n_bg-(1+alpha)*n_signal)**2/(n_obs)
    return np.sum(likelihood)+(alpha/sigma)**2

def Chi2(sig, bkg):
    likelihood = np.zeros(sig.shape[0])
    likelihood = (sig)**2 / np.sqrt(bkg**2 + 1)
    return np.sum(likelihood)


def Poisson(sig, bkg):
    print("s, b = ", np.sum(sig), np.sum(bkg))
    #bkg = bkg + 1
    likelihood = (np.sum(sig)) / np.sqrt(np.sum(bkg + sig))
    return likelihood





def main():
    # Get Neutrino backgrounds
    n_bg = np.zeros(energy_bins.shape[0]*len(timing_bins))
    n_nu = GetNeutrinoEvents()

    # Set up limits.
    mlist = np.logspace(0, np.log10(500), 100)
    eplist = np.ones_like(mlist)
    tmp = np.logspace(-20, 0, 20)


    use_save = False
    if use_save == True:
        print("using saved data")
        saved_limits = np.genfromtxt("limits/jsns2/dark_photon_limits_jsns_doublemed_withEta_20200520.txt", delimiter=",")
        mlist = saved_limits[:,0]
        eplist = saved_limits[:,1]
    else:
        # Binary search.
        print("Running dark photon limits...")
        outlist = open("limits/jsns2/dark_photon_limits_jsns_doublemed_withEta_20200520.txt", "w")
        for i in range(mlist.shape[0]):
            print("Running m_X = ", mlist[i])
            hi = np.log10(tmp[-1])
            lo = np.log10(tmp[0])
            while hi - lo > 0.05:
                mid = (hi + lo) / 2
                print("------- trying g = ", 10**mid)
                lg = Chi2(GetDMEvents(g=10**mid, m_med=mlist[i], m_dp=75, m_chi=25),n_nu)
                print("lg = ", lg)
                if lg < 6.18:
                  lo = mid
                else:
                  hi = mid
            eplist[i] = 10**mid
            outlist.write(str(mlist[i]))
            outlist.write(",")
            outlist.write(str(10**mid))
            outlist.write("\n")

        outlist.close()
    
if __name__ == "__main__":
    main()