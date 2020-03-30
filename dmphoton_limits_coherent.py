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


# Set up neutrino PDFs and efficiencies.
prompt_pdf = np.genfromtxt('data/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')
nin_pdf = np.genfromtxt('data/arrivalTimePDF_promptNeutrons.txt', delimiter=',')

# Get Neutrino Events
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

def ffs(q):
    r = 5.5 * (10 ** -15) / meter_by_mev
    s = 0.9 * (10 ** -15) / meter_by_mev
    r0 = np.sqrt(5/3 * (r ** 2) - 5 * (s ** 2))
    return (3 * spherical_jn(1, q * r0) / (q * r0) * np.exp((-(q * s) ** 2) / 2)) ** 2
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

def efficiency_lar(pe):
    a = 0.9
    k = 0.4942
    x0 = 10.8507
    f = a / (1 + np.exp(-k * (pe - x0)))
    if pe < 5:
        return 0
    if pe < 6:
        return 0.5 * f
    return f


# Define constants
pe_per_mev = 0.0878 * 13.348 * 1000
exp_csi = 4466
exp_ar = 1.5*274* 24  # tweaked to data
pim_rate = 0.0457



# Get Neutrino events (CsI)
# Set up energy and timing bins
hi_energy_cut = 0.026  # mev
lo_energy_cut = 0.014  # mev
hi_timing_cut = 1.5
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 0.0017) # energy resolution ~2keV
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
        n_prompt[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('csi'), exp_csi) \
                               * prompt_time(t) * efficiency(pe)
        n_delayed[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('csi'), exp_csi)
                                 + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, Detector('csi'), exp_csi)) \
                                * delayed_time(t) * efficiency(pe)
        flat_index += 1

n_nu_csi = n_prompt+n_delayed




# Get Neutrino events (LAr)
# Set up energy and timing bins
hi_energy_cut = 0.120  # mev
lo_energy_cut = 0.02 # mev
hi_timing_cut = 1.5
lo_timing_cut = 0.0
energy_edges_lar = np.arange(lo_energy_cut, hi_energy_cut, 0.005) # energy resolution ~10keVnr
energy_bins_lar = (energy_edges_lar[:-1] + energy_edges_lar[1:]) / 2
timing_edges_lar = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins_lar = (timing_edges_lar[:-1] + timing_edges_lar[1:]) / 2
flat_index = 0

n_meas_lar = np.zeros((energy_bins_lar.shape[0] * len(timing_bins_lar), 2))
n_prompt_lar = np.zeros(energy_bins_lar.shape[0] * len(timing_bins_lar))
n_delayed_lar = np.zeros(energy_bins_lar.shape[0] * len(timing_bins_lar))
n_bg_lar = np.zeros(energy_bins_lar.shape[0] * len(timing_bins_lar))

for i in range(0, energy_bins_lar.shape[0]):
  for j in range(0, timing_bins_lar.shape[0]):
    n_meas_lar[flat_index, 0] = energy_bins_lar[i]
    n_meas_lar[flat_index, 1] = timing_bins_lar[j]
    flat_index += 1

#for i in range(0, n_meas.shape[0]):
#  n_bg_lar[i] = (226 / (energy_bins.shape[0]) * prompt_time(n_meas[i,1])) + (10 / (energy_bins.shape[0]) * delayed_time(n_meas[i,1]))

flat_energies_lar = n_meas[:,0]
flat_times_lar = n_meas[:,1]


flat_index = 0
for i in range(0, energy_bins_lar.shape[0]):
    for j in range(0, timing_bins_lar.shape[0]):
        e_a = energy_edges_lar[i]
        e_b = energy_edges_lar[i + 1]
        pe = energy_bins_lar[i] * pe_per_mev
        t = timing_bins_lar[j]
        n_prompt_lar[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('ar'), exp_ar) \
                               * prompt_time(t) * efficiency_lar(pe) * 0.46 # ad hoc factor for greater dist
        n_delayed_lar[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('ar'), exp_ar)
                                 + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, Detector('ar'), exp_ar)) \
                                * delayed_time(t) * efficiency_lar(pe) * 0.46 # ad hoc factor for greater dist
        flat_index += 1

n_nu_lar = n_prompt_lar+n_delayed_lar


# Combine the neutrino events
n_nu = n_nu_csi
#n_nu = np.append(n_nu_csi), n_nu_lar*0.46)  # effective reduction of flux due to distance
#n_bg = np.append(n_bg), n_bg_lar)




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
                         life_time=0.001, expo=exp_csi, detector_type='csi')
    dm_gen_lar = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi,
                         life_time=0.001, expo=exp_csi, detector_type='ar')
    brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=m_dp, coupling=1,
                                dark_matter_mass=m_chi, pot_sample=1e5,
                                sampling_size=1000, life_time=0.0001,
                                detector_distance=19.3, brem_suppress=True, verbose=False)
    pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_dp, coupling_quark=1,
                                           dark_matter_mass=m_chi,
                                           pion_rate=pim_rate, detector_distance=19.3,
                                           life_time=0.0001)
    pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux,
                                  dark_photon_mass=m_dp, coupling_quark=1,
                                  dark_matter_mass=m_chi, detector_distance=19.3,
                                  life_time=0.0001)
    
    dm_gen.fx = brem_flux
    dm_gen_lar.fx = brem_flux
    brem_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                                channel="nucleus")[0]
    brem_events = np.append(brem_events,
                            dm_gen_lar.events(m_med, g, energy_edges_lar,
                                              timing_edges_lar, channel="nucleus")[0])

    dm_gen.fx = pim_flux
    dm_gen_lar.fx = pim_flux
    pim_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")[0]
    pim_events = np.append(pim_events,
                           dm_gen_lar.events(m_med, g, energy_edges_lar,
                                             timing_edges_lar, channel="nucleus")[0])

    dm_gen.fx = pi0_flux
    dm_gen_lar.fx = pi0_flux
    pi0_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")[0]
    pi0_events = np.append(pi0_events,
                           dm_gen_lar.events(m_med, g, energy_edges_lar,
                                             timing_edges_lar, channel="nucleus")[0])

    return brem_events + pim_events + pi0_events


brem_flux_2med = DMFluxIsoPhoton(brem_photons, dark_photon_mass=75, coupling=1,
                                dark_matter_mass=m_chi, pot_sample=1e5,
                                sampling_size=1000, life_time=0.0001,
                                detector_distance=19.3, brem_suppress=True, verbose=False)
pim_flux_2med = DMFluxFromPiMinusAbsorption(dark_photon_mass=75, coupling_quark=1,
                                        dark_matter_mass=m_chi,
                                        pion_rate=pim_rate, detector_distance=19.3,
                                        life_time=0.0001)
pi0_flux_2med = DMFluxFromPi0Decay(pi0_distribution=pion_flux,
                              dark_photon_mass=75, coupling_quark=1,
                              dark_matter_mass=m_chi, detector_distance=19.3,
                              life_time=0.0001)
def GetDMEventsDoubleMediator(g, m_med):
    dm_gen = DmEventsGen(dark_photon_mass=75, dark_matter_mass=m_med/3,
                         life_time=0.001, expo=exp_csi, detector_type='csi')
    dm_gen_lar = DmEventsGen(dark_photon_mass=75, dark_matter_mass=m_med/3,
                             life_time=0.001, expo=exp_csi, detector_type='ar')
    
    dm_gen.fx = brem_flux_2med
    dm_gen_lar.fx = brem_flux_2med
    brem_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                                channel="nucleus")[0]
    brem_events = np.append(brem_events,
                            dm_gen_lar.events(m_med, g, energy_edges_lar,
                                              timing_edges_lar, channel="nucleus")[0])

    dm_gen.fx = pim_flux_2med
    dm_gen_lar.fx = pim_flux_2med
    pim_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")[0]
    pim_events = np.append(pim_events,
                           dm_gen_lar.events(m_med, g, energy_edges_lar,
                                             timing_edges_lar, channel="nucleus")[0])

    dm_gen.fx = pi0_flux_2med
    dm_gen_lar.fx = pi0_flux_2med
    pi0_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")[0]
    pi0_events = np.append(pi0_events,
                           dm_gen_lar.events(m_med, g, energy_edges_lar,
                                             timing_edges_lar, channel="nucleus")[0])

    return brem_events + pim_events + pi0_events


# Statistics
def Chi2(n_signal, n_bg, n_obs, sigma):
    likelihood = np.zeros(n_obs.shape[0])
    alpha = np.sum(n_signal*(n_obs-n_bg)/n_obs)/(1/sigma**2+np.sum(n_signal**2/n_obs))
    likelihood += (n_obs-n_bg-(1+alpha)*n_signal)**2/(n_obs)
    return np.sum(likelihood)+(alpha/sigma)**2

def SimpleChi2(sig, bkg):
    likelihood = np.zeros(sig.shape[0])
    likelihood = (sig)**2 / np.sqrt(bkg**2 + 1)
    return np.sum(likelihood)

mlist = np.logspace(1, np.log10(400), 5)
eplist = np.ones_like(mlist)
tmp = np.logspace(-20, 0, 20)

use_save = False
if use_save == True:
    saved_limits = np.genfromtxt("limits/coherent/dark_photon_limits_coh_doublemed_csi-lar.txt",
                                  delimiter=",")
    mlist = saved_limits[:,0]
    eplist = saved_limits[:,1]
else:
    # Binary search.
    outlist = open("limits/coherent/dark_photon_limits_coh_doublemed_csi-lar.txt", "w")
    for i in range(mlist.shape[0]):
        print("Running m_X = ", mlist[i])
        hi = np.log10(tmp[-1])
        lo = np.log10(tmp[0])
        while hi - lo > 0.05:
            mid = (hi + lo) / 2
            print("------- trying g = ", 10**mid)
            lg = SimpleChi2(GetDMEvents(10**mid, mlist[i], 4.9, mlist[i]),
                            n_bg + n_nu)
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





# get existing limits
relic = np.genfromtxt('pyCEvNS/data/dark_photon_limits/relic.txt', delimiter=",")
ldmx = np.genfromtxt('pyCEvNS/data/dark_photon_limits/ldmx.txt')
lsnd = np.genfromtxt('pyCEvNS/data/dark_photon_limits/lsnd.csv', delimiter=",")
miniboone = np.genfromtxt('pyCEvNS/data/dark_photon_limits/miniboone.csv', delimiter=",")
na64 = np.genfromtxt('pyCEvNS/data/dark_photon_limits/na64.csv', delimiter=",")

# Convert from GeV mass to MeV
#relic[:,0] *= 3000
lsnd[:,0] *= 1000
miniboone[:,0] *= 1000
miniboone[:,2] *= 1000
na64[:,0] *= 1000

#relic[:,1] *= 2 * (3**4)
#ldmx[:,1] *= 2 * (3**4)
na64[:,1] *= 2 * (3**4)
lsnd[:,1] *= 4.5 * (3**4)  # TODO: check this
miniboone[:,1] *= 2 * (3**4)
miniboone[:,3] *= 2 * (3**4)

# Plot the existing limits.
plt.fill_between(miniboone[:,0], miniboone[:,1], y2=1, color="royalblue", alpha=0.3, label='MiniBooNE \n (Nucleus)')
plt.fill_between(na64[:,0], na64[:,1], y2=1, color="maroon", alpha=0.3, label='NA64')
plt.fill_between(miniboone[:,2], miniboone[:,3], y2=1, color="orchid", alpha=0.3, label='MiniBooNE \n (Electron)')
plt.fill_between(lsnd[:,0], lsnd[:,1], y2=1, color="crimson", alpha=0.3, label='LSND')

plt.plot(miniboone[:,0], miniboone[:,1], color="royalblue", ls="dashed")
plt.plot(na64[:,0], na64[:,1], color="maroon", ls="dashed")
plt.plot(miniboone[:,2], miniboone[:,3], color="orchid", ls="dashed")
plt.plot(lsnd[:,0], lsnd[:,1], color="crimson", ls="dashed")

# Plot relic density limit
plt.plot(relic[:,0], relic[:,1], label="Relic Density", color="k", linewidth=2)

# Plot the derived limits.
plt.plot(mlist, eplist, label="COHERENT CsI + LAr", linewidth=2, color="blue")
plt.title(r"$t<1.5$ $\mu$s, $m_X = m_V$, $m_\chi = 5$ MeV", loc='right')
plt.legend(loc="upper left", ncol=2, framealpha=1.0)

plt.xscale("Log")
plt.yscale("Log")
plt.xlim((10, 5e2))
plt.ylim((1e-11,3e-5))
plt.ylabel(r"$(\epsilon^\chi)^2$", fontsize=13)
plt.xlabel(r"$M_{A^\prime}$ [MeV]", fontsize=13)
plt.tight_layout()
plt.savefig("paper_plots/coherent_limits_singlemed_20keVEcut.png")
plt.show()
