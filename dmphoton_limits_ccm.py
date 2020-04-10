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
hi_energy_cut = 0.08  # mev
lo_energy_cut = 0.02  # mev
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
dm_gen = DmEventsGen(dark_photon_mass=75, dark_matter_mass=25,
                         life_time=0.001, expo=exposure, detector_type='ar')
brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=75, coupling=1,
                            dark_matter_mass=25, pot_sample=1e5, pot_mu=pot_mu,
                            pot_sigma=pot_sigma, sampling_size=1000, life_time=0.0001,
                            detector_distance=dist, brem_suppress=True, verbose=False)
pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=75, coupling_quark=1,
                                       dark_matter_mass=25, pion_rate=pim_rate,
                                       life_time=0.0001, pot_mu=pot_mu, pot_sigma=pot_sigma,
                                       detector_distance=dist)
pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=75, coupling_quark=1,
                              dark_matter_mass=25, pot_mu=pot_mu, pot_sigma=pot_sigma,
                              life_time=0.0001, detector_distance=dist)
def GetDMEvents(g, m_dp, m_chi, m_med):
    dm_gen.dp_mass = m_med
    dm_gen.dm_mass = m_chi
    
    brem_flux.dp_m = m_dp
    brem_flux.dm_m = m_chi
    pim_flux.dp_m = m_dp
    pim_flux.dm_m = m_chi
    pi0_flux.dp_m = m_dp
    pi0_flux.dm_m = m_chi
    
    brem_flux.simulate()
    pim_flux.simulate()
    pi0_flux.simulate()
    
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



def SimpleChi2(sig, bkg):
    likelihood = np.zeros(sig.shape[0])
    likelihood = (sig)**2 / np.sqrt(bkg**2 + 1)
    return np.sum(likelihood)



def main():
    mlist = np.logspace(0, np.log10(500), 7)
    eplist = np.ones_like(mlist)
    tmp = np.logspace(-20, 0, 20)

    use_save = False
    if use_save == True:
        print("using saved data")
        saved_limits = np.genfromtxt("limits/ccm/dark_photon_limits_doublemed_ccm_loose.txt", delimiter=",")
        mlist = saved_limits[:,0]
        eplist = saved_limits[:,1]
    else:
        # Binary search.
        print("Running dark photon limits...")
        outlist = open("limits/ccm/dark_photon_limits_doublemed_ccm_loose.txt", "w")
        for i in range(mlist.shape[0]):
            print("Running m_X = ", mlist[i])
            hi = np.log10(tmp[-1])
            lo = np.log10(tmp[0])
            while hi - lo > 0.05:
                mid = (hi + lo) / 2
                print("------- trying g = ", 10**mid)
                lg = SimpleChi2(GetDMEvents(10**mid, 75, 25, mlist[i]), n_nu)
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
    lsnd[:,0] *= 1000
    miniboone[:,0] *= 1000
    miniboone[:,2] *= 1000
    na64[:,0] *= 1000

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
    plt.plot(relic[:,0], relic[:,1], color="k", linewidth=2)

    # Plot the derived limits
    plt.plot(mlist, eplist, label="CCM LAr (3 years, 10 ton)", linewidth=2, color="blue")
    plt.title(r"$t<0.1$ $\mu$s, $E_r > 50$ keVnr, $m_X = m_V$, $m_\chi = 5$ MeV", loc='right')
    plt.legend(loc="upper left", ncol=2, framealpha=1.0)

    plt.xscale("Log")
    plt.yscale("Log")
    plt.xlim((10, 5e2))
    plt.ylim((1e-11,3e-5))
    plt.ylabel(r"$(\epsilon^\chi)^2$", fontsize=13)
    plt.xlabel(r"$M_{A^\prime}$ [MeV]", fontsize=13)
    plt.tight_layout()
    #plt.savefig("plots/dark_photon/dark_photon_limits_ccm.png")
    plt.show()





if __name__ == "__main__":
    main()
