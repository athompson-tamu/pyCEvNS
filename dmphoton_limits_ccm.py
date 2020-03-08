import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

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



# Set up the timing cut.
timing_cut = 0.0 # 100 ns


def prompt_time(t):
    if t < timing_cut or t > 11.75:
        return 0
    else:
        return prompt_pdf[int((t-0.25)/0.5), 1]


def delayed_time(t):
    if t < timing_cut or t > 11.75:
        return 0
    else:
        return delayed_pdf[int((t-0.25)/0.5), 1]

def nin_time(t):
    if t < timing_cut or t > 11.75:
        return 0
    else:
        return nin_pdf[int((t-0.25)/0.5), 1]


prompt_flux = Flux('prompt')
delayed_flux = Flux('delayed')

pe_per_mev = 0.0878 * 13.348 * 1000
exposure = 3 * 365 * 7000  #4466



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




def get_energy_bins(e_a, e_b):
    return np.arange(e_a, e_b, step=2/pe_per_mev)




# energy cut
hi_energy_cut = 0.1 # mev
lo_energy_cut = 0.05  # mev
energy_edges = get_energy_bins(0.050,0.100)
energy_bins = (energy_edges[:-1] + energy_edges[1:])/2
timing_bins = [0.015, 0.045, 0.075]

n_meas = np.zeros((energy_bins.shape[0]*len(timing_bins), 2))
n_prompt = np.zeros(energy_bins.shape[0]*len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0]*len(timing_bins))
n_nin = np.zeros(energy_bins.shape[0]*len(timing_bins))
n_bg = 405*exposure/exposure/(12*12)*np.ones(energy_bins.shape[0]*len(timing_bins))
flat_index = 0
for i in range(0,energy_bins.shape[0]):
    for j in range(len(timing_bins)):
        n_meas[flat_index, 0] = energy_bins[i]*pe_per_mev
        n_meas[flat_index, 1] = timing_bins[j]
        e_a = energy_edges[i]
        e_b = energy_edges[i+1]
        n_prompt[i] = efficiency(energy_bins[i]*pe_per_mev) * \
                      binned_events_nucleus(e_a, e_b, exposure, Detector('ar'), prompt_flux, NSIparameters(),
                                            flavor='mu', rn=5.5) * prompt_time(timing_bins[j])
        n_delayed[i] = efficiency(energy_bins[i]*pe_per_mev) * \
                       (binned_events_nucleus(e_a, e_b, exposure, Detector('ar'), delayed_flux, NSIparameters(),
                                              flavor='e', rn=5.5)
                        + binned_events_nucleus(e_a, e_b, exposure, Detector('ar'), delayed_flux, NSIparameters(),
                                                flavor='mubar', rn=5.5)) * delayed_time(timing_bins[j])
        # n_nin[i] = (efficiency(pe-0.5) * nin[int(pe-1), 1]+ efficiency(pe+0.5) * nin[int(pe), 1]) * nin_time(t) * 365000/exposure
        flat_index += 1
    
n_nu = n_prompt+n_delayed

photon_flux = np.genfromtxt("data/ccm_800mev_photon_spectra_1e5_POT.txt")  # binned photon spectrum from
# Rebecca
dm_gen = DmEventsGen(dark_photon_mass=75, dark_matter_mass=25, expo=exposure, life_time=1,
                     detector_distance=20, detector_type='ar')
dm_flux = DMFluxIsoPhoton(photon_flux, dark_photon_mass=75, coupling=1, dark_matter_mass=25,
                          detector_distance=20, pot_sample=1e5, sampling_size=2000,
                          verbose=False)
dm_gen.fx = dm_flux




def EventsGen(cube):
    med_mass = cube[0]
    eps = cube[1]
    return dm_gen.events(med_mass, eps, n_meas)  # mediator mass, epsilon


# mediator = dark photon scenario
dm_gen_2 = DmEventsGen(dark_photon_mass=1, dark_matter_mass=0, life_time=1, expo=exposure)
def EventsGen2(cube):
    dp_mass = cube[0]
    if cube[0] > 450:
        dp_mass = 450

    dm_gen_2.dm_mass = dp_mass / 3
    flux = DMFluxIsoPhoton(photon_flux, dark_photon_mass=dp_mass, coupling=1, dark_matter_mass=dp_mass/3,
                           detector_distance=20, pot_mu=0.145, pot_sigma=0.1, pot_sample=1e5, sampling_size=1000,
                           verbose=False)
    dm_gen_2.fx = flux
    return dm_gen_2.events(cube[0], cube[1], n_meas)  # mediator mass, epsilon


def Chi2(n_signal, n_bg, n_obs, sigma):
    likelihood = np.zeros(n_obs.shape[0])
    alpha = np.sum(n_signal*(n_obs-n_bg)/n_obs)/(1/sigma**2+np.sum(n_signal**2/n_obs))
    likelihood += (n_obs-n_bg-(1+alpha)*n_signal)**2/(n_obs)
    return np.sum(likelihood)+(alpha/sigma)**2




def main():
    mlist = np.logspace(0, 3, 50)
    eplist = np.ones_like(mlist)
    tmp = np.logspace(-20, 0, 20)
    saved_limits_coherent = np.genfromtxt("limits/dark_photon/dark_photon_limits_mediatorIsDP.txt", delimiter=",")
    mlist_coherent = saved_limits_coherent[:, 0]
    eplist_coherent = saved_limits_coherent[:, 1]

    use_save = False
    if use_save == True:
        print("using saved data")
        saved_limits = np.genfromtxt("limits/dark_photon/dark_photon_limits_ccm.txt", delimiter=",")
        mlist = saved_limits[:,0]
        eplist = saved_limits[:,1]
    else:
        # Binary search.
        print("Running dark photon limits...")
        outlist = open("limits/dark_photon/dark_photon_limits_ccm.txt", "w")
        for i in range(mlist.shape[0]):
            hi = tmp[-1]
            lo = tmp[0]
            mid = (hi + lo) / 2
            while np.log(hi) - np.log(lo) > 0.3:  # + events_gen2((mlist[i], tmp[j]))
                mid = (hi + lo) / 2
                lg = Chi2(EventsGen2((mlist[i], mid)), (n_bg+n_nu), (n_bg+n_nu), 0.28)
                #print(lg)
                if lg < 4:
                  lo = mid
                else:
                  hi = mid
            eplist[i] = mid
            outlist.write(str(mlist[i]))
            outlist.write(",")
            outlist.write(str(mid))
            outlist.write("\n")
            print(mid)

        outlist.close()





    # Plot the existing limits.
    #plt.fill_between(ldmx[:,0], ldmx[:,1], y2=1, label="LDMX", color="wheat", alpha=0.2)
    plt.fill_between(miniboone[:,0], miniboone[:,1], y2=1, color="plum", alpha=0.2)
    plt.fill_between(na64[:,0], na64[:,1], y2=1, color="maroon", alpha=0.2)
    plt.fill_between(miniboone[:,2], miniboone[:,3], y2=1, color="orchid", alpha=0.2)
    plt.fill_between(lsnd[:,0], lsnd[:,1], y2=1, color="crimson", alpha=0.2)

    #plt.plot(ldmx[:,0], ldmx[:,1], label="LDMX", color="gold", ls="dashed")
    plt.plot(miniboone[:,0], miniboone[:,1], color="plum", ls="dashed")
    plt.plot(na64[:,0], na64[:,1], color="maroon", ls="dashed")
    plt.plot(miniboone[:,2], miniboone[:,3], color="orchid", ls="dashed")
    plt.plot(lsnd[:,0], lsnd[:,1], color="crimson", ls="dashed")

    # Plot relic density limit
    plt.plot(relic[:,0], relic[:,1], color="k", linewidth=2)

    # Plot the derived limits.
    ep_smooth = signal.savgol_filter(eplist, 9, 3)
    ep_smooth_coherent = signal.savgol_filter(eplist_coherent, 9, 3)
    plt.plot(mlist, ep_smooth, label="CCM LAr (10 ton, 3 years)", linewidth=2, color="blue")
    plt.plot(mlist_coherent, ep_smooth_coherent, label="COHERENT CsI (14.6 kg, 1 year)", linewidth=2, color="blue", ls="dashed")


    #plt.text(30,1e-9,'LDMX', rotation=25, fontsize=9)
    plt.text(82,5e-7,'MiniBooNE \n (Nucleus)', rotation=0, fontsize=9, color="plum", weight="bold")
    plt.text(45,8e-6,'MiniBooNE \n (Electron)', rotation=0, fontsize=9, color="orchid", weight="bold")
    plt.text(40,2e-7,'NA64', rotation=40, fontsize=9, color="maroon", weight="bold")
    plt.text(10,8e-8,'LSND', rotation=40, fontsize=9, color="crimson", weight="bold")
    plt.text(15,1e-10,'Relic Density', rotation=35, fontsize=9, color="k", weight="bold")
    #plt.text(65,4e-7,'COHERENT Ar (1 year)', rotation=35, fontsize=9, color="blue", weight="bold")  # med=dp
    #plt.text(3,5e-10,'CCM LAr (10 ton, 1 year)', rotation=40, fontsize=9, color="blue", weight="bold")  #med only


    plt.legend(loc="upper left", framealpha=1.0)

    plt.xscale("Log")
    plt.yscale("Log")
    plt.xlim((1, 5e2))
    plt.ylim((1e-11,3e-5))
    plt.ylabel(r"$(\epsilon^\chi)^2$", fontsize=13)
    plt.xlabel(r"$M_{A^\prime}$ [MeV]", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/dark_photon/dark_photon_limits_ccm_coherent.png")
    plt.savefig("plots/dark_photon/dark_photon_limits_ccm_coherent.pdf")





if __name__ == "__main__":
    # Plot timing spectra

    dm_flux_test = DMFluxIsoPhoton(photon_flux, dark_photon_mass=75, coupling=1e-6, dark_matter_mass=25,
                                   detector_distance=20, pot_mu=0.145, pot_sigma=0.1, pot_sample=1e5,
                                   sampling_size=2000, verbose=False)
    plt.hist(dm_flux_test.timing, bins=25)
    plt.ylabel("a.u.")
    plt.xlabel(r"Arrival Time ($\mu$s)")
    plt.savefig("plots/dark_photon/spectra/ccm_timing_signal.png")
    plt.clf()
    dm_signal = EventsGen2((75, 0.001))
    energies = n_meas[:, 0] / pe_per_mev
    plt.plot(energies,dm_signal, drawstyle="steps-mid")
    plt.savefig("plots/dark_photon/spectra/ccm_energy_signal.png")
    plt.clf()

    t = prompt_pdf[:,0]
    p = prompt_pdf[:,1]

    t_smooth = np.linspace(prompt_pdf[0,0],prompt_pdf[-1,0],1000)
    p_smooth = np.empty_like(t_smooth)
    i = 0
    for tt in t_smooth:
        p_smooth[i] = np.interp(tt, t, p)
        i += 1

    print(np.sum(p), np.sum(p_smooth))

    plt.plot(t,p,drawstyle="steps-mid")
    plt.plot(t_smooth,p_smooth/np.sum(p_smooth))
    plt.savefig("plots/dark_photon/spectra/prompt_distribution.png")


    main()
