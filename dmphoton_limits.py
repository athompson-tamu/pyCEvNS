import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

prompt_pdf = np.genfromtxt('data/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')
nin_pdf = np.genfromtxt('data/arrivalTimePDF_promptNeutrons.txt', delimiter=',')

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


prompt_flux = Flux('prompt')
delayed_flux = Flux('delayed')

pe_per_mev = 0.0878 * 13.348 * 1000


n_prompt = np.zeros(7*2)
n_delayed = np.zeros(7*2)
n_nin = np.zeros(7*2)
n_bg = 405*4466/4466/(12*12)*np.ones(7*2)
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

n_meas = np.zeros((7*2, 2))
for i in range(7*2):
    pe = (i%7)*2+17
    t = i//7*0.5+0.25
    n_meas[i, 0] = pe
    n_meas[i, 1] = t
    n_prompt[i] = efficiency(pe) * binned_events_nucleus((pe-1)/pe_per_mev, (pe+1) / pe_per_mev, 4466, Detector('csi'), prompt_flux, NSIparameters(), flavor='mu', rn=5.5) * prompt_time(t)
    n_delayed[i] = efficiency(pe) *         (binned_events_nucleus((pe-1)/pe_per_mev, (pe+1) / pe_per_mev, 4466, Detector('csi'), delayed_flux, NSIparameters(), flavor='e', rn=5.5) + 
        binned_events_nucleus((pe-1)/pe_per_mev, (pe+1) / pe_per_mev, 4466, Detector('csi'), delayed_flux, NSIparameters(), flavor='mubar', rn=5.5)) * delayed_time(t)
    # n_nin[i] = (efficiency(pe-0.5) * nin[int(pe-1), 1]+ efficiency(pe+0.5) * nin[int(pe), 1]) * nin_time(t) * 365000/4466
    
n_nu = n_prompt+n_delayed


photon_flux = np.genfromtxt("pyCEvNS/data/photon_flux_COHERENT_log_binned.txt")  # binned photon spectrum from Rebecca
dm_gen = DmEventsGen(dark_photon_mass=75, life_time=5e-17, dark_matter_mass=25, expo=4466)
dm_flux = DMFluxIsoPhoton(photon_flux, dark_photon_mass=75, coupling=1, dark_matter_mass=25,
                           life_time=5e-17, pot_sample=100000, sampling_size=2000, verbose=False)
dm_gen.fx = dm_flux
#dm_gen.fx = DMFluxFromPi0Decay(photon_flux, 75, 5e-17, 1, 5)


def EventsGen(cube):
    med_mass = cube[0]
    eps = cube[1]
    return dm_gen.events(med_mass, eps, n_meas)  # mediator mass, epsilon


# mediator = dark photon scenario
dm_gen_2 = DmEventsGen(dark_photon_mass=1, life_time=5e-17, dark_matter_mass=0, expo=4466)
def EventsGen2(cube):
    dp_mass = cube[0]
    if cube[0] > 450:
        dp_mass = 450

    dm_gen_2.dm_mass = dp_mass / 3
    flux = DMFluxIsoPhoton(photon_flux, dark_photon_mass=dp_mass, coupling=1, dark_matter_mass=0,
                           life_time=5e-17, pot_sample=100000, sampling_size=1000, verbose=False)
    dm_gen_2.fx = flux
    return dm_gen_2.events(cube[0], cube[1], n_meas)  # mediator mass, epsilon


def Chi2(n_signal, n_bg, n_obs, sigma):
    likelihood = np.zeros(n_obs.shape[0])
    alpha = np.sum(n_signal*(n_obs-n_bg)/n_obs)/(1/sigma**2+np.sum(n_signal**2/n_obs))
    likelihood += (n_obs-n_bg-(1+alpha)*n_signal)**2/(n_obs)
    return np.sum(likelihood)+(alpha/sigma)**2


mlist = np.logspace(0, 3, 200)
eplist = np.ones_like(mlist)
tmp = np.logspace(-20, 0, 20)

use_save = True
if use_save == True:
    saved_limits = np.genfromtxt("limits/dark_photon/dark_photon_limits_mediatorIsDP.txt", delimiter=",")
    mlist = saved_limits[:,0]
    eplist = saved_limits[:,1]
else:
    # Binary search.
    outlist = open("limits/dark_photon/dark_photon_limits_mediatorIsDP.txt", "w")
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
plt.fill_between(miniboone[:,0], miniboone[:,1], y2=1, label="MiniBooNE Nucleus", color="plum", alpha=0.2)
plt.fill_between(na64[:,0], na64[:,1], y2=1, label="NA64", color="maroon", alpha=0.2)
plt.fill_between(miniboone[:,2], miniboone[:,3], y2=1, label="MiniBooNE Electron", color="orchid", alpha=0.2)
plt.fill_between(lsnd[:,0], lsnd[:,1], y2=1, label="LSND", color="crimson", alpha=0.2)

#plt.plot(ldmx[:,0], ldmx[:,1], label="LDMX", color="gold", ls="dashed")
plt.plot(miniboone[:,0], miniboone[:,1], label="MiniBooNE Nucleus", color="plum", ls="dashed")
plt.plot(na64[:,0], na64[:,1], label="NA64", color="maroon", ls="dashed")
plt.plot(miniboone[:,2], miniboone[:,3], label="MiniBooNE Electron", color="orchid", ls="dashed")
plt.plot(lsnd[:,0], lsnd[:,1], label="LSND", color="crimson", ls="dashed")

# Plot relic density limit
plt.plot(relic[:,0], relic[:,1], label="Relic Density", color="k", linewidth=2)

# Plot the derived limits.
spl = UnivariateSpline(mlist, eplist,k=5,s=0.1)
ep_smooth = signal.savgol_filter(eplist, 9, 3)
plt.plot(mlist, ep_smooth, label="COHERENT CsI", linewidth=2, color="blue")


#plt.text(30,1e-9,'LDMX', rotation=25, fontsize=9)
plt.text(82,5e-7,'MiniBooNE \n (Nucleus)', rotation=0, fontsize=9, color="plum", weight="bold")
plt.text(45,8e-6,'MiniBooNE \n (Electron)', rotation=0, fontsize=9, color="orchid", weight="bold")
plt.text(40,2e-7,'NA64', rotation=40, fontsize=9, color="maroon", weight="bold")
plt.text(10,8e-8,'LSND', rotation=40, fontsize=9, color="crimson", weight="bold")
plt.text(65,4e-7,'COHERENT CsI (1 year)', rotation=35, fontsize=9, color="blue", weight="bold")  # med=dp
#plt.text(1.2,5e-9,'COHERENT CsI (1 year)', rotation=0, fontsize=9, color="blue", weight="bold")  #med only
plt.text(15,1e-10,'Relic Density', rotation=35, fontsize=9, color="k", weight="bold")


#plt.legend(loc="upper left", framealpha=1.0)

plt.xscale("Log")
plt.yscale("Log")
plt.xlim((1, 5e2))
plt.ylim((1e-11,3e-5))
plt.ylabel(r"$(\epsilon^\chi)^2$", fontsize=13)
plt.xlabel(r"$M_{A^\prime}$ [MeV]", fontsize=13)
plt.tight_layout()
plt.savefig("plots/dark_photon/dark_photon_limits_coherent_CsI_1yr_mediatorIsDp.png")
plt.savefig("plots/dark_photon/dark_photon_limits_coherent_CsI_1yr_mediatorIsDp.pdf")





# DEPRECATED CODE
# Dark photon mass, lifetime, dark matter mass
#dm_gen = DmEventsGen(75, 0.001, 5, expo=4466)
#dm_gen_pi0 = DmEventsGen(75, 5e-17, 5, expo=4466)
#dm_gen_pi0.fx = DMFluxFromPi0Decay(events_pd.values, 75, 5e-17, 1, 5)


#def events_gen(cube):
    # cube on (mediator mass, coupling)
 #   return dm_gen.events(10**cube[0], 10**cube[1], n_meas)

#for i in range(mlist.shape[0]):
 #   for j in range(tmp.shape[0]):  # + events_gen2((mlist[i], tmp[j]))
  #      lg = Chi2(DarkMatterEventsGen((mlist[i], tmp[j])), (n_bg+n_nu), (n_bg+n_nu), 0.28)
        #print(lg)
   #     if lg >= 4:
    #        eplist[i] = tmp[j]
     #       print(tmp[j])
      #      break