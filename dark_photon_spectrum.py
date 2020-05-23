import sys
from pyCEvNS.events import *
from pyCEvNS.flux import *
import matplotlib.pyplot as plt
import numpy as np

from dmphoton import DMFluxFromPhoton

pot_day = 5e20
pot_sample = 100000
photon_scale = 1 / pot_sample #306 * pot_day / pot_sample

neutron = np.genfromtxt("data/neutron_flux.txt", delimiter=",")
proton = np.genfromtxt("data/proton_flux.txt", delimiter=",")
pi0 = np.genfromtxt("data/pi0_flux.txt", delimiter=",")
pim = np.genfromtxt("data/pim_flux.txt", delimiter=",")
pip = np.genfromtxt("data/pip_flux.txt", delimiter=",")
ep = np.genfromtxt("data/ep_flux.txt", delimiter=",")
em = np.genfromtxt("data/em_flux.txt", delimiter=",")
triton = np.genfromtxt("data/triton_flux.txt", delimiter=",")

neutron[:,1] *= photon_scale
proton[:,1] *= photon_scale
pi0[:,1] *= photon_scale
pip[:,1] *= photon_scale
pim[:,1] *= photon_scale
ep[:,1] *= photon_scale
em[:,1] *= photon_scale
triton[:,1] *= photon_scale

nbins = 42
edges = np.logspace(0, 3, nbins+1)
centers = np.zeros(nbins)
for i in range(0, nbins):
  centers[i] = edges[i] + (edges[i+1] - edges[i]) / 2


photon_flux = np.genfromtxt("pyCEvNS/data/photon_flux_COHERENT_log_binned.txt")

"""
dm_flux_1 = DMFluxIsoPhoton(photon_flux, dark_photon_mass=5, coupling=1e-4, dark_matter_mass=5/3,
                            life_time=5e-17, pot_sample=100000/306, sampling_size=2000, nbins=20, verbose=False)
dm_flux_2 = DMFluxIsoPhoton(photon_flux, dark_photon_mass=10, coupling=1e-4, dark_matter_mass=10/3,
                            life_time=5e-17, pot_sample=100000/306, sampling_size=2000, nbins=20, verbose=False)
dm_flux_3 = DMFluxIsoPhoton(photon_flux, dark_photon_mass=100, coupling=1e-4, dark_matter_mass=100/3,
                            life_time=5e-17, pot_sample=100000/306, sampling_size=2000, nbins=20, verbose=False)
dm_flux_4 = DMFluxIsoPhoton(photon_flux, dark_photon_mass=200, coupling=1e-4, dark_matter_mass=200/3,
                            life_time=5e-17, pot_sample=100000/306, sampling_size=2000, nbins=20, verbose=False)
"""

pim_wgt = np.ones_like(pim[:,0]) * photon_scale
pip_wgt = np.ones_like(pip[:,0]) * photon_scale
pi0_wgt = np.ones_like(pi0[:,0]) * photon_scale
ep_wgt = np.ones_like(ep[:,0]) * photon_scale
em_wgt = np.ones_like(em[:,0]) * photon_scale
neutron_wgt = np.ones_like(neutron[:,0]) * photon_scale
proton_wgt = np.ones_like(proton[:,0]) * photon_scale

fig, ax1 = plt.subplots()

#ax1.hist([pim[:,0],pip[:,0],proton[:,0],neutron[:,0],em[:,0],ep[:,0],pi0[:,0]],
 #        color=['navy','blue','darkred','dimgray','steelblue','skyblue','indigo',],
  #       weights=[pim_wgt, pip_wgt, proton_wgt, neutron_wgt, em_wgt, ep_wgt, pi0_wgt],
   #      label=[r"$\pi^-$", r"$\pi^+$",r"Proton",r"Neutron",r"$e^-$", r"$e^+$", r"$\pi^0$", ],
    #     bins=centers,stacked=True)

ax1.hist([pim[:,0],pip[:,0],pi0[:,0]],
         color=['navy','blue','indigo',],
         weights=[pim_wgt, pip_wgt, pi0_wgt],
         label=[r"$\pi^-$", r"$\pi^+$", r"$\pi^0$", ],
         bins=centers,stacked=True)

"""
ax2 = ax1.twinx()

print(dm_flux_2.norm)

#plt.hist(dm_flux_1.ev, bins=20,weights=10*dm_flux_1.fx*dm_flux_1.norm, label=r"$M_{A^\prime} = 5$ MeV", linewidth=1.5, histtype='step')
ax2.hist(dm_flux_2.energy, bins=20,weights=dm_flux_2.getScaledWeights(), label=r"$M_{A^\prime} = 10$ MeV",
         color='orange',linewidth=1.5, histtype='step')
ax2.hist(dm_flux_3.energy, bins=20,weights=dm_flux_3.getScaledWeights(), label=r"$M_{A^\prime} = 100$ MeV",
         color='red',linewidth=1.5, histtype='step')
ax2.hist(dm_flux_4.energy, bins=20,weights=dm_flux_4.getScaledWeights(), label=r"$M_{A^\prime} = 200$ MeV",
         color='lime',linewidth=1.5, histtype='step')
"""

ax1.set_xlim((2.86,1000))
#ax1.set_ylim((2e18,2e24))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel(r"Events / POT", fontsize=13)
ax1.set_xlabel(r"$E_\gamma$ [MeV]", fontsize=13)
ax1.legend(framealpha=1)

"""
ax2.set_ylim((5e12,5e20))
ax2.set_yscale("log")
ax2.set_ylabel(r"$A^\prime \rightarrow \chi \chi$ Events / Year", fontsize=13)
ax2.legend(framealpha=1,loc="upper center")
"""
#ax1.set_title(r"$\epsilon = 10^{-4}$",loc="right")
fig.tight_layout()


fig.savefig("plots/dark_photon/COHERENT_photon_spectrum_1pot_pionly.png")
fig.savefig("plots/dark_photon/COHERENT_photon_spectrum_1pot_pionly.pdf")