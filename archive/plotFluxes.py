import numpy as np
import matplotlib.pyplot as plt

pot_day = 5e20
pot_sample = 100000
photon_scale = 308 * pot_day / pot_sample


neutron = np.genfromtxt("data/neutron_flux.txt", delimiter=",")
proton = np.genfromtxt("data/proton_flux.txt", delimiter=",")
pi0 = np.genfromtxt("data/pi0_flux.txt", delimiter=",")
pim = np.genfromtxt("data/pim_flux.txt", delimiter=",")
pip = np.genfromtxt("data/pip_flux.txt", delimiter=",")
ep = np.genfromtxt("data/ep_flux.txt", delimiter=",")
em = np.genfromtxt("data/em_flux.txt", delimiter=",")
triton = np.genfromtxt("data/triton_flux.txt", delimiter=",")



nbins = 42
edges = np.logspace(0, 3, nbins+1)
hist_pr, edges = np.histogram(proton[:,0], bins=edges)
hist_nu, edges = np.histogram(neutron[:,0], bins=edges)
hist_pi0, edges = np.histogram(pi0[:,0], bins=edges)
hist_pim, edges = np.histogram(pim[:,0], bins=edges)
hist_pip, edges = np.histogram(pip[:,0], bins=edges)
hist_ep, edges = np.histogram(ep[:,0], bins=edges)
hist_em, edges = np.histogram(em[:,0], bins=edges)
hist_triton, edges = np.histogram(triton[:,0], bins=edges)

centers = np.zeros(nbins)
for i in range(0, nbins):
  centers[i] = edges[i] + (edges[i+1] - edges[i]) / 2

plt.plot(centers, hist_pr, label="Proton", drawstyle='steps-mid', color='r')
plt.plot(centers, hist_nu, label="Neutron", drawstyle='steps-mid', color='k')
plt.plot(centers, hist_pi0, label=r"$\pi^0$", drawstyle='steps-mid', color='m')
plt.plot(centers, hist_pim, label=r"$\pi^-$", drawstyle='steps-mid', color='m', ls='dashed')
plt.plot(centers, hist_pip, label=r"$\pi^+$", drawstyle='steps-mid', color='m', ls='dotted')
plt.plot(centers, hist_ep, label=r"$e^+$", drawstyle='steps-mid', color='b')
plt.plot(centers, hist_em, label=r"$e^-$", drawstyle='steps-mid', color='b', ls='dashed')
plt.plot(centers, hist_triton, label="Triton", drawstyle='steps-mid', color='orange')
plt.yscale('log')
plt.xscale('log')
plt.xlim((2.86,1000))
plt.legend()
plt.xlabel("Energy [MeV]")
plt.ylabel("Events")
plt.savefig("plots/coherent_fluxes_separated.png")
plt.savefig("plots/coherent_fluxes_separated.pdf")

plt.clf()
# Plot angular distribution

nbins=20
edges = np.linspace(-1, 1, nbins+1)
edges_coarse = np.linspace(-1,1,11)
e_high = 500000
e_low = 50
proton = proton[(proton[:,0] >= e_low ) & (proton[:,0] <= e_high)]
neutron = neutron[(neutron[:,0] >= e_low) & (neutron[:,0] <= e_high)]
ep = ep[(ep[:,0] >= e_low) & (ep[:,0] <= e_high)]
em = em[(em[:,0] >= e_low) & (em[:,0] <= e_high)]
pi0 = pi0[(pi0[:,0] >= e_low) & (pi0[:,0] <= e_high)]
pip = pip[(pip[:,0] >= e_low) & (pip[:,0] <= e_high)]
pim = pim[(pim[:,0] >= e_low) & (pim[:,0] <= e_high)]

norm = False
hist_pr, edges = np.histogram(proton[:,1], bins=edges, density=norm)
hist_nu, edges = np.histogram(neutron[:,1], bins=edges, density=norm)
hist_pi0, edges = np.histogram(pi0[:,1], bins=edges, density=norm)
hist_pim, edges = np.histogram(pim[:,1], bins=edges, density=norm)
hist_pip, edges = np.histogram(pip[:,1], bins=edges, density=norm)
hist_ep, edges = np.histogram(ep[:,1], bins=edges, density=norm)
hist_em, edges = np.histogram(em[:,1], bins=edges, density=norm)

centers = np.zeros(nbins)
for i in range(0, nbins):
  centers[i] = edges[i] + (edges[i+1] - edges[i]) / 2


centers_coarse = np.zeros(10)
for i in range(0, 10):
  centers_coarse[i] = edges_coarse[i] + (edges_coarse[i+1] - edges_coarse[i]) / 2


plt.plot(centers, hist_pr, label="Proton", drawstyle='steps-mid', color='r')
plt.plot(centers, hist_nu, label="Neutron", drawstyle='steps-mid', color='k')
plt.plot(centers, hist_pi0, label=r"$\pi^0$", drawstyle='steps-mid', color='m')
plt.plot(centers, hist_pim, label=r"$\pi^-$", drawstyle='steps-mid', color='m', ls='dashed')
plt.plot(centers, hist_pip, label=r"$\pi^+$", drawstyle='steps-mid', color='m', ls='dotted')
plt.plot(centers, hist_ep, label=r"$e^+$", drawstyle='steps-mid', color='b')
plt.plot(centers, hist_em, label=r"$e^-$", drawstyle='steps-mid', color='b', ls='dashed')
plt.yscale('log')
plt.title(r"$\geq 50$ MeV", loc="right")
plt.legend()
plt.xlabel(r"$\cos\theta$")
plt.ylabel("a.u.")
plt.savefig("plots/coherent_fluxes_separated_cosines_50MeVandAbove.png")
plt.savefig("plots/coherent_fluxes_separated_cosine_50MeVandAbove.pdf")
