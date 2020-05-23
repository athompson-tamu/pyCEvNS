import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt("data/photon_fluxes_3gev_hg.txt", skip_header=1)

bins = np.logspace(np.log10(1.6),np.log10(930),46)
print(bins)
print(data.shape)

ep = data[:,0]
neutron = data[:,1]
proton = data[:,2]
pim = data[:,3]
pip = data[:,4]
kaon0L = data[:,5]
sigma0 = data[:,6]
kaon0S = data[:,7]
kaonP = data[:,8]
pi0 = data[:,9]
mum = data[:,10]

plt.plot(bins,ep,drawstyle="steps-mid")
plt.plot(bins,neutron,drawstyle="steps-mid")
plt.plot(bins,proton,drawstyle="steps-mid")
plt.plot(bins,pim,drawstyle="steps-mid")
plt.plot(bins,pip,drawstyle="steps-mid")
plt.plot(bins,kaon0L,drawstyle="steps-mid")
plt.plot(bins,kaon0S,drawstyle="steps-mid")
plt.plot(bins,kaonP,drawstyle="steps-mid")
plt.plot(bins,pi0,drawstyle="steps-mid")
plt.plot(bins,mum,drawstyle="steps-mid")

plt.yscale('log')
plt.xscale('log')
plt.savefig('3gev_photon_spectra.png')


photon_totals = np.empty([46,2])
photon_totals.shape
photon_totals[:,0] = bins
photon_totals[:,1] = ep + neutron + proton + pim + pip + kaonP + kaon0S + kaon0L + sigma0 + pi0 + mum
np.savetxt("jsns_3gev_photon_totals_1e5_POT.txt", photon_totals)

plt.clf()

data_800mev_em = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_em.txt")
data_800mev_ep = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_ep.txt")
data_800mev_mum = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_mum.txt")
data_800mev_neutron = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_neutron.txt")
data_800mev_pi0 = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_pi0.txt")
data_800mev_pim = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_pim.txt")
data_800mev_pip = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_pip.txt")
data_800mev_proton = np.genfromtxt("data/W_0.8GeV/gamma_particle/Gamma_proton.txt")

nbins = 42
edges = np.logspace(0, 3, nbins+1)
centers = np.zeros(nbins)
for i in range(0, nbins):
  centers[i] = edges[i] + (edges[i+1] - edges[i]) / 2

print(data_800mev_em.shape[0], data_800mev_ep.shape[0], data_800mev_mum.shape[0], data_800mev_neutron.shape[0],
      data_800mev_pim.shape[0], data_800mev_pip.shape[0], data_800mev_pi0.shape[0], data_800mev_proton.shape[0])
plt.hist([data_800mev_em[:,4], data_800mev_ep[:,4], data_800mev_mum[:,4], data_800mev_neutron[:,4],
          data_800mev_pim[:,4], data_800mev_pip[:,4], data_800mev_pi0[:,4], data_800mev_proton[:,4]],
         stacked=True, bins=centers)

plt.yscale('log')
plt.xscale('log')
plt.savefig("ccm_flux.png")

h_em = np.histogram(data_800mev_em[:,4], bins=edges)
h_ep = np.histogram(data_800mev_ep[:,4], bins=edges)
h_mum = np.histogram(data_800mev_mum[:,4], bins=edges)
h_neutron = np.histogram(data_800mev_neutron[:,4], bins=edges)
h_pim = np.histogram(data_800mev_pim[:,4], bins=edges)
h_pip = np.histogram(data_800mev_pip[:,4], bins=edges)
h_pi0 = np.histogram(data_800mev_pi0[:,4], bins=edges)
h_proton = np.histogram(data_800mev_proton[:,4], bins=edges)

photon_800mev_counts = h_em[0] + h_ep[0] + h_mum[0] + h_neutron[0] + h_pim[0] + h_pip[0] + h_pi0[0] + h_proton[0]
photon_800mev_energies = centers

print(np.sum(photon_800mev_counts))
print(photon_800mev_counts)
print(photon_800mev_energies)


photon_800mev_totals = np.empty([42,2])
photon_800mev_totals[:,0] = photon_800mev_energies
photon_800mev_totals[:,1] = photon_800mev_counts
np.savetxt("ccm_800mev_photon_spectra_1e5_POT.txt", photon_800mev_totals)







