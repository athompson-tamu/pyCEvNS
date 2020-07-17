# Flux from pythia8 (Doojin)
import numpy as np
import matplotlib.pyplot as plt

pot_per_year = 1.1e21
pot_sample = 10000
scale = pot_per_year / pot_sample / 365 / 24 / 3600
gamma_data = np.genfromtxt("data/dune/hepmc_gamma_flux_from_pi0.txt")
gamma_e = 1000*gamma_data[:,0] # convert to mev
gamma_theta = gamma_data[:,-1]
gamma_wgt = scale*np.ones_like(gamma_e)

# zip event-by-event flux array
flux = np.array(list(zip(gamma_wgt,gamma_e,gamma_theta)))

# make 1d histogram
energy_edges = np.linspace(0, 80000, 500)
hist_flux = np.histogram(gamma_e, weights=gamma_wgt, bins=energy_edges)
hist_weights = hist_flux[0]
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
binned_flux = np.array(list(zip(hist_weights,
                                energy_bins)))
binned_flux = binned_flux[binned_flux[:,0]>0]
#np.savetxt("data/dune/dune_binned_pythia8_gamma_pot_per_s_1d.txt", binned_flux)

plt.hist(energy_bins, weights=hist_weights, bins=energy_edges)
plt.yscale('log')
plt.show()
plt.close()





# make weighted histogram of flux
energy_edges = np.linspace(0, 80000, 200)
theta_edges = np.linspace(0, 1.5, 100)
hist_flux = np.histogram2d(gamma_e, gamma_theta, weights=gamma_wgt, bins=[energy_edges,theta_edges])
hist_weights = hist_flux[0]
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
theta_bins = (theta_edges[:-1] + theta_edges[1:]) / 2
flattened_weights = hist_weights.flatten()
flattened_energies = np.repeat(energy_bins,theta_bins.shape[0])
flattened_thetas = np.tile(theta_bins,energy_bins.shape[0])
binned_flux = np.array(list(zip(flattened_weights,
                                flattened_energies,
                                flattened_thetas)))
binned_flux = binned_flux[binned_flux[:,0]>0]
np.savetxt("data/dune/dune_binned_pythia8_gamma_pot_per_s_2d.txt", binned_flux)
