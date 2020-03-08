import numpy as np
import matplotlib.pyplot as plt

flux = np.genfromtxt("flux.txt")
nbins = 42
edges = np.power(10, np.linspace(0.217, 2.7, nbins+1))
hist, edges = np.histogram(flux, bins=edges)

centers = np.zeros(nbins)
for i in range(0, nbins):
  centers[i] = edges[i] + (edges[i+1] - edges[i]) / 2


plt.plot(centers, hist, drawstyle='steps-mid')
plt.yscale('log')
plt.xlabel("Energy [keV]")
plt.ylabel("Counts")
plt.savefig("flux.png")

# Save the histogram of fluxes.
x = np.array([centers,hist])
x = x.transpose()
print(x.shape)
np.savetxt("photon_flux_COHERENT_log_binned.txt", x)


