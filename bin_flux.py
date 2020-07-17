import numpy as np
import matplotlib.pyplot as plt

import sys

def main():
  flux = np.genfromtxt("/data/jsns/Hg_3GeV/gamma_process/Gamma_eBrem.txt")  # expect (angle, px, py, pz, E)
  nbins=100
  edges = np.linspace(min(flux[:,4]), max(flux[:,4]), nbins+1)
  hist, edges = np.histogram(flux[:,4], bins=edges)

  centers = np.zeros(nbins)
  for i in range(0, nbins):
    centers[i] = edges[i] + (edges[i+1] - edges[i]) / 2


  plt.plot(centers, hist, drawstyle='steps-mid')
  plt.yscale('log')
  plt.xlabel("Energy [MeV]")
  plt.ylabel("Counts")
  plt.xlim((0,max(flux[:,4])))
 

  # Save the histogram of fluxes.
  x = np.array([centers,hist])
  x = x.transpose()
  print(x.shape)
  np.savetxt("brem.txt", x)

  plt.show()


if __name__ == "__main__":
  main()

