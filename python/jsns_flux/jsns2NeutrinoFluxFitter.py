import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scipy.integrate import quad

# JSNS^2
nu_e = np.genfromtxt("jsns_nu_e.txt", delimiter=',')
nu_mu = np.genfromtxt("jsns_nu_mu_nodelta.txt", delimiter=',')
nubar_mu = np.genfromtxt("jsns_nubar_mu.txt", delimiter=',')

bin_width = 2 # mev
pot_per_s = 2e15  # pot / s
nu_per_s = bin_width * pot_per_s

# Get the JSNS^2 interpolated flux function.
def smoothFlux(nrg, data):
    return np.interp(nrg, data[:,0], data[:,1])

norm_nu_e = quad(smoothFlux, 0, 300, args=(nu_e,))[0]
norm_nu_mu = quad(smoothFlux, 0, 300, args=(nu_mu,))[0]
norm_nubar_mu = quad(smoothFlux, 0, 300, args=(nubar_mu,))[0]


def numuPDF(energy):
    return smoothFlux(energy, nu_mu) / norm_nu_mu

def nuePDF(energy):
    return smoothFlux(energy, nu_e) / norm_nu_e

def nubarmuPDF(energy):
    return smoothFlux(energy, nubar_mu) / norm_nubar_mu


# Prepare bins
nrg_edges = np.arange(0, 302, 2)
nrg_bins = (nrg_edges[:-1] + nrg_edges[1:]) / 2

nu_mu_corr = smoothFlux(nrg_bins, nu_mu)
nu_e_corr = smoothFlux(nrg_bins, nu_e)
nubar_mu_corr = smoothFlux(nrg_bins, nubar_mu)

fine_bins = np.linspace(0,300,10000)

# determine normalization
mubar_binned = smoothFlux(nrg_bins, nubar_mu)
e_binned = smoothFlux(nrg_bins, nu_e)
mu_binned = smoothFlux(nrg_bins, nu_mu)
print("mubar norm = ", np.sum(mubar_binned)* nu_per_s)
print("mu norm = ", np.sum(mu_binned)* nu_per_s)
print("e norm = ", np.sum(e_binned)* nu_per_s)

# Plot JSNS^2
plt.scatter(nrg_bins, smoothFlux(nrg_bins, nu_mu) * nu_per_s, color='r')
plt.scatter(nrg_bins, smoothFlux(nrg_bins, nu_e) * nu_per_s, color='m')
plt.scatter(nrg_bins, smoothFlux(nrg_bins, nubar_mu) * nu_per_s, color='k')
plt.plot(fine_bins, smoothFlux(fine_bins, nu_mu) * nu_per_s, color='r')
plt.plot(fine_bins, smoothFlux(fine_bins, nu_e) * nu_per_s, color='m')
plt.plot(fine_bins, smoothFlux(fine_bins, nubar_mu) * nu_per_s, color='k')

#plt.plot(nrg_bins, de(nrg_bins, *de_opt) * nu_per_s, color='m', ls='dashed')
#plt.plot(nrg_bins, de(nrg_bins, *dmubar_opt) * nu_per_s, color='k', ls='dashed')
plt.xlim((0,302))
plt.yscale('log')
plt.show()

