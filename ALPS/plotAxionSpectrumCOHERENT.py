import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

import sys

from pyCEvNS.axion import Axion

# conversion between units
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600 * 24
me = 0.511


def PlotSpectra(flux, detector, mass, coupling):
    bins = np.linspace(0, np.max(flux[:, 0]), flux.shape[0])
    bins = np.zeros(flux.shape[0]+1)
    for i in range(0, flux.shape[0]-1):
        width = (flux[i + 1, 0] - flux[i, 0]) / 2
        bins[i] = flux[i, 0] - width
    last_width = (flux[-1, 0] - flux[-2, 0]) / 2
    bins[flux.shape[0]-1] = flux[-2, 0] + last_width
    bins[flux.shape[0]] = flux[-1, 0] + last_width
    centers = flux[:, 0]  # (bins[1:] + bins[:-1]) / 2

    # Configure parameters.
    if detector == "csi":
        det_dis = 19.3
        min_decay_length = 10
        det_mass = 14.6
        det_am = 123.8e3
        det_z = 55
        days = 2 * 365
        det_density = 4510  # kg/m^3
    if detector == "ar":
        min_decay_length = 14.7
        det_dis = 28.3
        det_mass = 7000  # 610
        det_am = 37.211e3
        det_z = 18
        days = 3 * 365
        det_density = 1396  # kg/m^3
    mercury_mass = 186850
    mercury_z = 80
    mercury_n_gamma = 3e-23
    det_r = (det_mass / (det_density) * 3 / (4 * np.pi)) ** (1 / 3)


    # Simulate axion production
    axion_sim = Axion(flux, mass, coupling, mercury_mass,
                      mercury_z, mercury_n_gamma, det_dis, min_decay_length)
    axion_sim.simulate(5000)

    decay_energies, decay_photons = axion_sim.photon_events_binned(4*np.pi*det_r**2, days*s_per_day, 0)
    scatter_energies, scatter_photons = axion_sim.scatter_events_binned(det_mass*mev_per_kg/det_am,
                                                                        det_z, days*s_per_day, 0)





    decay_y, decay_x = np.histogram(decay_energies, weights=decay_photons, bins=bins)
    scatter_y, scatter_x = np.histogram(scatter_energies, weights=scatter_photons, bins=bins)

    #print(decay_y, scatter_y)

    #plt.plot(centers, decay_y, ls="steps-mid", label="Decays")
    plt.plot(centers, scatter_y, ls="steps-mid", label="Scatters", color="orange")
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlim((0,np.max(flux[:,0])))
    #plt.ylim((1e-5, 1e11))
    plt.xlabel("E [MeV]")
    plt.ylabel("Counts")
    plt.legend()
    plt.title(r"$m_a =$ 1 eV", loc="right")
    plt.savefig("plots/coherent_photon_spectrum/coherent_signal_1ev.png")
    plt.savefig("plots/coherent_photon_spectrum/coherent_signal_1ev.pdf")


def main(detector, axion_mass, axion_coupling):
    coherent = np.genfromtxt('data/photon_flux_COHERENT_log_binned.txt')  # flux at reactor surface
    coherent[:, 1] *= 1e11  # get flux at the core
    PlotSpectra(coherent, detector, axion_mass, axion_coupling)




if __name__ == "__main__":
  main(str(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))