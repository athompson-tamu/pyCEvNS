import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.special import exp1

from pyCEvNS.axion import PrimakoffAxionFromBeam

# Declare constants.
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600*24
pot_per_year = 1.1e21


def runGenerator(flux):
    generator = PrimakoffAxionFromBeam(photon_rates=flux, axion_mass=0.001, axion_coupling=5e-8, target_mass=28e3,
                                target_z=14, target_photon_cross=1e-24, detector_distance=304,
                                detector_length=6, detector_area=21)
    generator.simulate(100)
    print(len(generator.axion_angle))
    return generator.axion_angle, generator.gamma_sep_angle




def main():
    # Flux from pythia8 (Doojin)
    pot_sample = 10000
    scale = pot_per_year / pot_sample / 365 / 24 / 3600
    gamma_data = np.genfromtxt("data/dune/hepmc_gamma_flux_from_pi0.txt")
    gamma_e = 1000*gamma_data[:,0] # convert to mev
    gamma_theta = gamma_data[:,-1]
    gamma_wgt = scale*np.ones_like(gamma_e)

    # zip event-by-event flux array
    flux = np.array(list(zip(gamma_wgt,gamma_e,gamma_theta)))

    # make weighted histogram of flux
    energy_edges = np.linspace(0, 10000, 20)
    theta_edges = np.linspace(0, 1.5, 20)
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
    np.savetxt("data/dune/dune_binned_pythia8_gamma_pot_per_s.txt", binned_flux)
    
    
    
    
    

    axion_angles, deltatheta = runGenerator(binned_flux)
    axion_angles = [180*x/np.pi for x in axion_angles]
    deltatheta = [180*x/np.pi for x in deltatheta]


    # Plot the axion angular distribution
    angle_edges = np.linspace(0, 180, 100)
    plt.hist(axion_angles, bins=angle_edges, density=True, histtype="step", label=r"$a$ in target")
    plt.hist(180*gamma_theta/np.pi, weights=gamma_wgt, bins=angle_edges, density=True, histtype="step", label=r"$\gamma$ in target (Pythia8)")
    plt.xlabel(r"$\theta$ [deg]", fontsize=15)
    plt.title(r"$m_a = 5$ MeV", loc="right", fontsize=15)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    plt.clf()



    # Plot the photon angular separation
    angle_edges = np.linspace(0, 45, 20)
    plt.hist(deltatheta, density=True, bins=angle_edges, histtype="step", label=r"$a\to\gamma\gamma$")
    plt.xlabel(r"$\Delta\theta_{\gamma\gamma}$ [deg]", fontsize=15)
    plt.title(r"$m_a = 5$ MeV", loc="right", fontsize=15)
    plt.xlim(right=45)
    plt.ylim(bottom=0)
    plt.show()
    plt.clf()
    
    
    # show gamma distribution
    plt.hist2d(binned_flux[:,1], binned_flux[:,2], weights=binned_flux[:,0], range=[[0,8000],[0,1.5]], bins=[energy_bins, theta_bins])
    plt.xlim((0,8000))
    plt.ylim((0,1.5))
    plt.show()

    plt.hist2d(gamma_e, gamma_theta, weights=gamma_wgt, range=[[0,8000],[0,1.5]], bins=200)
    plt.xlabel(r"$E_\gamma$ [MeV]")
    plt.ylabel(r"$\theta$ [rad]")
    plt.title("pythia8 photons from POT (hard process)", loc="right")
    plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()