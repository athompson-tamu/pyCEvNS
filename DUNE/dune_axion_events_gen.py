import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.special import exp1

import sys
sys.path.append("../")
from pyCEvNS.axion import PrimakoffAxionFromBeam

from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Declare constants.
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600*24
pot_per_year = 1.1e21


# detector constants
det_mass = 50000
det_am = 37.211e3  # mass of target atom in MeV
det_z = 18  # atomic number
days = 1000  # days of exposure
det_area = 3*6  # cross-sectional det area
det_thresh = 0  # energy threshold
sig_limit = 2.0  # poisson significance (2 sigma)

def runGenerator(flux, ma, g):
    generator = PrimakoffAxionFromBeam(photon_rates=flux, axion_mass=ma, axion_coupling=g, target_mass=28e3,
                                target_z=14, target_photon_cross=1e-24, detector_distance=304,
                                detector_length=10, detector_area=50)
    generator.simulate(10)
    return generator




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
    energy_edges = np.linspace(0, 80000, 400)
    theta_edges = np.linspace(0, 1.5, 80)
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
    #np.savetxt("data/dune/dune_binned_pythia8_gamma_pot_per_s_2d.txt", binned_flux)

    
    def BestDecayCoupling(ma, eg, l):
        hc = 6.58e-22 * 3e8
        return np.sqrt(64 * np.pi * eg * hc / l / ma**4)


    
    # RUN THE GENERATORS
    axion_gen = runGenerator(binned_flux, 1, BestDecayCoupling(1, 1000, 574))
    axion_gen_001 = runGenerator(binned_flux, 0.1, BestDecayCoupling(0.1, 1000, 574))
    axion_gen_100 = runGenerator(binned_flux, 10, BestDecayCoupling(10, 1000, 574))
    axion_gen_1000 = runGenerator(binned_flux, 500, BestDecayCoupling(500, 2000, 574))

    # Print sum of weights
    print(np.sum(axion_gen.scatter_axion_weight))
    print(np.sum(axion_gen_001.scatter_axion_weight))
    print(np.sum(axion_gen_100.scatter_axion_weight))
    print(np.sum(axion_gen_1000.scatter_axion_weight))
    print(np.sum(axion_gen.decay_axion_weight))
    print(np.sum(axion_gen_001.decay_axion_weight))
    print(np.sum(axion_gen_100.decay_axion_weight))
    print(np.sum(axion_gen_1000.decay_axion_weight))
    

    # DECLARE CONSTANTS
    axion_angles = axion_gen.axion_angle
    axion_energies = axion_gen.axion_energy
    axion_angles = [180*x/np.pi for x in axion_angles]
    axion_angles_001 = axion_gen_001.axion_angle
    axion_energies_001 = axion_gen_001.axion_energy
    axion_angles_001 = [180*x/np.pi for x in axion_angles_001]
    axion_angles_100 = axion_gen_100.axion_angle
    axion_energies_100 = axion_gen_100.axion_energy
    axion_angles_100 = [180*x/np.pi for x in axion_angles_100]
    axion_angles_1000 = axion_gen_1000.axion_angle
    axion_energies_1000 = axion_gen_1000.axion_energy
    axion_angles_1000 = [180*x/np.pi for x in axion_angles_1000]

    deltatheta = axion_gen.gamma_sep_angle
    deltatheta = [180*x/np.pi for x in deltatheta]
    deltatheta_001 = axion_gen_001.gamma_sep_angle
    deltatheta_001 = [180*x/np.pi for x in deltatheta_001]
    deltatheta_100 = axion_gen_100.gamma_sep_angle
    deltatheta_100 = [180*x/np.pi for x in deltatheta_100]
    deltatheta_1000 = axion_gen_1000.gamma_sep_angle
    deltatheta_1000 = [180*x/np.pi for x in deltatheta_1000]
    
    # Plot the photon angular separation
    angle_edges = np.linspace(0, 45, 40)
    plt.hist(deltatheta, density=True, bins=angle_edges, histtype="step", label=r"$m_a = 10$ MeV")
    plt.hist(deltatheta_001, density=True, bins=angle_edges, histtype="step",  ls=":", label=r"$m_a = 100$ keV")
    plt.hist(deltatheta_100, density=True, bins=angle_edges, histtype="step",  ls="-.", label=r"$m_a = 1$ GeV")
    plt.hist(deltatheta_1000, density=True, bins=angle_edges, histtype="step",  ls="--", label=r"$m_a = 10$ GeV")
    plt.xlabel(r"$\Delta\theta_{\gamma\gamma}$ [deg]", fontsize=15)
    plt.title(r"$a\to\gamma\gamma$", loc="right", fontsize=15)
    plt.legend(loc='upper right', framealpha=1.0, fontsize=15)
    plt.yscale('log')
    #plt.xlim((0,45))
    plt.show()
    plt.close()



    # Plot photon energies from ALP decays at detector
    density=True
    alp_energy_edges = np.linspace(0,80,80)
    plt.hist(np.array(axion_gen.axion_energy)/1000, weights=axion_gen.decay_axion_weight, bins=alp_energy_edges,
             histtype='step', label=r"$m_a = 1$ MeV", density=density)
    plt.hist(np.array(axion_gen_001.axion_energy)/1000, weights=axion_gen_001.decay_axion_weight, bins=alp_energy_edges,
             histtype='step', ls=":", label=r"$m_a = 100$ keV", density=density)
    plt.hist(np.array(axion_gen_100.axion_energy)/1000, weights=axion_gen_100.decay_axion_weight, bins=alp_energy_edges,
             histtype='step', ls="dashed", label=r"$m_a = 10$ MeV", density=density)
    plt.hist(np.array(axion_gen_1000.axion_energy)/1000, weights=axion_gen_1000.decay_axion_weight, bins=alp_energy_edges,
             histtype='step',  ls="-.", label=r"$m_a = 0.5$ GeV", density=density)
    plt.title(r"$a \to \gamma \gamma$", loc="right")
    plt.xlabel(r"$E_{\gamma_1} + E_{\gamma_2}$ [GeV]", fontsize=15)
    plt.ylabel("a.u.", fontsize=15)
    plt.yscale('log')
    #plt.ylim((1e-6, 10))
    #plt.xlim((0,50))
    plt.legend(loc="upper right", framealpha=1.0, fontsize=15)
    plt.show()
    plt.clf()


    
    # Plot axion energies at detector from primakoff scattering
    density=True
    alp_energy_edges = np.linspace(0,80,80)
    plt.hist(np.array(axion_gen_001.axion_energy)/1000, weights=axion_gen_001.scatter_axion_weight, bins=alp_energy_edges,
             histtype='step', label=r"$m_a = 100$ keV", density=density)
    plt.hist(np.array(axion_gen.axion_energy)/1000, weights=axion_gen.scatter_axion_weight, bins=alp_energy_edges,
             histtype='step',  ls=":", label=r"$m_a = 1$ MeV", density=density)
    plt.hist(np.array(axion_gen_100.axion_energy)/1000, weights=axion_gen_100.scatter_axion_weight, bins=alp_energy_edges,
             histtype='step', ls="-.", label=r"$m_a = 10$ MeV", density=density)
    plt.hist(np.array(axion_gen_1000.axion_energy)/1000, weights=axion_gen_1000.scatter_axion_weight, bins=alp_energy_edges,
             histtype='step', ls="--", label=r"$m_a = 500$ MeV", density=density)
    plt.title(r"$a Z \to \gamma Z$", loc="right")
    #plt.xlabel(r"$E_\gamma$ [MeV]", fontsize=15)
   #plt.ylabel("a.u.", fontsize=15)
    plt.yscale('log')
    plt.legend(loc="upper right", fontsize=15)
    plt.show()
    plt.close()
    


    # Plot the axion angular distribution
    angle_edges = np.linspace(0, 180, 100)
    plt.hist(axion_angles, bins=angle_edges, density=True, histtype="step", label=r"1 MeV")
    plt.hist(axion_angles_001, bins=angle_edges, ls=":", density=True, histtype="step", label=r"100 keV")
    plt.hist(axion_angles_100, bins=angle_edges,  ls="-.", density=True, histtype="step", label=r"100 MeV")
    plt.hist(axion_angles_1000, bins=angle_edges,  ls="--", density=True, histtype="step", label=r"500 MeV")
    plt.hist(180*gamma_theta/np.pi, weights=gamma_wgt, bins=angle_edges, density=True, histtype="step", label=r"$\gamma$ in target (Pythia8)")
    #plt.xlabel(r"$\theta_z$ [deg]", fontsize=15)
    plt.title(r"$m_a = 5$ MeV", loc="right", fontsize=15)
    #plt.ylim(bottom=0)
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.close()



    
    
    
    # show gamma distribution
    plt.hist2d(binned_flux[:,1], binned_flux[:,2], weights=binned_flux[:,0], range=[[0,8000],[0,1.5]], bins=[energy_bins, theta_bins])
    plt.xlim((0,8000))
    plt.ylim((0,1.5))
    plt.show()

    plt.hist2d(gamma_e, gamma_theta, weights=gamma_wgt, range=[[0,8000],[0,1.5]], bins=200)
    plt.xlabel(r"$E_\gamma$ [MeV]")
    plt.ylabel(r"$\theta$ [rad]")
    plt.title("pythia8 photons from POT (hard process)", loc="right")
    plt.close()
    


if __name__ == "__main__":
    main()