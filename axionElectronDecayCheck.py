import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from pyCEvNS.axion import IsotropicAxionFromCompton
from matplotlib.pylab import rc

import sys


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def main():
    det_dis = 2.25
    det_mass = 4
    det_am = 65.13e3
    det_z = 32
    days = 1000
    det_area = 0.2 ** 2
    det_thresh = 1e-3
    dru_limit = 0.1
    bkg_dru = 100
    
    # conversion between units
    hbar = 6.58212e-22  # MeV*s
    c_light = 2.998e8  # m/s
    meter_by_mev = hbar * c_light  # MeV*m
    mev_per_kg = 5.6095887e29  # MeV/kg
    s_per_day = 3600 * 24
    me = 0.511

    # axion parameters
    axion_mass = 1  # MeV
    axion_coupling = 1e-6


    miner_flux = np.genfromtxt('data/reactor_photon.txt')  # flux at reactor surface
    miner_flux[:, 1] *= 1e8  # get flux at the core
    generator = IsotropicAxionFromCompton(miner_flux, 2.2, 1e-6, 240e3, 90, 15e-24, det_dis, 0.1)
    generator.simulate()
    
    generator.axion_coupling = 3e-6
    print(generator.scatter_events(det_mass * mev_per_kg / det_am, det_z, days, det_thresh) * s_per_day)
    print(generator.pair_production_events(det_area, days, 0) * s_per_day)

    
    
    masses = np.linspace(2*me, 10*me, 100)
    couplings = np.logspace(-10,10, 100)
    p_decay = np.empty_like(couplings)
    for i in range(0, couplings.shape[0]):
        generator.axion_coupling = couplings[i]
        p_decay[i] = generator.AxionDecayProb(10)
        #print(p_decay[i])
    
    
    
    plt.plot(couplings, p_decay)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    
    
    

if __name__ == "__main__":
    main()