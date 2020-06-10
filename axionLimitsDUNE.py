import sys

import numpy as np
from numpy import log, log10, sqrt, pi
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve, fmin_tnc

from pyCEvNS.axion import PrimakoffAxionFromBeam, IsotropicAxionFromPrimakoff

from matplotlib.pylab import rc
import matplotlib.ticker as tickr
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
days = 10*364  # days of exposure
det_area = 3*7  # cross-sectional det area
det_thresh = 0.028e-3  # energy threshold
sig_limit = 1.0  # poisson significance (2 sigma)



def grad_desc(f, w0, delta0, alpha):
    # Set up empty list of points w and functions f(w)
    fs = []
    ws = []

    # Stopping criterion
    crit = 0.1  # distance from the minimum of the function (0)
    
    # Get the first two points
    fs.extend([f(w0), f(w0+delta0)])
    ws.extend([w0, w0+delta0])
    
    # Get the first derivative of the first 2 points
    delta_f0 = (fs[1] - fs[0])/(ws[1] - ws[0])
    k = 2

    # Append the 2nd point using the gradient descent step
    ws.append(ws[1] - alpha*delta_f0)  # w_2
    print("k=2, ", ws, fs)
    while abs(fs[k-1]) > crit:
        print("k = ", k)
        fs.append(f(ws[k]))  # append f_k
        delta_fk = (fs[k] - fs[k-1])/(ws[k] - ws[k-1])
        ws.append(ws[k] - alpha*delta_fk)
        print("deltafk = ", alpha*delta_fk)
        print("(ws, fs) = (%0.3f, %0.3f)" % (ws[k], fs[k]))
        k += 1
    
    return ws[-2]


def GradientDescent(generator, mass_array, g_array, save_file):
    upper_array = np.zeros_like(mass_array)
    lower_array = np.ones_like(mass_array)
    print("starting scan...")
    for i in range(mass_array.shape[0]):
        generator.axion_mass = mass_array[i]

        def EventsGenerator(logg):
            generator.axion_coupling = 10.0**logg
            generator.simulate()
            ev = generator.decay_events(days*s_per_day, det_thresh)
            ev += generator.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            return 2-sqrt(ev)
        
        # Lower limit:
        g0_lower = -10
        g_value = grad_desc(EventsGenerator, g0_lower, 1.2, 10)
        lower_array[i] = g_value


    limits_array = [mass_array, lower_array, upper_array]
    np.savetxt(save_file, limits_array)
    return limits_array




def SandwichSearch(generator, mass_array, g_array, save_file, weight=1.0):
    print(mass_array)
    upper_array = np.zeros_like(mass_array)
    lower_array = np.ones_like(mass_array)
    print("starting scan...")
    for i in range(mass_array.shape[0]):
        generator.axion_mass = mass_array[i]
        print("\n **** SETTING ALP MASS = %0.6f MeV **** \n" % mass_array[i])

        
        # lower bound
        print(" *********** scanning lower bound...")
        for g in g_array:
            generator.axion_coupling = g
            generator.simulate()
            ev = generator.decay_events(days*s_per_day, det_thresh)
            ev += generator.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            sig = sqrt(ev*weight)
            if sig > sig_limit:
                print("FOUND DELTA CHI2 = %0.6f at g=%0.9f MeV^-1" % (sig, g))
                lower_array[i] = g
                break
        
        # upper bound
        print(" ********** scanning upper bound...")
        for g in g_array[::-1]:
            generator.axion_coupling = g
            generator.simulate()
            ev = generator.decay_events(days*s_per_day, det_thresh)
            ev += generator.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            sig = sqrt(ev*weight)
            if sig > sig_limit:
                print("FOUND DELTA CHI2 = %0.6f at g=%0.9f MeV^-1" % (sig, g))
                upper_array[i] = g
                break


    limits_array = [mass_array, lower_array, upper_array]
    np.savetxt(save_file, limits_array)
    return limits_array


def main(flux_file, save_dir, show_plots):
    # Declare event generators.
    flux = np.genfromtxt(flux_file)
    # 5% POT from beam dump
    axion_gen = PrimakoffAxionFromBeam(photon_rates=flux, target_mass=12e3, target_z=6,
                                       target_photon_cross=1e-24, detector_distance=574,
                                       detector_length=10, detector_area=21)  # not sure about area and length

    target_generator = IsotropicAxionFromPrimakoff(photon_rates=flux, target_mass=12e3, target_z=6,
                                                   target_photon_cross=1e-24, detector_distance=574,
                                                   detector_length=14, detector_area=det_area)
    dump_generator = IsotropicAxionFromPrimakoff(photon_rates=flux, target_mass=28e3, target_z=14,
                                                   target_photon_cross=1e-24, detector_distance=304,
                                                   detector_length=14, detector_area=det_area)
 
    
    # Run the scan.
    mass_array = np.logspace(-2, 4, 100)
    g_array = np.logspace(-13, -3, 80)
    save_file = save_dir
    rerun = True
    if rerun == True:
      print("Rerunning limits")
      limits_target = SandwichSearch(axion_gen, mass_array, g_array, save_file)
    else:
      limits_target = np.genfromtxt(save_file)


   
    upper_limit_target = limits_target[2]
    lower_limit_target = limits_target[1]
    
    
    print(show_plots)
    
    # Plotting.
    if show_plots:
    
        # TARGET
        # Find where the upper and lower arrays intersect at the tongue and clip

        diff_upper_lower = upper_limit_target - lower_limit_target
        upper_limit_target = np.delete(upper_limit_target, np.where(diff_upper_lower < 0))
        lower_limit_target = np.delete(lower_limit_target, np.where(diff_upper_lower < 0))
        mass_array_target = np.delete(mass_array, np.where(diff_upper_lower < 0))
        
        # join upper and lower bounds
        joined_limits_target = np.append(lower_limit_target, upper_limit_target[::-1])
        joined_masses_target = np.append(mass_array_target, mass_array_target[::-1])
        
        """
        # original scan
        original_limits = np.genfromtxt("limits/DUNE/isotropic_limits_original.txt", delimiter=',')
        joined_limits_target = original_limits[:,1]
        joined_masses_target = original_limits[:,0]
        """

        # Read in data.
        beam = np.genfromtxt('data/existing_limits/beam.txt')
        eeinva = np.genfromtxt('data/existing_limits/eeinva.txt')
        lep = np.genfromtxt('data/existing_limits/lep.txt')
        lsw = np.genfromtxt('data/existing_limits/lsw.txt')
        nomad = np.genfromtxt('data/existing_limits/nomad.txt')

        # Astrophyiscal limits
        cast = np.genfromtxt("data/existing_limits/cast.txt", delimiter=",")
        hbstars = np.genfromtxt("data/existing_limits/hbstars.txt", delimiter=",")
        sn1987a_upper = np.genfromtxt("data/existing_limits/sn1987a_upper.txt", delimiter=",")
        sn1987a_lower = np.genfromtxt("data/existing_limits/sn1987a_lower.txt", delimiter=",")

        plt.plot(joined_masses_target*1e6, joined_limits_target*1e3, color="crimson", label='DUNE ND (target)')
        #plt.plot(joined_masses_dump*1e6, joined_limits_dump*1e3, color="crimson", ls='dashed', label='DUNE ND (dump)')


        # Plot astrophysical limits
        plt.fill(hbstars[:,0]*1e9, hbstars[:,1]*0.367e-3, label="HB Stars", color="mediumpurple", alpha=0.3)
        plt.fill(cast[:,0]*1e9, cast[:,1]*0.367e-3, label="CAST", color="orchid", alpha=0.3)
        plt.fill_between(sn1987a_lower[:,0]*1e9, y1=sn1987a_lower[:,1]*0.367e-3, y2=sn1987a_upper[:,1]*0.367e-3,
                        label="SN1987a", color="lightsteelblue", alpha=0.3)


        # Plot lab limits
        plt.fill(beam[:,0], beam[:,1], label='Beam Dump', color="b", alpha=0.7)
        plt.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
                color="orange", label=r'$e^+e^-\rightarrow inv.+\gamma$', alpha=0.7)
        plt.fill(lep[:,0], lep[:,1], label='LEP', color="green", alpha=0.7)
        plt.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
                color="yellow", label='NOMAD', alpha=0.7)


        plt.legend(loc="lower left", framealpha=1, ncol=2, fontsize=9)
        plt.title(r"Primakoff Scattering and $a\to\gamma\gamma$, 50t fiducial mass, 1000 days exposure", loc="right")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim((1,1e10))
        plt.ylim(1e-13,1e-1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('$m_a$ [eV]', fontsize=15)
        plt.ylabel('$g_{a\gamma\gamma}$ [GeV$^{-1}$]', fontsize=15)

        plt.tick_params(axis='x', which='minor')

        plt.show()


if __name__ == "__main__":
    main(flux_file=str(sys.argv[1]), save_dir=str(sys.argv[2]), show_plots=sys.argv[3])

