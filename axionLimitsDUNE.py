import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.special import exp1

from pyCEvNS.axion import IsotropicAxionFromCompton, IsotropicAxionFromPrimakoff

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

# Flux from pythia8 (Doojin)
pot_sample = 10000
scale = pot_per_year / pot_sample / 365 / 24 / 3600
gamma_data = np.genfromtxt("data/dune/hepmc_gamma_flux_from_pi0.txt")
gamma_e = 1000*gamma_data[:,0] # convert to mev

# bin coarse flux
flux_edges = np.linspace(min(gamma_e), max(gamma_e), 25)
flux_bins = (flux_edges[:-1] + flux_edges[1:]) / 2
flux_hist = np.histogram(gamma_e, weights=scale*np.ones_like(gamma_e), bins=flux_bins)[0]
flux = np.array(list(zip(flux_bins,flux_hist)))

# detector constants
det_dis = 304  # from dump area to ND
det_mass = 50000
det_am = 37.211e3
det_z = 18
days = 1000
det_area = 3*6
det_thresh = 0.028e-3
bkg_dru = 1
sig_limit = 2.0
bkg = bkg_dru * days * det_mass




def SandwichSearch(generator, mass_array, g_array):
    upper_array = np.zeros_like(mass_array)
    lower_array = np.ones_like(mass_array)
    print("starting scan...")
    for i in range(mass_array.shape[0]):
        generator.axion_mass = mass_array[i]
        # lower bound
        for g in g_array:
            generator.axion_coupling = g
            generator.simulate()
            ev = generator.photon_events(det_area, days*s_per_day, det_thresh)
            ev += generator.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            ev *= 4*np.pi*det_dis**2
            sig = np.sqrt(ev)
            if sig > sig_limit:
                lower_array[i] = g
                break

        # upper bound
        for g in g_array[::-1]:
            generator.axion_coupling = g
            generator.simulate()
            ev = generator.photon_events(det_area, days*s_per_day, det_thresh)
            ev += generator.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            ev *= 4*np.pi*det_dis**2
            sig = np.sqrt(ev)
            if sig > sig_limit:
                upper_array[i] = g
                break


    limits_array = [mass_array, lower_array, upper_array]
    np.savetxt("limits/DUNE/preliminary_limits.txt", limits_array)
    return limits_array


def BinarySearch():
    # Assume soil/rock dump area/target, so z=6
    # DUNE ND is 3x6 facing the beam, with 2m or so long. Additional detectors behind. Assume total detector length = 6m?
    photon_gen = IsotropicAxionFromPrimakoff(photon_rates=flux, axion_mass=1, axion_coupling=1e-6, target_mass=28e3,
                                             target_z=14, target_photon_cross=1e-24, detector_distance=det_dis,
                                             detector_length=6)
    
    upper_array = np.ones_like(mass_array)
    lower_array = np.zeros_like(mass_array)
    print("starting scan...")
    for i in range(mass_array.shape[0]):
        photon_gen.axion_mass = mass_array[i]
        # lower bound
        lo = -11
        hi = -3
        while hi - lo > 0.005:
            mid = (hi+lo)/2
            print("trying mid = ", mid)
            photon_gen.axion_coupling = 10**mid
            photon_gen.simulate()
            ev = photon_gen.photon_events(det_area, days*s_per_day, det_thresh)
            ev += photon_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            ev *= 4*np.pi*det_dis**2
            sig = np.sqrt(ev)
            print("sig = ", sig)
            if sig < sig_limit:
                lo = mid
                print("going higher")
            else:
                hi = mid
                print("going lower")
        upper_array[i] = 10**mid
        # upper bound
        lo = -11
        hi = -3
        while hi - lo > 0.005:
            mid = (hi+lo)/2
            photon_gen.axion_coupling = 10**mid
            photon_gen.simulate()
            ev = photon_gen.photon_events(det_area, days*s_per_day, det_thresh)
            ev += photon_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            ev *= 4*np.pi*det_dis**2
            sig = np.sqrt(ev)
            if sig > sig_limit:
                lo = mid
            else:
                hi = mid
        lower_array[i] = 10**mid

    limits_array = [mass_array, lower_array, upper_array]
    np.savetxt("limits/DUNE/preliminary_limits.txt", limits_array)
    return limits_array


def main():
    axion_gen_target = IsotropicAxionFromPrimakoff(photon_rates=flux, axion_mass=1, axion_coupling=1e-6,
                                                   target_mass=28e3, target_z=14, target_photon_cross=1e-24,
                                                   detector_distance=det_dis, detector_length=6)
    axion_gen_dump = IsotropicAxionFromPrimakoff(photon_rates=flux, axion_mass=1, axion_coupling=1e-6,
                                                 target_mass=28e3, target_z=14, target_photon_cross=1e-24,
                                                 detector_distance=det_dis, detector_length=6)

    mass_array = np.logspace(-6, 4, 100)
    g_array = np.logspace(-13, -3, 100)
    rerun = True
    if rerun == True:
      print("Rerunning limits")
      g_array = SandwichSearch(axion_gen_target, mass_array, g_array)
    else:
      g_array = np.genfromtxt("limits/DUNE/preliminary_limits.txt")


    upper_limit = g_array[2]
    lower_limit = g_array[1]
    
    # Find where the upper and lower arrays intersect at the tongue and clip
    diff_upper_lower = upper_limit - lower_limit
    upper_limit = np.delete(upper_limit, np.where(diff_upper_lower < 0))
    lower_limit = np.delete(lower_limit, np.where(diff_upper_lower < 0))
    mass_array = np.delete(mass_array, np.where(diff_upper_lower < 0))
    
    # join upper and lower bounds
    joined_limits = np.append(lower_limit, upper_limit[::-1])
    joined_masses = np.append(mass_array, mass_array[::-1])


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

    plt.plot(joined_masses*1e6, joined_limits*1e3, color="crimson", label='DUNE ND')
    #plt.plot(mass_array*1e6, lower_limit*1e3, color="crimson", label='DUNE ND')
    #plt.plot(mass_array*1e6, upper_limit*1e3, color="crimson")


    # Plot astrophysical limits
    plt.fill(hbstars[:,0]*1e9, hbstars[:,1]*0.367e-3, label="HB Stars", color="mediumpurple")
    plt.fill(cast[:,0]*1e9, cast[:,1]*0.367e-3, label="CAST", color="orchid")
    plt.fill_between(sn1987a_lower[:,0]*1e9, y1=sn1987a_lower[:,1]*0.367e-3, y2=sn1987a_upper[:,1]*0.367e-3, label="SN1987a", color="lightsteelblue")


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
    main()

