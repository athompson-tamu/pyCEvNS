import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.special import exp1

from pyCEvNS.axion import MinerAxionElectron, MinerAxionPhoton

from matplotlib.pylab import rc
import matplotlib.ticker as tickr


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)




# Read in data.
beam = np.genfromtxt('data/beam.txt')
eeinva = np.genfromtxt('data/eeinva.txt')
lep = np.genfromtxt('data/lep.txt')
lsw = np.genfromtxt('data/lsw.txt')
nomad = np.genfromtxt('data/nomad.txt')

# Astrophyiscal limits
cast = np.genfromtxt("data/cast.txt", delimiter=",")
hbstars = np.genfromtxt("data/hbstars.txt", delimiter=",")
sn1987a_upper = np.genfromtxt("data/sn1987a_upper.txt", delimiter=",")
sn1987a_lower = np.genfromtxt("data/sn1987a_lower.txt", delimiter=",")

# Declare constants.
# conversion between units
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600*24
pot_per_year = 1.1e21

# COHERENT flux used 100,000 POT (1e-11 s equivalent)
#pot_sample = 100000
#scale = pot_per_year / pot_sample / 365
#flux = np.genfromtxt('data/dune/nu_spectra_nd.txt', delimiter=",")
#flux[:, 1] *= scale
#flux[:, 0] *= 1000


# Flux from pythia8 (Doojin)
pot_sample = 10000
scale = pot_per_year / pot_sample / 365 / 24 / 3600
gamma_data = np.genfromtxt("data/dune/hepmc_gamma_flux_from_pi0.txt")
gamma_e = gamma_data[:,0]
gamma_wgt = scale * np.ones_like(gamma_e)
flux = np.array([gamma_e, gamma_wgt])
flux = flux.transpose()
print(flux)




mass_array = np.logspace(-6, 5, 200)  # Mass array to test

def BinarySearch():
    det_dis = 304  # from dump area to ND
    det_mass = 50000
    det_am = 37.211e3
    det_z = 18
    days = 1000
    det_area = 3*6
    det_thresh = 0.028e-3
    bkg_dru = 1

    sig_limit = 2
    bkg = bkg_dru * days * det_mass

    coupling_array = np.ones_like(mass_array)  # Coupling array to test
    # Assume soil/rock dump area/target, so z=6
    # DUNE ND is 3x6 facing the beam, with 2m or so long. Additional detectors behind. Assume total detector length = 6m?
    photon_gen = MinerAxionPhoton(photon_rates=flux, axion_mass=1, axion_coupling=1e-6, target_mass=28e3,
                                  target_z=14, target_photon_cross=1e-24, detector_distance=det_dis,
                                  detector_length=6)
    # Axion decay regime
    print("starting scan...")
    for i in range(mass_array.shape[0]):
        lo = -40
        hi = 5
        photon_gen.axion_mass = mass_array[i]
        print("scan for m_a = ", mass_array[i])
        while hi - lo > 0.005:
            mid = (hi+lo)/2
            photon_gen.axion_coupling = 10**mid
            photon_gen.simulate()
            ev = photon_gen.photon_events(det_area, days, det_thresh) * s_per_day
            ev += photon_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days, det_thresh) * s_per_day
            sig = np.sqrt(ev)
            if sig < sig_limit:
                lo = mid
            else:
                hi = mid
        coupling_array[i] = 10**mid

    np.savetxt("limits/DUNE/preliminary_limits.txt", coupling_array)
    return coupling_array



rerun = True
if rerun == True:
  coup_array = BinarySearch()

else:
  coup_array = np.genfromtxt("limits/DUNE/preliminary_limits.txt")







plt.plot(mass_array*1e6, coup_array*1e3, color="crimson", label='DUNE ND')


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
plt.ylim(1e-11,1e-1)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('$m_a$ [eV]', fontsize=15)
plt.ylabel('$g_{a\gamma\gamma}$ [GeV$^{-1}$]', fontsize=15)

plt.tick_params(axis='x', which='minor')

plt.show()


