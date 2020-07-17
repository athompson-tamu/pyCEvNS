import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal

from pyCEvNS.axion import IsotropicAxionFromPrimakoff

from matplotlib.pylab import rc
import matplotlib.ticker as tickr


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


class MinerEstimate:
  def __init__(self, photon_rates, axion_mass, axion_coupling, target_mass, target_z, target_photon_cross,
               detector_distance, min_decay_length):
    self.photon_rates = photon_rates  # per second
    self.axion_mass = axion_mass  # MeV
    self.axion_coupling = axion_coupling  # MeV^-1
    self.target_mass = target_mass  # MeV
    self.target_z = target_z
    self.target_photon_cross = target_photon_cross  # cm^2
    self.detector_distance = detector_distance  # meter
    self.min_decay_length = min_decay_length
    self.photon_energy = []
    self.photon_weight = []
    self.axion_energy = []
    self.axion_weight = []
    self.simulate()

  def form_factor(self):
    # place holder for now, included in the cross section
    return 1

  def primakoff_production_xs(self, z, a):
    me = 0.511
    prefactor = (1 / 137 / 4) * (self.axion_coupling ** 2)
    return prefactor * ((z ** 2) * (np.log(184 * np.power(z, -1 / 3)) + np.log(403 * np.power(a, -1 / 3) / me))
                        + z * np.log(1194 * np.power(z, -2 / 3)))

  def primakoff_scattering_xs(self, energy, z):
    if energy < 1.5 * self.axion_mass:
      return 0

    beta = np.sqrt(energy ** 2 - self.axion_mass ** 2) / energy
    chi = 1
    prefactor = (1 / 137 / 2) * (self.axion_coupling * z) ** 2
    return prefactor * chi * (1 / beta) * ((1 + beta ** 2) / (2 * beta) * np.log((1 + beta) / (1 - beta)) - 1)

  def branching_ratio(self):
    cross_prim = self.primakoff_production_xs(self.target_z, 2 * self.target_z)
    #print(cross_prim, self.target_photon_cross / (100 * meter_by_mev) ** 2)
    return cross_prim / (cross_prim + (self.target_photon_cross / (100 * meter_by_mev) ** 2))

  def simulate_single(self, energy, rate):
    if energy < self.axion_mass:
      return
    prob = self.branching_ratio()
    axion_p = np.sqrt(energy ** 2 - self.axion_mass ** 2)
    axion_v = axion_p / energy
    axion_boost = energy / self.axion_mass
    tau = 64 * np.pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost  # lifetime for a -> gamma gamma
    decay_length = meter_by_mev * axion_v * tau
    decay_prob =  1 - np.exp(-self.min_decay_length / meter_by_mev / axion_v / tau)
    decay_in_detector = 1 - np.exp(-(self.detector_distance-self.min_decay_length) / meter_by_mev / axion_v / tau)
    self.photon_energy.append(energy/2)
    self.photon_weight.append(rate * prob * (1-decay_prob) * decay_in_detector
                                / (4 * np.pi * (self.detector_distance - self.min_decay_length) ** 2))
    self.axion_energy.append(energy)
    self.axion_weight.append((1 - decay_prob) * rate * prob)

  # Loops over photon flux and fills the photon and axion energy arrays.
  def simulate(self):
    self.photon_energy = []
    self.photon_weight = []
    self.axion_energy = []
    self.axion_weight = []
    for f in self.photon_rates:
      self.simulate_single(f[0], f[1])

  def photon_events(self, detector_area, detection_time, threshold):
    res = 0
    for i in range(len(self.photon_energy)):
      if self.photon_energy[i] >= threshold:
        res += self.photon_weight[i]
    return res * detection_time * detector_area

  def scatter_events(self, detector_number, detector_z, detection_time, threshold):
    res = 0
    for i in range(len(self.axion_energy)):
      if self.axion_energy[i] >= threshold:
        res += self.axion_weight[i] * self.primakoff_scattering_xs(self.axion_energy[i], detector_z)
    return res * meter_by_mev ** 2 * detection_time * detector_number / (
          4 * np.pi * self.detector_distance ** 2)

  def photon_events_binned(self, detector_area, detection_time, threshold):
    res = np.zeros(len(self.photon_weight))
    scale = detection_time * detector_area / (4 * np.pi * self.detector_distance ** 2)
    for i in range(len(self.photon_energy)):
      if self.photon_energy[i] >= threshold:
        res[i] = self.photon_weight[i]
    return res * scale

  def scatter_events_binned(self, detector_number, detector_z, detection_time, threshold):
    res = np.zeros(len(self.axion_weight))
    for i in range(len(self.axion_energy)):
      if self.axion_energy[i] >= threshold:
        res[i] = self.axion_weight[i] * self.primakoff_scattering_xs(self.axion_energy[i], detector_z)
    return res * meter_by_mev ** 2 * detection_time * detector_number / (
          4 * np.pi * self.detector_distance ** 2)


# Declare global constants.
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600*24

# Flux
miner = np.genfromtxt('data/miner/reactor_photon.txt')
miner[:,1] *= 100000000  # factor to get flux at the core
mw_flux = miner
gw_flux = miner
gw_flux[:,1] *= 3900 # convert MW reactor to GW


# Given an axion flux generator and detector information, scans over the m_a grid
# and performs a binary search at each m_a value to find the solution of g_a\gamma\gamma
# that satisfies the Poisson test-statistic (2 sigma limit)
def BinarySearch(flux, detector, mass_grid, save_file, sig_limit):
    print("--- Running ", detector)
    if detector == "ge":
        # Ge
        det_dis = 2.25
        det_mass = 4
        det_am = 65.13e3
        det_z = 32
        days = 1000
        det_area = 0.2 ** 2
        det_thresh = 1e-3
        bkg_dru = 100
    if detector == "csi":
        det_dis = 2.5
        det_mass = 200
        det_am = 123.8e3
        det_z = 55
        days = 1000
        det_area = (0.4) ** 2
        det_thresh = 2.6
        bkg_dru = 100
    if detector == "csi_2ton":
        det_dis = 2.5
        det_mass = 2000
        det_am = 123.8e3
        det_z = 55
        days = 1000
        det_area = 4*(0.4) ** 2
        det_thresh = 2.6
        bkg_dru = 100
    if detector == "connie":
        det_dis = 30
        det_mass = 0.1
        det_am = 65.13e3 / 2
        det_z = 14
        days = 1000
        det_area = 4*0.0036
        det_thresh = 0.028e-3
        bkg_dru = 700
    if detector == "conus":
        det_dis = 17
        det_mass = 4
        det_am = 65.13e3
        det_z = 32
        days = 1000
        det_area = 0.2 ** 2
        det_thresh = 1e-3
        bkg_dru = 100
    if detector == "nucleus":
        det_dis = 40
        det_mass = 0.01
        det_am = 65.13e3 * 3
        det_z = 51
        days = 1000
        det_area = 0.005 ** 2
        det_thresh = 0.02e-3
        bkg_dru = 100

    bkg = bkg_dru * days * det_mass * 5  # DRU within 0-5 keVnr
    upper_array = np.ones_like(mass_grid)  # Coupling array to test upper bound
    lower_array = np.ones_like(mass_grid)  # Coupling array to test lower bound
    
    #generator.detector_distance = det_dis
    #generator.min_decay_length = 0.9*det_dis
    #generator.photon_rates = flux
    #axion_gen = IsotropicAxionFromPrimakoff(photon_rates=mw_flux, axion_mass=1, axion_coupling=1e-6, target_mass=240e3,
     #                                       target_z=90, target_photon_cross=15e-24, detector_distance=det_dis,
      #                                      detector_length=np.sqrt(det_area))
    axion_gen = MinerEstimate(flux, 1, 1e-6, 240e3, 90, 15e-24, det_dis, 0.9*det_dis)

    # Begin the scan
    for i in range(mass_grid.shape[0]):
        lo = -12
        hi = -1
        axion_gen.axion_mass = mass_grid[i]
        print("Evaluating likelihood at m_a = ", mass_grid[i])
        # lower bound
        while hi - lo > 0.002:
            mid = (hi+lo)/2
            axion_gen.axion_coupling = 10**mid
            axion_gen.simulate()
            ev = axion_gen.photon_events(det_area, days*s_per_day, det_thresh)
            ev += axion_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            sig = ev / np.sqrt(ev + bkg)
            if sig < sig_limit:
                lo = mid
            else:
                hi = mid
        lower_array[i] = 10**mid
        print(10**mid)
        lo = -12
        hi = -1
        # upper bound
        while hi - lo > 0.002:
            mid = (hi+lo)/2
            axion_gen.axion_coupling = 10**mid
            axion_gen.simulate()
            ev = axion_gen.photon_events(det_area, days*s_per_day, det_thresh)
            ev += axion_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, det_thresh)
            sig = ev / np.sqrt(ev + bkg)
            if sig > sig_limit:
                lo = mid
            else:
                hi = mid
        upper_array[i] = 10**mid
        print(10**mid)

    save_array = np.array([mass_grid, lower_array, upper_array])
    save_array = np.transpose(save_array)
    np.savetxt(save_file, save_array)
    return save_array


def main():  
    #axion_gen = IsotropicAxionFromPrimakoff(photon_rates=mw_flux, axion_mass=1, axion_coupling=1e-6, target_mass=240e3,
     #                                       target_z=90, target_photon_cross=15e-24, detector_distance=2.25,
      #                                      detector_length=0.2)
    #axion_gen = MinerEstimate(mw_flux, 1, 1e-6, 240e3, 90, 15e-24, 2.25, 0.2)
    mass_array = np.logspace(-6, 1, 100)
    rerun = True
    if rerun == True:
        limits_miner = BinarySearch(mw_flux, "ge", mass_array, "limits/miner_photon/ge.txt", 2)
        limits_connie = BinarySearch(gw_flux, "connie", mass_array, "limits/miner_photon/connie_repro.txt", 2)
        limits_conus = BinarySearch(gw_flux, "conus", mass_array, "limits/miner_photon/conus_repro.txt", 2)
        limits_nucleus = BinarySearch(gw_flux, "nucleus", mass_array, "limits/miner_photon/nucleus_repro.txt", 2)

    else:
        limits_miner = np.genfromtxt("limits/miner_photon/ge.txt")
        limits_connie = np.genfromtxt("limits/miner_photon/connie.txt")
        limits_conus = np.genfromtxt("limits/miner_photon/conus.txt")
        limits_nucleus = np.genfromtxt("limits/miner_photon/nucleus.txt")

    # Existing limits
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
    
    # Plot derived limits (lower bound)
    plt.plot(limits_miner[0]*1e6, limits_miner[1]*1e3, color="crimson", label='MINER Ge (4 kg)')
    plt.plot(limits_nucleus[0]*1e6, limits_nucleus[1]*1e3, color="navy", ls='dashed', label=r'NUCLEUS CaWO$_4$(Al$_2$O$_3$) (0.01 kg)')
    plt.plot(limits_connie[0]*1e6, limits_connie[1]*1e3, color="orange",ls='dashdot',label='CONNIE Si Skipper CCD (0.1 kg)')
    plt.plot(limits_conus[0]*1e6, limits_conus[1]*1e3, color="teal",ls='dotted', label='CONUS Ge PPC (4 kg)')
    
    # Plot derived limits (upper bound)
    plt.plot(limits_miner[0]*1e6, limits_miner[2]*1e3, color="crimson")
    plt.plot(limits_nucleus[0]*1e6, limits_nucleus[2]*1e3, color="navy", ls='dashed')
    plt.plot(limits_connie[0]*1e6, limits_connie[2]*1e3, color="orange",ls='dashdot')
    plt.plot(limits_conus[0]*1e6, limits_conus[2]*1e3, color="teal",ls='dotted')

    # Plot astrophysical limits
    astro_alpha = 0.1
    plt.fill(hbstars[:,0]*1e9, hbstars[:,1]*0.367e-3, color="mediumpurple", alpha=astro_alpha)
    plt.fill(cast[:,0]*1e9, cast[:,1]*0.367e-3, color="orchid", alpha=astro_alpha)
    plt.fill_between(sn1987a_lower[:,0]*1e9, y1=sn1987a_lower[:,1]*0.367e-3, y2=sn1987a_upper[:,1]*0.367e-3, color="darkgoldenrod", alpha=astro_alpha)

    # Plot lab limits
    lab_alpha=0.9
    plt.fill(beam[:,0], beam[:,1], color="b", alpha=lab_alpha)
    plt.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
            color="orange", alpha=lab_alpha)
    plt.fill(lep[:,0], lep[:,1], color="green", alpha=lab_alpha)
    plt.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
            color="yellow", alpha=lab_alpha)


    # Draw text for existing backgrounds
    text_fs = 16
    plt.text(300,5e-6,'HB Stars', rotation=0, fontsize=text_fs, color="k", weight="bold")
    plt.text(1,6e-6,'CAST', rotation=0, fontsize=text_fs, color="k", weight="bold")
    plt.text(10000,8e-7,'SN1987a', rotation=0, fontsize=text_fs, color="k", weight="bold")
    plt.text(1e5,1e-4,'Beam Dump', rotation=0, fontsize=text_fs, color="white", weight="bold")
    plt.text(1e4,1e-3,r'$e^+e^-\rightarrow inv.+\gamma$', rotation=0, fontsize=text_fs, color="white", weight="bold")
    plt.text(1e7,1e-1,'LEP', rotation=0, fontsize=text_fs, color="white", weight="bold")
    plt.text(5,1e-3,'NOMAD', rotation=0, fontsize=text_fs, color="k", weight="bold")

    plt.legend(loc="upper left", framealpha=1, fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(1,5e7)
    plt.ylim(3e-7,1)
    plt.xlabel('$m_a$ [eV]', fontsize=24)
    plt.ylabel('$g_{a\gamma\gamma}$ [GeV$^{-1}$]', fontsize=24)
    plt.tight_layout()



    plt.tick_params(axis='x', which='minor')
    plt.show()


if __name__ == "__main__":
    main()