import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal

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

  # Primakoff cross-section for axion production
  #def photon_axion_cross(self, pgamma):
   # ma = self.axion_mass
    #it = 1 / (ma ** 2 * pgamma ** 2 - pgamma ** 4) + (ma ** 2 - 2 * pgamma ** 2) * np.arctanh(
     # 2 * pgamma * np.sqrt(-ma ** 2 + pgamma ** 2) / (ma ** 2 - 2 * pgamma ** 2)) / (
      #         2 * pgamma ** 3 * (-ma ** 2 + pgamma ** 2) ** 1.5)
    #return 1 / 4 * self.axion_coupling ** 2 * 1 / 137 * self.target_z ** 2 * (
     #     pgamma ** 2 - self.axion_mass ** 2) ** 2 * it * self.form_factor()

  # probability of axion production through primakoff
  #def axion_probability(self, pgamma):
    # target_n_gamma is in cm^2
    # assuming that target is thick enough and photon cross section is large enough that all photon is converted / absorbed
   # cross_prim = self.photon_axion_cross(pgamma)
    #return cross_prim / (cross_prim + (self.target_photon_cross / (100 * meter_by_mev) ** 2))

  # Calculate axion and photon populations.
  # get axion production from primakoff process, surviving population after decay probability to gamma gamma
  # Get photon population from a -> gamma gamma decay by convolving Primakoff photon loss with Axion to photon production
  def simulate_single(self, energy, rate):
    if energy < 1.5 * self.axion_mass \
        or np.abs(2*energy*np.sqrt(-self.axion_mass**2+energy**2)/(self.axion_mass**2-2*energy**2))>=1:
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



# Read in data.
coherent = np.genfromtxt('data/photon_flux_COHERENT_log_binned.txt')
# COHERENT flux used 100,000 POT (1e-11 s equivalent)
miner = np.genfromtxt('data/reactor_photon.txt')
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


miner[:, 1] *= 100000000
flux = miner

mass_array = np.logspace(-6, 2, 300)  # Mass array to test

def BinarySearch(detector):
    print("Running ", detector)
    if detector == "ge":
        # Ge
        det_dis = 2.25
        det_mass = 4
        det_am = 65.13e3
        det_z = 32
        days = 1000
        det_area = 0.2 ** 2
        det_thresh = 1e-3
        dru_limit = 0.1
        bkg_dru = 100
    if detector == "csi":
        det_dis = 2.5
        det_mass = 200
        det_am = 123.8e3
        det_z = 55
        days = 1000
        det_area = (0.4) ** 2
        det_thresh = 2.6
        dru_limit = 0.01
        bkg_dru = 100
    if detector == "csi_2ton":
        det_dis = 2.5
        det_mass = 2000
        det_am = 123.8e3
        det_z = 55
        days = 1000
        det_area = 4*(0.4) ** 2
        det_thresh = 2.6
        dru_limit = 0.01
        bkg_dru = 100
    if detector == "connie":
        det_dis = 30
        det_mass = 0.1
        det_am = 65.13e3 / 2
        det_z = 14
        days = 1000
        det_area = 4*0.0036
        det_thresh = 0.028e-3
        dru_limit = 0.01
        bkg_dru = 700
    if detector == "conus":
        det_dis = 17
        det_mass = 4
        det_am = 65.13e3
        det_z = 32
        days = 1000
        det_area = 0.2 ** 2
        det_thresh = 1e-3
        dru_limit = 0.01
        bkg_dru = 100
    if detector == "nucleus":
        det_dis = 40
        det_mass = 0.01
        det_am = 65.13e3 * 3
        det_z = 51
        days = 1000
        det_area = 0.005 ** 2
        det_thresh = 0.02e-3
        dru_limit = 0.01
        bkg_dru = 100

    sig_limit = 2
    bkg = bkg_dru * days * det_mass * 5  # DRU within 0-5 keVnr

    coupling_array = np.zeros_like(mass_array)  # Coupling array to test

    photon_gen = MinerEstimate(flux, 1, 1e-6, 240e3, 90, 15e-24, det_dis, 0.2)
    # Axion decay regime
    for i in range(mass_array.shape[0]):
        lo = -12
        hi = -1
        photon_gen.axion_mass = mass_array[i]
        print("trying ma = ", mass_array[i])
        while hi - lo > 0.001:
            mid = (hi+lo)/2
            photon_gen.axion_coupling = 10**mid
            photon_gen.simulate()
            ev = photon_gen.photon_events(det_area, days, det_thresh) * s_per_day
            ev += photon_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days, det_thresh) * s_per_day
            sig = ev / np.sqrt(ev + bkg)
            print("sig = ", sig)
            print("ev,bkg = ", ev, bkg)
            if sig < sig_limit:
                lo = mid
            else:
                hi = mid
        coupling_array[i] = 10**mid




    if detector == "ge":
      np.savetxt("limits/miner_photon/ge.txt", coupling_array)
    if detector == "csi":
      np.savetxt("limits/miner_photon/csi.txt", coupling_array)
    if detector == "csi_2ton":
      np.savetxt("limits/miner_photon/csi_2ton.txt", coupling_array)
    if detector == "connie":
      np.savetxt("limits/miner_photon/connie.txt", coupling_array)
    if detector == "conus":
      np.savetxt("limits/miner_photon/conus.txt", coupling_array)
    if detector == "nucleus":
      np.savetxt("limits/miner_photon/nucleus.txt", coupling_array)

    return coupling_array

rerun = False
if rerun == True:
  coup_array_1 = BinarySearch("ge")
  flux[:,1] *= 3900
  coup_array_4 = BinarySearch("connie")
  coup_array_5 = BinarySearch("conus")
  coup_array_6 = BinarySearch("nucleus")

else:
  coup_array_1 = np.genfromtxt("limits/miner_photon/ge.txt")
  #coup_array_2 = np.genfromtxt("limits/miner_photon/csi.txt")
  #coup_array_3 = np.genfromtxt("limits/miner_photon/csi_2ton.txt")
  coup_array_4 = np.genfromtxt("limits/miner_photon/connie.txt")
  coup_array_5 = np.genfromtxt("limits/miner_photon/conus.txt")
  coup_array_6 = np.genfromtxt("limits/miner_photon/nucleus.txt")







plt.plot(mass_array*1e6, coup_array_1*1e3, color="crimson", label='MINER Ge (4 kg)')
plt.plot(mass_array*1e6, coup_array_6*1e3, color="navy", ls='dashed', label=r'NUCLEUS CaWO$_4$(Al$_2$O$_3$) (0.01 kg)')
plt.plot(mass_array*1e6, coup_array_4*1e3, color="orange",ls='dashdot',label='CONNIE Si Skipper CCD (0.1 kg)')
plt.plot(mass_array*1e6, coup_array_5*1e3, color="teal",ls='dotted', label='CONUS Ge PPC (4 kg)')



# Plot astrophysical limits
astro_alpha = 0.1
plt.fill(hbstars[:,0]*1e9, hbstars[:,1]*0.367e-3, color="mediumpurple", alpha=astro_alpha)
plt.fill(cast[:,0]*1e9, cast[:,1]*0.367e-3, color="orchid", alpha=astro_alpha)
plt.fill_between(sn1987a_lower[:,0]*1e9, y1=sn1987a_lower[:,1]*0.367e-3, y2=sn1987a_upper[:,1]*0.367e-3, color="darkgoldenrod", alpha=astro_alpha)

# Plot existing limits
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
#plt.savefig('plots/alps_paper/axion_photon_limits_benchmark_x10bkg.pdf')
#plt.savefig('plots/alps_paper/axion_photon_limits_benchmark_x10bkg.png')

plt.show()


