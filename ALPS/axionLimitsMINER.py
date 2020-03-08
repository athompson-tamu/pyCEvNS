import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy import signal




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
    decay_prob =  1 - np.exp(-self.min_decay_length / meter_by_mev / axion_v / tau) \
      if self.detector_distance / decay_length < 100 else 1
    decay_in_detector = 1 - np.exp(-(self.detector_distance-self.min_decay_length) / meter_by_mev / axion_v / tau)
    self.photon_energy.append(energy)
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


# Declare constants.
# conversion between units
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600*24


miner[:, 1] *= 100000000
flux = miner

mass_array = np.logspace(-6, 1, 300)  # Mass array to test

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
    bkg = bkg_dru * days * det_mass * 500

    coupling_array = np.zeros_like(mass_array)  # Coupling array to test

    photon_gen = MinerEstimate(flux, 1, 1e-6, 240e3, 90, 15e-24, det_dis, 0.9*det_dis)
    # Axion decay regime
    for i in range(mass_array.shape[0]):
        lo = -12
        hi = -1
        photon_gen.axion_mass = mass_array[i]
        while hi - lo > 0.002:
            mid = (hi+lo)/2
            photon_gen.axion_coupling = 10**mid
            photon_gen.simulate()
            ev = photon_gen.photon_events(det_area, days, det_thresh) * s_per_day
            ev += photon_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days, det_thresh) * s_per_day
            sig = ev / np.sqrt(ev + bkg)
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

rerun = True
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


# Get CCM limits
upper = np.genfromtxt("limits/ccm/upper_limit.txt")
removed = np.genfromtxt("limits/ccm/removed_limit.txt")
scatter = np.genfromtxt("limits/ccm/scatter_limit.txt")
scatter_trimmed = np.genfromtxt("limits/ccm/scatter_limit_trimmed.txt")


upper_ccm = np.flip(np.flip(upper[4:,:],axis=1))
upper_ccm = upper_ccm[(upper_ccm[:,1] > 0)]
fit = np.poly1d(np.polyfit(np.log10(upper_ccm[:,0]), np.log10(upper_ccm[:,1]), 1))
total_limit_ccm = np.vstack((scatter[:-2,:], removed, upper_ccm))
total_limit_ccm = total_limit_ccm[(total_limit_ccm[:,1] > 0)]
total_limit_ccm = np.vstack((total_limit_ccm, [0.1, 10**fit(-1)]))

upper = np.genfromtxt("limits/lar_610/upper_limit.txt")
removed = np.genfromtxt("limits/lar_610/removed_limit.txt")
scatter = np.genfromtxt("limits/lar_610/scatter_limit.txt")
scatter_trimmed = np.genfromtxt("limits/lar_610/scatter_limit_trimmed.txt")

upper_coh = np.flip(np.flip(upper[4:,:],axis=1))
upper_coh = upper_coh[(upper_coh[:,1] > 0)]
fit = np.poly1d(np.polyfit(np.log10(upper_coh[:,0]), np.log10(upper_coh[:,1]), 1))
total_limit_coh = np.vstack((scatter[:-2,:], removed, upper_coh))
total_limit_coh = total_limit_coh[(total_limit_coh[:,1] > 0)]
total_limit_coh = np.vstack((total_limit_coh, [0.1, 10**fit(-1)]))


# Use filtering?
use_savgol = False
if use_savgol:
    coup_array_1 = signal.savgol_filter(coup_array_1, 15, 3)
    coup_array_4 = signal.savgol_filter(coup_array_4, 15, 3)
    coup_array_5 = signal.savgol_filter(coup_array_5, 15, 3)
    coup_array_6 = signal.savgol_filter(coup_array_6, 15, 3)



plt.clf()

fig, ax = plt.subplots()
ax.plot(mass_array*1e6, coup_array_1*1e3, color="crimson", label='MINER Ge (4 kg)')
#ax.plot(mass_array, coup_array_3*1e3, color="crimson", ls='dashed', label='MINER CsI (2 ton)')
ax.plot(mass_array*1e6, coup_array_6*1e3, color="navy", ls='dashed', label=r'NUCLEUS CaWO$_4$(Al$_2$O$_3$) (0.01 kg)')
ax.plot(mass_array*1e6, coup_array_4*1e3, color="orange",ls='dashdot',label='CONNIE Si Skipper CCD (0.1 kg)')
ax.plot(mass_array*1e6, coup_array_5*1e3, color="teal",ls='dotted', label='CONUS Ge PPC (4 kg)')
#plt.plot(mass_array_2, coup_array_2*1e3, label='MINER CsI (200 kg)', color="crimson", ls='dashed')
#plt.plot(mass_array_3, coup_array_3*1e3, label='MINER CsI (2 ton)', color="crimson", ls='dotted')
#ax.plot(total_limit_coh[:,0]*1e6, total_limit_coh[:,1]*1e3,
 #       label="COHERENT LAr (610 kg, 3 years)",color="crimson", ls='dashed')
#ax.plot(total_limit_ccm[:,0]*1e6, total_limit_ccm[:,1]*1e3,
 #       label="CCM LAr (10 ton, 3 years)",color="crimson", ls="dotted")


# Plot existing limits
ax.fill(beam[:,0], beam[:,1], label='Beam Dump', color="b", alpha=0.7)
ax.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
         color="orange", label=r'$e^+e^-\rightarrow inv.+\gamma$', alpha=0.7)
ax.fill(lep[:,0], lep[:,1], label='LEP', color="green", alpha=0.7)
#ax.fill(np.hstack((lsw[:,0], np.min(lsw[:,0]))), np.hstack((lsw[:,1], np.max(lsw[:,1]))), label='LSW', alpha=0.5)
ax.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
         color="yellow", label='NOMAD', alpha=0.7)
plt.legend(loc="upper left", framealpha=1, fontsize=8)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,1e8)
plt.ylim(3e-7,1)
plt.xlabel('$m_a$ [eV]', fontsize=13)
plt.ylabel('$g_{a\gamma\gamma}$ [GeV$^{-1}$]', fontsize=13)
plt.tight_layout()
plt.savefig('plots/alps_paper/axion_photon_limits_benchmark.pdf')
plt.savefig('plots/alps_paper/axion_photon_limits_benchmark.png')


