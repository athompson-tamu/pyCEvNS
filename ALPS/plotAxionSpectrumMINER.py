import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline




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
  def photon_axion_cross(self, pgamma):
    ma = self.axion_mass
    it = 1 / (ma ** 2 * pgamma ** 2 - pgamma ** 4) + (ma ** 2 - 2 * pgamma ** 2) * np.arctanh(
      2 * pgamma * np.sqrt(-ma ** 2 + pgamma ** 2) / (ma ** 2 - 2 * pgamma ** 2)) / (
               2 * pgamma ** 3 * (-ma ** 2 + pgamma ** 2) ** 1.5)
    return 1 / 4 * self.axion_coupling ** 2 * 1 / 137 * self.target_z ** 2 * (
          pgamma ** 2 - self.axion_mass ** 2) ** 2 * it * self.form_factor()

  # probability of axion production through primakoff
  def axion_probability(self, pgamma):
    # target_n_gamma is in cm^2
    # assuming that target is thick enough and photon cross section is large enough that all photon is converted / absorbed
    cross_prim = self.photon_axion_cross(pgamma)
    return cross_prim / (cross_prim + (self.target_photon_cross / (100 * meter_by_mev) ** 2))

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
    decay_prob =  1 - np.exp(-self.detector_distance / meter_by_mev / axion_v / tau) \
      if self.detector_distance / decay_length < 100 else 1
    decay_past_shielding = np.exp(-self.min_decay_length / meter_by_mev / axion_v / tau) \
                           - np.exp(-self.detector_distance / meter_by_mev / axion_v / tau)
    self.photon_energy.append(energy)
    self.photon_weight.append(rate * prob * decay_past_shielding
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

# CsI

det_dis = 2.5
det_mass = 4
det_am = 123.8e3
det_z = 32
days = 1
det_area = (0.4)**2
det_thresh = 0
dru_limit = 0.01

"""
# Ge
det_dis = 2.5
det_mass = 4
det_am = 65.13e3
det_z = 32
days = 1
det_area = 0.2**2
det_thresh = 2.6
dru_limit = 0.1
"""

# conversion between units
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600*24

# axion parameters
axion_mass = 1 # MeV
axion_coupling = 1e-6


# Set the flux.
flux = miner # flux at reactor surface
flux[:,1] *= 100000000






# Plot fluxes.
axion_sim1 = MinerEstimate(flux, 1, 1e-8, 240e3, 90, 15e-24, det_dis, 0)
axion_sim2 = MinerEstimate(flux, 0.1, 1e-6, 240e3, 90, 15e-24, det_dis, 0)
axion_sim3 = MinerEstimate(flux, 0.1, 1e-6, 240e3, 90, 15e-24, det_dis, 0)
axion_sim1.simulate()
axion_sim2.simulate()
axion_sim3.simulate()

#axion_sim.scatter_events(det_mass*mev_per_kg/det_mass, det_z, 1, 1e-5)*3600*24
#axion_sim.photon_events(det_area, days, det_thresh)*3600*24
gamma_e1 = axion_sim1.photon_energy
gamma_w1 = axion_sim1.photon_weight
gamma_e2 = axion_sim2.photon_energy
gamma_w2 = axion_sim2.photon_weight
gamma_e3 = axion_sim3.photon_energy
gamma_w3 = axion_sim3.photon_weight
plt.plot(flux[:,0], flux[:,1], label="1 MW Reactor Flux")
plt.yscale('log')
plt.legend(framealpha=1, fontsize=13)
plt.xlim((0.0,10))
plt.ylabel(r"Flux [$s^{-1}$]", fontsize=13)
plt.xlabel(r"$E_\gamma$ [MeV]", fontsize=13)
plt.savefig("plots/miner_spectrum/miner_photon_flux.png")
plt.savefig("plots/miner_spectrum/miner_photon_flux.pdf")

plt.clf()


fig = plt.figure(figsize=(7,4))
ax = plt.subplot(111)
kg_25kev_to_dru = 25 * det_mass


sc_events2 = axion_sim2.scatter_events_binned(det_mass*mev_per_kg/det_am, det_z, days, 1e-5) * s_per_day #/ kg_25kev_to_dru
ph_events2 = axion_sim2.photon_events_binned(det_area, days, det_thresh) * s_per_day #/ kg_25kev_to_dru # dru conversion

sc_events3 = axion_sim3.scatter_events_binned(det_mass*mev_per_kg/det_am, det_z, days, 1e-5) * s_per_day #/ kg_25kev_to_dru # dru conversion
ph_events3 = axion_sim3.photon_events_binned(det_area, days, det_thresh) * s_per_day #/ kg_25kev_to_dru # dru conversion
bkg_x = [0.0, 1.0, 2.0, 2.59, 2.6]
bkg_y = [0.006, 0.006, 0.005, 0.005, 0.0]
x = np.linspace(0.0, 3.0, 100)
fit = UnivariateSpline(bkg_x, bkg_y)

fig, ax1 = plt.subplots()
ax1.plot(flux[:,0], flux[:,1], color='k', label="1 MW Reactor Flux")
ax1.set_yscale('log')
ax1.set_ylabel(r"Flux [$s^{-1}$]")
ax1.set_xlabel(r"Energy [MeV]")


#ax2 = ax1.twinx()
ax1.plot(gamma_e3, sc_events3, color="royalblue",label="ALP Scattering")
ax1.plot(gamma_e2, ph_events2, color="crimson",ls="dashed",label="ALP Decay")
#ax2.plot(x, fit(x), color="gray", label="Background")
#ax2.fill_between(x, fit(x), color="gray", alpha="0.2")
#ax2.set_yscale('log')
#ax2.set_ylabel("DRU", fontsize=13)
ax1.set_xlim((0.0,8))
ax1.set_ylim((0.1,1e21))







#plt.ylim((1e-3,1e0))

ax1.legend(framealpha=1, loc="upper right")
#ax2.legend(framealpha=1, loc="upper right")
plt.title(r"$g_{a\gamma\gamma} = 10^{-3}$ GeV$^{-1}$, $m_a = 100$ keV", loc="right")

plt.xlabel(r"$E_\gamma$ [MeV]", fontsize=13)
plt.tight_layout()
plt.savefig("plots/alps_paper/miner_events_spectrum.png")
plt.savefig("plots/alps_paper/miner_events_spectrum.pdf")