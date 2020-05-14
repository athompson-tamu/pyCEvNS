from .constants import *
from .helper import *

from numpy import log, log10, pi


# Define form factors
def _nuclear_ff(t, m, z, a):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    # a: number of nucleons
    return (2*m*z**2) / (1 + t / 164000*np.power(a, -2/3))**2

def _atomic_elastic_ff(t, m, z):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    b = 184*np.power(2.718, -1/2)*np.power(z, -1/3) / me
    return (z*t*b**2)**2 / (1 + t*b**2)**2

def _atomic_elastic_ff(t, m, z):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    b = 1194*np.power(2.718, -1/2)*np.power(z, -2/3) / me
    return (z*t*b**2)**2 / (1 + t*b**2)**2



# Directional axion production and detection
class Axion:
    def __init__(self, photon_rates, axion_mass, axion_coupling, target_mass, target_z,
                 target_photon_cross, detector_distance, detector_length):
        self.photon_rates = photon_rates # per second
        self.axion_mass = axion_mass # MeV
        self.axion_coupling = axion_coupling # MeV^-1
        self.target_mass = target_mass # MeV
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross # cm^2
        self.detector_distance = detector_distance # meter
        self.detector_length = detector_length
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        self.simulate()


    def form_factor(self):
        # place holder for now, included in the cross section
        return 1

    def screening(self):
        return 1.62 * np.power(self.axion_mass, 1.92) # taken from fit to PhysRevD.37.618

    def primakoff_production_xs(self, z, a):
        me = 0.511
        prefactor = (1 / 137 / 4) * (self.axion_coupling ** 2)
        return prefactor * ((z ** 2) * (np.log(184 * np.power(z, -1 / 3)) \
               + np.log(403 * np.power(a, -1 / 3) / me)) \
               + z * np.log(1194 * np.power(z, -2 / 3)))

    def primakoff_scattering_xs(self, energy, z):
        if energy < self.axion_mass:
            return

        beta = min(np.sqrt(energy**2 - self.axion_mass**2) / energy, 0.99999999999)
        chi = 1 #self.screening()
        prefactor = (1/137/2) * (self.axion_coupling * z) ** 2
        return prefactor * chi * (1 / beta) * ((1 + beta**2)/(2*beta) * np.log((1 + beta) / (1 - beta)) - 1)

    def branching_ratio(self):
        cross_prim = self.primakoff_production_xs(self.target_z, 2*self.target_z)
        return cross_prim / (cross_prim + (self.target_photon_cross / (100 * meter_by_mev) ** 2))

    def photon_axion_cross(self, pgamma):
        # Primakoff
        # def func(cs):
        #     t = self.axion_mass**2 + 2*(-pgamma**2+pgamma*np.sqrt(pgamma**2-self.axion_mass**2)*cs)
        #     return (1-cs**2)/t**2
        # return 1/4*self.axion_coupling**2*1/137*self.target_z**2*(pgamma**2-self.axion_mass**2)**2*quad(func, -1, 1)[0]*self.form_factor()
        ma = self.axion_mass
        it = 1/(ma**2*pgamma**2-pgamma**4)+(ma**2-2*pgamma**2)*np.arctanh(2*pgamma*np.sqrt(-ma**2+pgamma**2)/(ma**2-2*pgamma**2))/(2*pgamma**3*(-ma**2+pgamma**2)**1.5)
        return 1/4*self.axion_coupling**2*1/137*self.target_z**2*(pgamma**2-self.axion_mass**2)**2*it*self.form_factor()

    def simulate_single(self, energy, rate, nsamplings=1000):
        if energy <= 1.5*self.axion_mass:# or np.abs(2*pgamma*np.sqrt(-ma**2+pgamma**2)/(ma**2-2*pgamma**2))>=1:
            return
        prob = self.branching_ratio() # probability of axion conversion
        axion_p = np.sqrt(energy ** 2 - self.axion_mass ** 2)
        axion_v = min(axion_p / energy, 0.99999999999)
        axion_boost = energy / self.axion_mass
        tau = 64 * pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost  # lifetime
        decay_length = meter_by_mev * axion_v * tau
        decay_prob = 1 - np.exp(-self.detector_distance / meter_by_mev / axion_v / tau) \
            if self.detector_distance / decay_length < 100 else 1
        axion_decay = np.random.exponential(tau, nsamplings)  # draw from distribution of decay times
        axion_pos = axion_decay * axion_v  # x = t * v
        photon_cs = np.random.uniform(-1, 1, nsamplings)
        for i in range(nsamplings):
            photon1_momentum = np.array([self.axion_mass/2, self.axion_mass/2*np.sqrt(1-photon_cs[i]**2), 0, self.axion_mass/2*photon_cs[i]])
            photon1_momentum = lorentz_boost(photon1_momentum, np.array([0, 0, axion_v]))
            photon2_momentum = np.array([self.axion_mass/2, -self.axion_mass/2*np.sqrt(1-photon_cs[i]**2), 0, -self.axion_mass/2*photon_cs[i]])
            photon2_momentum = lorentz_boost(photon2_momentum, np.array([0, 0, axion_v]))
            r = axion_pos[i]
            pos = np.array([0, 0, r])
            # If axion decays outside detector sphere, see if either 1 or both photons make it
            # backward boosted to the detector
            if r > (self.detector_distance + self.detector_length) / meter_by_mev:
                threshold = np.sqrt(r**2 - (self.detector_distance / meter_by_mev)**2) / r
                cs1 = np.sum(-photon1_momentum[1:] * pos) / np.sqrt(np.sum(photon1_momentum[1:] ** 2) * r ** 2)
                cs2 = np.sum(-photon2_momentum[1:] * pos) / np.sqrt(np.sum(photon2_momentum[1:] ** 2) * r ** 2)
                if cs1 >= threshold:
                    self.photon_energy.append(photon1_momentum[0])
                    self.photon_weight.append(rate * prob / nsamplings)
                if cs2 >= threshold:
                    self.photon_energy.append(photon2_momentum[0])
                    self.photon_weight.append(rate * prob / nsamplings)
            elif r > self.detector_distance / meter_by_mev:  # else if we are inside, guarantee flux to the det
                self.photon_energy.append(photon1_momentum[0])
                self.photon_weight.append(rate*prob/nsamplings)
                self.photon_energy.append(photon2_momentum[0])
                self.photon_weight.append(rate*prob/nsamplings)
        self.axion_energy.append(energy)
        self.axion_weight.append(rate*prob*(1-decay_prob))

    def simulate(self, nsamplings=1000):
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        for f in self.photon_rates:
            self.simulate_single(f[0], f[1], nsamplings)

    def photon_events(self, detector_area, detection_time, threshold):
        res = 0
        for i in range(len(self.photon_energy)):
            if self.photon_energy[i] >= threshold:
                res += self.photon_weight[i]
        return res * detection_time * detector_area / (4*pi*self.detector_distance**2)

    def photon_events_binned(self, detector_area, detection_time, threshold):
        decay_photon_weight = []
        exposure = detection_time * detector_area / (4*pi*self.detector_distance**2)
        for i in range(len(self.photon_energy)):
            if self.photon_energy[i] >= threshold:
                decay_photon_weight.append(self.photon_weight[i] * exposure)
        return self.photon_energy, decay_photon_weight

    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        exposure = meter_by_mev**2 * detection_time * detector_number \
                   / (4 * pi * self.detector_distance ** 2)
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res += self.axion_weight[i] * self.primakoff_scattering_xs(self.axion_energy[i], detector_z)
        return res * exposure

    def scatter_events_binned(self, detector_number, detector_z, detection_time, threshold):
        scatter_photon_weight = []
        exposure = meter_by_mev**2 * detection_time * detector_number \
                   / (4 * pi * self.detector_distance ** 2)
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                scatter_photon_weight.append(exposure * self.axion_weight[i]
                                             * self.primakoff_scattering_xs(self.axion_energy[i], detector_z))
        return self.axion_energy, scatter_photon_weight



class IsotropicAxionFromPrimakoff:
  def __init__(self, photon_rates, axion_mass, axion_coupling, target_mass, target_z,
               target_photon_cross, detector_distance, detector_length):
    self.photon_rates = photon_rates  # per second
    self.axion_mass = axion_mass  # MeV
    self.axion_coupling = axion_coupling  # MeV^-1
    self.target_mass = target_mass  # MeV
    self.target_z = target_z
    self.target_photon_cross = target_photon_cross  # cm^2
    self.detector_distance = detector_distance  # meter
    self.detector_back = detector_length + detector_distance
    self.detector_length = detector_length
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
    return prefactor * ((z ** 2) * (np.log(184 * np.power(z, -1 / 3)) \
           + np.log(403 * np.power(a, -1 / 3) / me)) \
           + z * np.log(1194 * np.power(z, -2 / 3)))

  def primakoff_scattering_xs(self, energy, z):
    if energy < self.axion_mass:
      return 0

    beta = np.sqrt(energy ** 2 - self.axion_mass ** 2) / energy
    chi = 1
    prefactor = (1 / 137 / 2) * (self.axion_coupling * z) ** 2
    return prefactor * chi * (1 / beta) * ((1 + beta ** 2) / (2 * beta) * np.log((1 + beta) / (1 - beta)) - 1)

  def branching_ratio(self):
    cross_prim = self.primakoff_production_xs(self.target_z, 2 * self.target_z)
    return cross_prim / (cross_prim + (self.target_photon_cross / (100 * meter_by_mev) ** 2))

  # Convolute axion production and decay rates with a photon flux
  def simulate_single(self, energy, rate):
    prob = self.branching_ratio()
    axion_p = np.sqrt(energy ** 2 - self.axion_mass ** 2)
    axion_v = axion_p / energy
    axion_boost = energy / self.axion_mass
    tau = 64 * pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost
    decay_length = meter_by_mev * axion_v * tau
    lg_surv_prob =  -self.detector_distance / meter_by_mev / axion_v / tau
    decay_in_detector = 1 - np.exp(-self.detector_length / meter_by_mev / axion_v / tau)
    lg_wgt = np.log(rate) + np.log(prob) \
             + lg_surv_prob \
             + np.log(decay_in_detector) \
             - np.log(4 * pi * (self.detector_distance) ** 2)
    self.photon_energy.append(energy)
    self.photon_weight.append(np.exp(lg_wgt))
    self.axion_energy.append(energy)
    self.axion_weight.append(np.exp(lg_surv_prob) * rate * prob / (4*pi*self.detector_distance ** 2))

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
        res += self.axion_weight[i] * self.primakoff_scattering_xs(self.axion_energy[i], detector_z) \
               * detection_time * detector_number * meter_by_mev ** 2
    return res 

  def photon_events_binned(self, detector_area, detection_time, threshold):
    res = np.zeros(len(self.photon_weight))
    scale = detection_time * detector_area
    for i in range(len(self.photon_energy)):
      if self.photon_energy[i] >= threshold:
        res[i] = self.photon_weight[i]
    return res * scale

  def scatter_events_binned(self, detector_number, detector_z, detection_time, threshold):
    res = np.zeros(len(self.axion_weight))
    for i in range(len(self.axion_energy)):
      if self.axion_energy[i] >= threshold:
        res[i] = self.axion_weight[i] * self.primakoff_scattering_xs(self.axion_energy[i], detector_z) \
                 * detection_time * detector_number * meter_by_mev ** 2
    return res 



class IsotropicAxionFromCompton:
    def __init__(self, photon_rates, axion_mass, axion_coupling, target_mass, target_z,
                 target_photon_cross, detector_distance, detector_length):
        self.photon_rates = photon_rates  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_mass = target_mass  # MeV
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross  # cm^2
        self.detector_distance = detector_distance  # meter
        self.detector_length = detector_length
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        self.epem_energy = []
        self.epem_weight = []
        self.electron_energy = []
        self.electron_weight = []
        self.axion_scatter_cross = []
        self.simulate(1)

    def form_factor(self):
        # place holder for now, included in the cross section
        return 1

    def simulate_single(self, eg, rate):
        s = 2 * me * eg + me ** 2
        a = 1 / 137
        aa = self.axion_coupling ** 2 / 4 / pi
        ma = self.axion_mass

        ne = 50
        axion_energies = np.linspace(ma, eg, ne) # version 2
        de = (axion_energies[-1] - axion_energies[0]) / (ne - 1)
        axion_energies = (axion_energies[1:] + axion_energies[:-1]) / 2
        dde = self.AxionProductionXS(axion_energies, eg) * de
        cross_prod = np.sum(dde)
        cross_scatter = self.AxionElectronScatteringXS(axion_energies)
        if np.any(dde) < 0:
            return

        # Both photons and axions decrease with decay_prob, since we assume e+e- does not make it to the detector.
        for i in range(ne - 1):
            axion_prob = dde[i] * self.target_z / (dde[i] + (self.target_photon_cross / (100 * meter_by_mev) ** 2))
            surv_prob = self.AxionSurvProb(axion_energies[i])
            decay_prob = self.AxionDecayProb(axion_energies[i])
            self.axion_energy.append(axion_energies[i])
            self.axion_weight.append(surv_prob * rate * axion_prob * (dde[i] / cross_prod) / (4 * pi * self.detector_distance ** 2))
            self.epem_energy.append(axion_energies[i])
            self.epem_weight.append(decay_prob * rate * axion_prob * (dde[i] / cross_prod) / (4 * pi * self.detector_distance ** 2))
            self.axion_scatter_cross.append(cross_scatter[i])

    def AxionElectronHighEnergyDiffXS(self, ea, et):
        a = 1 / 137
        aa = self.axion_coupling ** 2 / 4 / pi
        prefact = a * aa * pi / 2 / me
        sigma = prefact * (et ** 2) / ((et ** 3) * (ea - et))
        return sigma

    def AxionElectronScatteringXS(self, ea):
        a = 1 / 137
        aa = self.axion_coupling ** 2 / 4 / pi
        prefact = a * aa * pi / 2 / me / ea**2
        sigma = prefact * 2 * ea * (-(2*ea * (3*ea + me)/(2 * ea + me)**2) + np.log(2 * ea / me + 1))
        return sigma

    def AxionElectronScatteringDiffXS(self, ea, et):
        # dSigma / dEt   electron kinetic energy
        # ea: axion energy
        # et: transferred electron energy = E_e - m_e.
        y = 2 * me * ea + self.axion_mass ** 2
        prefact = (1/137) * self.axion_coupling ** 2 / (4 * me ** 2)
        pa = np.sqrt(ea ** 2 - self.axion_mass ** 2)
        eg = ea - et
        return -(prefact / pa) * (1 - (8 * me * eg / y) + (12 * (me * eg / y) ** 2)
                                  - (32 * me * (pa * self.axion_mass) ** 2) * eg / (3 * y ** 3))

    def AxionProductionXS(self, ea, eg):
        # Differential cross-section dS/dE_a. gamma + e > a + e.
        a = 1 / 137
        aa = self.axion_coupling ** 2 / 4 / pi
        ma = self.axion_mass
        s = 2 * me * eg + me ** 2
        if np.sqrt(s) < me + ma:
            return 0
        x = (ma**2 / (2*eg*me)) - ea / eg + 1
        return (1 / eg) * pi * a * aa / (s - me ** 2) * (x / (1 - x) * (-2 * ma ** 2 / (s - me ** 2) ** 2
                                                                 * (s - me ** 2 / (1 - x) - ma ** 2 / x) + x))

    def AxionDecayProb(self, ea):
        # Decay the axions in flight to e+ e-.
        # Returns probability that it will decay inside the detector volume.
        axion_p = np.sqrt(ea ** 2 - self.axion_mass ** 2)
        axion_v = axion_p / ea
        axion_boost = ea / self.axion_mass
        tau = (8 * pi) / (self.axion_coupling ** 2 * self.axion_mass
                             * np.power(1 - 4 * (me / self.axion_mass) ** 2, 1 / 2)) \
              if 1 - 4 * (me / self.axion_mass) ** 2 > 0 else np.inf  # lifetime for a -> gamma gamma
        tau *= axion_boost
        return np.exp(-self.detector_distance / meter_by_mev / axion_v / tau) \
               * (1 - np.exp(-self.detector_length / meter_by_mev / axion_v / tau))
    
    def AxionSurvProb(self, ea):
        # Decay the axions in flight to e+ e-.
        # Returns probability that it will decay inside the detector volume.
        axion_p = np.sqrt(ea ** 2 - self.axion_mass ** 2)
        axion_v = axion_p / ea
        axion_boost = ea / self.axion_mass
        tau = (8 * pi) / (self.axion_coupling ** 2 * self.axion_mass
                             * np.power(1 - 4 * (me / self.axion_mass) ** 2, 1 / 2)) \
              if 1 - 4 * (me / self.axion_mass) ** 2 > 0 else np.inf  # lifetime for a -> gamma gamma
        tau *= axion_boost
        return np.exp(-self.detector_distance / meter_by_mev / axion_v / tau)

    def simulate(self, nsamplings=1000):
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        self.electron_energy = []
        self.electron_weight = []
        self.epem_energy = []
        self.epem_weight = []
        for f in self.photon_rates:
          self.simulate_single(f[0], f[1])

    def photon_events(self, detector_area, detection_time, threshold):
        res = 0
        for i in range(len(self.photon_energy)):
          if self.photon_energy[i] >= threshold:
            res += self.photon_weight[i]
        return res * detection_time * detector_area

    def electron_events_binned(self, nbins, detector_number, detector_z, detection_time, threshold):
        self.electron_energy = []
        self.electron_weight = []
        self.photon_weight = []
        self.photon_energy = []
        for i in range(len(self.axion_energy) - 1): # integrate over E_a
            Et_max = 2 * np.max(self.axion_energy[i]) ** 2 / (me + 2 * np.max(self.axion_energy[i]))
            Et = np.linspace(0, Et_max, nbins)  # electron energies
            delta_Et = (Et[-1] - Et[0]) / (nbins - 1)
            Et = (Et[1:] + Et[:-1]) / 2  # get bin centers

            # Get differential scattering rate
            dSigma_dEt = self.AxionElectronHighEnergyDiffXS(self.axion_energy[i], Et)
            if np.any(dSigma_dEt < 0):
                continue

            sigma = np.sum(dSigma_dEt) * delta_Et  # total cross-section

            # Fill in electrons..
            for j in range(Et.shape[0]-1): # Integrate over E_t
                if Et[j] < threshold:
                    continue
                if Et[j] > Et_max:
                    continue

                scatter_rate = self.axion_weight[i] * (dSigma_dEt[j] / sigma) * self.axion_scatter_cross[i] * delta_Et
                exposure = meter_by_mev ** 2 * detection_time * detector_number * detector_z
                self.electron_weight.append(scatter_rate * exposure)
                self.electron_energy.append(Et[j])
                self.photon_weight.append(scatter_rate * exposure)
                self.photon_energy.append(self.axion_energy[i]-Et[j])

        return np.sum(self.electron_weight), np.sum(self.photon_weight)


    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res += self.axion_weight[i] * self.axion_scatter_cross[i]  # approx scatter_xs = prod_xs
        return res * meter_by_mev ** 2 * detection_time * detector_number * detector_z

    def scatter_events_binned(self, detector_number, detector_z, detection_time, threshold):
        res = np.zeros(len(self.axion_weight))
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res[i] = self.axion_weight[i] * self.axion_scatter_cross[i]  # approx scatter_xs = prod_xs
        return res * meter_by_mev ** 2 * detection_time * detector_number * detector_z

    def pair_production_events(self, detector_area, detection_time, threshold):
        res = 0
        for i in range(len(self.electron_energy)):
            if self.axion_energy[i] >= threshold:
                res += self.epem_weight[i]
        return res * detection_time * detector_area