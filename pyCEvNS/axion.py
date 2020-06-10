from .constants import *
from .helper import *

import time
import math
import multiprocessing as multi

from numpy import log, log10, pi, exp, sin, cos, sin, sqrt, arccos
from scipy.special import exp1
from scipy.integrate import quad, cumtrapz
from scipy.optimize import fsolve

import mpmath as mp
from mpmath import mpmathify, fsub, fadd
mp.dps = 15



# Define cross-sections
def primakoff_production_xs(energy, z, a, ma, g): #Tsai, '86
    if energy < ma:
        return 0
    me = 0.511
    prefactor = (1 / 137 / 4) * (g ** 2)
    return prefactor * ((z ** 2) * (np.log(184 * np.power(z, -1 / 3)) \
        + np.log(403 * np.power(a, -1 / 3) / me)) \
        + z * np.log(1194 * np.power(z, -2 / 3)))


def primakoff_scattering_xs(energy, z, ma, g):  # Avignone '87
    if energy < ma:
        return 0
    beta = min(np.sqrt(energy ** 2 - ma ** 2) / energy, 0.999999)
    chi = _screening(energy, ma)
    prefactor = (1 / 137 / 2) * (g * z) ** 2
    return prefactor * chi * (1 / beta) * ((1 + beta ** 2) / (2 * beta) * np.log((1 + beta) / (1 - beta)) - 1)


def primakoff_production_diffxs(theta, energy, z, ma, g=1):
    if energy < ma:
        return 0
    pa = sqrt(energy**2 - ma**2)
    t = energy*(energy - pa*cos(theta)) + ma**2
    ff = 1 #_nuclear_ff(t, ma, z, 2*z)
    return kAlpha * (g * z * ff * (ma * pa)**2 / t)**2 * sin(theta)**3 / 4


def primakoff_production_cdf(theta, energy, z, ma):
    norm = quad(primakoff_production_diffxs, 0, pi, args=(energy,z,ma))[0]
    if norm == 0:
        return 0
    return quad(primakoff_production_diffxs,
                0, theta, args=(energy,z,ma))[0] / norm


# Quantile (inverse CDF) for the Primakoff production angular distribution
def primakoff_prod_quant(energy, z, ma, u):
    _theta_list = np.linspace(0, pi, 35)
    norm = quad(primakoff_production_diffxs, 0, pi, args=(energy,z,ma))[0]
    if norm == 0:
        return math.nan
    
    cdf_list = np.empty_like(_theta_list)
    for i in range(0, _theta_list.shape[0]-1):
        cdf_list[i] = quad(primakoff_production_diffxs,
                           _theta_list[i], _theta_list[i+1], args=(energy,z,ma))[0]
    cdf_list = np.cumsum(cdf_list) / norm
    return np.interp(u, cdf_list, _theta_list)


def primakoff_production_weights(theta_edges, energy, z, ma):
    norm = quad(primakoff_production_diffxs, 0, pi, args=(energy,z,ma))[0]
    if norm == 0:
        return math.nan
    binned_diff_xs = np.zeros(len(theta_edges)-1)
    for i in range(0, theta_edges.shape[0]-1):
        binned_diff_xs[i] = quad(primakoff_production_diffxs,
                           theta_edges[i], theta_edges[i+1], args=(energy,z,ma))[0]
    return binned_diff_xs / norm


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

def _screening(e, ma):
    if ma == 0:
        return 0
    r0 = 1/0.001973  # 0.001973 MeV A -> 1 A (Ge) = 1/0.001973
    x = (r0 * ma**2 / (4*e))**2
    numerator = 2*log(2*e/ma) - 1 - exp(-x) * (1 - exp(-x)/2) + (x + 0.5)*exp1(2*x) - (1+x)*exp1(x)
    denomenator = 2*log(2*e/ma) - 1
    return numerator / denomenator
    


# Directional axion production from beam-produced photon distribution
class PrimakoffAxionFromBeam:
    def __init__(self, photon_rates=[1.,1.,0.], target_mass=240e3, target_z=90, target_photon_cross=15e-24,
                 detector_distance=4., detector_length=0.2, detector_area=0.04, axion_mass=0.1, axion_coupling=1e-3):
        self.photon_rates = photon_rates  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_mass = target_mass  # MeV
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross  # cm^2
        self.det_dist = detector_distance  # meter
        self.det_back = detector_length + detector_distance
        self.det_length = detector_length
        self.det_area = detector_area
        self.axion_energy = []
        self.axion_angle = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.gamma_sep_angle = []
    
    def det_sa(self):
        return np.arctan(sqrt(self.det_area / 4 / pi) / self.det_dist)

    def branching_ratio(self, energy):
        cross_prim = primakoff_production_xs(energy, self.target_z, 2*self.target_z,
                                             self.axion_mass, self.axion_coupling)
        return cross_prim / (cross_prim + (self.target_photon_cross / (100 * meter_by_mev) ** 2))
    
    def get_beaming_angle(self, v):
        return np.arcsin(sqrt(1-v**2))
    
    def theta_z(self, theta, cosphi, theta_gamma):
        return abs(arccos(sin(theta_gamma)*cosphi*sin(theta) + cos(theta_gamma)*cos(theta)))
    
    def wgt_grid(self, w1, w2):
        return w1*w2

    def simulate_single(self, photon, nsamples=1):
        #t1 = time.time()
        data_tuple = ([], [], [], [], [])
        
        if photon[1] < self.axion_mass:
            return data_tuple
        unit_support = np.ones(nsamples)
        rate = photon[0]
        e_gamma = photon[1]
        theta_gamma = photon[2]
        
        # Draw axion primakoff scattering angle
        #u_list = np.random.uniform(0,1,nsamples)
        theta_edges = np.linspace(0,self.det_sa(),nsamples+1)
        theta_a = (theta_edges[1:]+theta_edges[:-1])/2

        #cosphi_axion = np.random.uniform(-1, 1, nsamples)  # disk around photon
        
        #theta_a = primakoff_prod_quant(e_gamma, self.target_z, self.axion_mass, u_list) # gamma-axion separation
        theta_wgts = primakoff_production_weights(theta_edges,e_gamma,self.target_z,self.axion_mass)
        #theta_az = abs(arccos(sin(theta_gamma)*cosphi_axion*sin(theta_edges) + cos(theta_gamma)*cos(theta_edges)))

        # Get axion Lorentz transformations and kinematics
        br = self.branching_ratio(e_gamma)
        axion_p = sqrt(e_gamma** 2 - self.axion_mass ** 2)
        axion_v = axion_p / e_gamma
        axion_boost = e_gamma / self.axion_mass
        tau = 64 * pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost
        
        # Get decay and survival probabilities
        surv_prob =  mp.exp(-self.det_dist / meter_by_mev / axion_v / tau)
        decay_in_detector = fsub(1,mp.exp(-self.det_length / meter_by_mev / axion_v / tau))
        in_solid_angle = (self.det_sa() >= theta_a)  # array of truth values
        #t2 = time.time()

        # Push back lists and weights
        data_tuple[0].extend(e_gamma * unit_support) # elastic limit
        data_tuple[1].extend(theta_a)
        data_tuple[2].extend(rate * br * surv_prob * theta_wgts * in_solid_angle)  # scatter weights
        data_tuple[3].extend(rate * br * surv_prob * decay_in_detector * theta_wgts * in_solid_angle)  #decay weights
        data_tuple[4].extend(unit_support * np.arcsin(sqrt(1-axion_v**2))) # beaming formula for iso decay
        
        #t3 = time.time()
        #print("MC = ", t2 - t1)
        #print("loop = ", t3 - t2)
        
        return data_tuple
    
    def simulate(self, nsamples=10):
        t1 = time.time()
        self.axion_energy = []
        self.axion_angle = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []
        self.gamma_sep_angle = []
        
        with multi.Pool(max(1, multi.cpu_count()-1)) as pool:
            ntuple = pool.map(self.simulate_single, [f for f in self.photon_rates])
            pool.close()
        
        t2 = time.time()
        
        for tup in ntuple:
            self.axion_energy.extend(tup[0])
            self.axion_angle.extend(tup[1])
            self.scatter_axion_weight.extend(tup[2])
            self.decay_axion_weight.extend(tup[3])
            self.gamma_sep_angle.extend(tup[4])     
    
    def decay_events(self, detection_time, threshold):
        res = 0
        for i in range(len(self.decay_axion_weight)):
            if self.axion_energy[i] >= threshold:
                res += self.decay_axion_weight[i]
        return res * detection_time

    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        for i in range(len(self.scatter_axion_weight)):
            if self.axion_energy[i] >= threshold:
                res += self.scatter_axion_weight[i] \
                    * primakoff_scattering_xs(self.axion_energy[i], detector_z,
                                              self.axion_mass, self.axion_coupling) \
                    * detection_time * detector_number * meter_by_mev ** 2
        return res
    
    def detect(self, detector_number, detector_z, detection_time, threshold):
        for i in range(len(self.scatter_axion_weight)):
            if self.axion_energy[i] >= threshold:
                self.scatter_axion_weight[i] *= self.scatter_axion_weight[i] \
                    * primakoff_scattering_xs(self.axion_energy[i], detector_z,
                                              self.axion_mass, self.axion_coupling) \
                    * detection_time * detector_number * meter_by_mev ** 2



class IsotropicAxionFromPrimakoff:
    def __init__(self, photon_rates=[1,1], axion_mass=0.1, axion_coupling=1e-4,
                 target_mass=240e3, target_z=90, target_photon_cross=15e-24, detector_distance=4,
                 detector_length=0.2, detector_area=20):
        self.photon_rates = photon_rates  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_mass = target_mass  # MeV
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross  # cm^2
        self.detector_distance = detector_distance  # meter
        self.detector_back = detector_length + detector_distance
        self.detector_length = detector_length
        self.det_area = detector_area
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        self.axion_decay_prob = []
        self.axion_surv_prob = []
        self.axion_velocity = []
        self.simulate()


    def branching_ratio(self, energy):
        cross_prim = primakoff_production_xs(energy, self.target_z, 2 * self.target_z,
                                             self.axion_mass, self.axion_coupling)
        return cross_prim / (cross_prim + (self.target_photon_cross / (100 * meter_by_mev) ** 2))

    # Convolute axion production and decay rates with a photon flux
    def simulate_single(self, energy, rate):
        if energy <= self.axion_mass:
            return
        prob = mpmathify(self.branching_ratio(energy))
        axion_p = mp.sqrt(energy ** 2 - self.axion_mass ** 2)
        axion_v = mpmathify(axion_p / energy)
        self.axion_velocity.append(axion_v)
        axion_boost = mpmathify(energy / self.axion_mass)
        tau = mpmathify(64 * pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost)
        surv_prob =  mp.exp(-self.detector_distance / meter_by_mev / axion_v / tau)
        decay_in_detector = fsub(1,mp.exp(-self.detector_length / meter_by_mev / axion_v / tau))
        self.axion_decay_prob.append(decay_in_detector)
        self.axion_surv_prob.append(surv_prob)
        self.photon_energy.append(energy)
        self.photon_weight.append(rate * prob * surv_prob * decay_in_detector / (4*pi*self.detector_distance ** 2))
        self.axion_energy.append(energy)
        self.axion_weight.append(surv_prob * rate * prob / (4*pi*self.detector_distance ** 2))

    # Loops over photon flux and fills the photon and axion energy arrays.
    def simulate(self):
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.axion_weight = []
        self.axion_decay_prob = []
        self.axion_surv_prob = []
        self.axion_velocity = []
        for f in self.photon_rates:
            self.simulate_single(f[0], f[1])

    def decay_events(self, detection_time, threshold):
        res = 0
        for i in range(len(self.photon_energy)):
            if self.photon_energy[i] >= threshold:
                res += self.photon_weight[i]
        return res * detection_time * self.det_area

    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res += self.axion_weight[i] * primakoff_scattering_xs(self.axion_energy[i], detector_z,
                                                                      self.axion_mass, self.axion_coupling) \
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
                res[i] = self.axion_weight[i] * primakoff_scattering_xs(self.axion_energy[i], detector_z,
                                                                        self.axion_mass, self.axion_coupling) \
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