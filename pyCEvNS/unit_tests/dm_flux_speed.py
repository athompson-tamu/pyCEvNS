from pstats import Stats
from cProfile import Profile
import numpy as np

from pyCEvNS.constants import *


from pyCEvNS.flux import DMFluxIsoPhoton, DMFluxFromPi0Decay, DMFluxFromPiMinusAbsorption



brem_photons = np.genfromtxt("data/ccm/brem.txt")
brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=75, coupling=1,
                            dark_matter_mass=25, pot_sample=1e5, pot_mu=0.145,
                            pot_sigma=0.05, sampling_size=100, life_time=0.0001,
                            detector_distance=20, brem_suppress=True, verbose=False)

pim_rate=0.45
pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=75, coupling_quark=1,
                                        dark_matter_mass=25,
                                        pion_rate=pim_rate, detector_distance=19.3,
                                        life_time=0.0001)
Pi0Info = np.genfromtxt("data/coherent/Pi0_Info.txt")
pion_energy = Pi0Info[:,4] - massofpi0
pion_azimuth = np.arccos(Pi0Info[:,3] / np.sqrt(Pi0Info[:,1]**2 + Pi0Info[:,2]**2 + Pi0Info[:,3]**2))
pion_cos = np.cos(np.pi/180 * Pi0Info[:,0])
pion_flux = np.array([pion_azimuth, pion_cos, pion_energy])
pion_flux = pion_flux.transpose()
pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux,
                              dark_photon_mass=75, coupling_quark=1,
                              dark_matter_mass=25, detector_distance=19.3,
                              life_time=0.0001)

profiler = Profile()



profiler.runcall(lambda: pim_flux.simulate())
stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_callers()




