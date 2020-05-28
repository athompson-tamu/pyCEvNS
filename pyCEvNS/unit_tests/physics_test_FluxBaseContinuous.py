from pyCEvNS.events import *
from pyCEvNS.flux import *

from unittest import TestCase, main




# Currently, DMFluxIsoPhoton formally is a child of FluxBaseContinuous
# but works as a DMFlux.
# Goal: compare (DMFlux, DMEventsGen) to (FluxBaseContinuous child, DMNucleusElasticVector)
# Eradicate DMFlux and DMEventsGen if slower.


# set common constants
exposure = 100000
dm_mass = 25
dark_photon_mass = 75
mediator_mass = 75
energy_edges = np.arange(0.02, 0.12, 0.005) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.linspace(0.0, 0.4, 2) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

brem_photons = np.genfromtxt("data/ccm/brem.txt")
brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=dark_photon_mass, coupling=1,
                            dark_matter_mass=dm_mass, pot_sample=1e5, pot_mu=0.145,
                            pot_sigma=0.05, sampling_size=100, life_time=0.0001,
                            detector_distance=20, brem_suppress=True, verbose=False)

brem_flux.simulate()



# DMEventsGen style
dm_gen = DmEventsGen(dark_photon_mass=dark_photon_mass, dark_matter_mass=dm_mass,
                     life_time=0.001, expo=exposure, detector_type='ar')
dm_gen.fx = brem_flux
dm_events_gen_results = dm_gen.events(mediator_mass, 1, energy_edges, timing_edges, channel="nucleus")
DME_obs = dm_events_gen_results[0]
DME_energies = dm_events_gen_results[1][:,0]


# FluxBaseContinuous Style
det = Detector("ar", eff_coherent)
dm_gen_new = DMNucleusElasticVector(epsilon_dm=1/e_charge, epsilon_q=1/e_charge, mediator_mass=mediator_mass)
dm_nuc_elastic_vector_obs = np.empty_like(energy_bins)
for i in range(energy_edges.shape[0]-1):
    dm_nuc_elastic_vector_obs[i] = dm_gen_new.events(energy_edges[i], energy_edges[i+1], flux=brem_flux, detector=det, exposure=exposure)

# perfect match up!




import numpy as np
import matplotlib.pyplot as plt

density=False
plt.hist(DME_energies, weights=DME_obs, bins=energy_bins, histtype='step', density=density, ls='dashed', label="DMEventsGen")
plt.hist(energy_bins, weights=dm_nuc_elastic_vector_obs, bins=energy_bins, histtype='step', density=density, label="DMNucleusElasticVector")
plt.legend()

plt.show()



# speed test
from pstats import Stats
from cProfile import Profile

profiler = Profile()

def get_events_DMNEV():
    for i in range(energy_edges.shape[0]-1):
        dm_nuc_elastic_vector_obs[i] = dm_gen_new.events(energy_edges[i], energy_edges[i+1], flux=brem_flux, detector=det, exposure=exposure)

def get_events_DMEG():
    dm_gen.events(mediator_mass, 1, energy_edges, timing_edges, channel="nucleus")


profiler.runcall(lambda: get_events_DMEG())
stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_callers()
