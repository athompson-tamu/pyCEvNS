import numpy as np
import matplotlib.pyplot as plt

from pyCEvNS.flux import *
from pyCEvNS.axion import Axion


# Read in the photon flux
gamma_flux = np.genfromtxt("pyCEvNS/data/photon_flux_COHERENT_log_binned.txt")


ax = Axion(photon_rates=gamma_flux, axion_mass=1, axion_coupling=0.00001, target_mass=1000,
           target_z=20, target_photon_cross=100, detector_distance=8, min_decay_length=0)
ax.simulate()

result = ax.scatter_events(detector_number=2, detector_z=20, detection_time=20, threshold=0)

print(result)